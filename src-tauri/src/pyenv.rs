//! The Python used for exercises and notebooks. Three options, in this order:
//!   1. a custom interpreter path from Settings,
//!   2. the app's own uv environment in ~/.study-quest/.venv (Settings → "Set up with uv"),
//!   3. in development, the repo's ./.venv (`uv sync`), then the system Python.

use std::path::{Path, PathBuf};
use std::process::Command;
use tauri::AppHandle;

use crate::platform;

/// The app-managed virtual environment.
pub fn venv_dir() -> PathBuf {
    platform::home().join(".study-quest").join(".venv")
}

/// The interpreter inside a venv.
pub fn venv_python(venv: &Path) -> PathBuf {
    if cfg!(windows) { venv.join("Scripts").join("python.exe") } else { venv.join("bin").join("python") }
}

/// GUI apps don't get the shell's PATH, so look where the uv installers put it.
fn find_uv() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "uv.exe" } else { "uv" };
    let home = platform::home();
    let mut candidates = vec![home.join(".local").join("bin").join(exe), home.join(".cargo").join("bin").join(exe)];
    if cfg!(target_os = "macos") {
        candidates.push(PathBuf::from("/opt/homebrew/bin/uv"));
        candidates.push(PathBuf::from("/usr/local/bin/uv"));
    } else if cfg!(windows) {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            candidates.push(PathBuf::from(local).join("Microsoft").join("WinGet").join("Links").join(exe));
        }
    } else {
        candidates.push(PathBuf::from("/usr/local/bin/uv"));
        candidates.push(PathBuf::from("/usr/bin/uv"));
    }
    candidates.into_iter().find(|p| p.exists()).or_else(|| {
        // Last try: whatever is on PATH.
        platform::quiet(Command::new(exe).arg("--version")).output().ok().filter(|o| o.status.success()).map(|_| PathBuf::from(exe))
    })
}

const PROBE: &str = r#"import json, sys, importlib.util as u
print(json.dumps({"version": sys.version.split()[0], "executable": sys.executable,
  "packages": {m: u.find_spec(m) is not None for m in ("marimo", "numpy", "matplotlib", "torch")}}))"#;

/// Run the interpreter and report its version and which packages it has.
pub fn probe(py: &Path) -> serde_json::Value {
    match platform::quiet(Command::new(py).args(["-c", PROBE])).output() {
        Ok(o) if o.status.success() => serde_json::from_slice(&o.stdout)
            .unwrap_or_else(|_| serde_json::json!({"ok": false, "error": "unexpected output"}))
            .as_object()
            .map(|m| { let mut m = m.clone(); m.insert("ok".into(), true.into()); serde_json::Value::Object(m) })
            .unwrap(),
        Ok(o) => serde_json::json!({"ok": false, "error": String::from_utf8_lossy(&o.stderr).lines().last().unwrap_or("failed").to_string()}),
        Err(e) => serde_json::json!({"ok": false, "error": format!("can't run {}: {e}", py.display())}),
    }
}

/// Settings → "Test": check a custom interpreter before using it.
#[tauri::command]
pub async fn check_python(path: String) -> serde_json::Value {
    let p = PathBuf::from(path.trim());
    if path.trim().is_empty() {
        return serde_json::json!({"ok": false, "error": "enter a path"});
    }
    probe(&p)
}

/// Settings → "Set up with uv": create ~/.study-quest/.venv and install the pinned packages (optionally PyTorch).
/// Returns the tail of uv's output.
#[tauri::command]
pub async fn setup_python(app: AppHandle, with_torch: bool) -> Result<String, String> {
    let uv = find_uv().ok_or_else(|| {
        if cfg!(windows) {
            "uv isn't installed. Install it from https://docs.astral.sh/uv/ (e.g. `winget install astral-sh.uv`), then try again.".to_string()
        } else {
            "uv isn't installed. Install it from https://docs.astral.sh/uv/ (e.g. `brew install uv` or the install script), then try again.".to_string()
        }
    })?;
    let venv = venv_dir();
    let mut log = String::new();
    let mut run = |args: &[&str]| -> Result<(), String> {
        let out = platform::quiet(Command::new(&uv).args(args)).output().map_err(|e| e.to_string())?;
        log.push_str(&String::from_utf8_lossy(&out.stdout));
        log.push_str(&String::from_utf8_lossy(&out.stderr));
        if out.status.success() { Ok(()) } else { Err(format!("uv {} failed:\n{}", args.join(" "), tail(&log))) }
    };
    let venv_s = venv.to_string_lossy().to_string();
    if !venv_python(&venv).exists() {
        // uv downloads a managed Python 3.12 if the system has none.
        run(&["venv", "--python", "3.12", &venv_s])?;
    }
    let req = crate::resource(&app, "../requirements.txt", "requirements.txt");
    let py = venv_python(&venv).to_string_lossy().to_string();
    run(&["pip", "install", "--python", &py, "-r", &req.to_string_lossy()])?;
    if with_torch {
        run(&["pip", "install", "--python", &py, "torch==2.14.1"])?;
    }
    Ok(tail(&log))
}

fn tail(s: &str) -> String {
    let lines: Vec<&str> = s.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(12)..].join("\n")
}
