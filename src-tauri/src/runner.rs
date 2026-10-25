//! Runs playground code with the local Python: tests through harness.py, or free-form scratch code.
//! Code runs as the user, on their own machine, in a temporary directory with a time limit.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::AppHandle;

#[derive(serde::Serialize)]
pub struct RunOutput {
    stdout: String,
    stderr: String,
    timed_out: bool,
    ms: u128,
}

fn temp_dir() -> std::path::PathBuf {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("study_quest_{}_{}", std::process::id(), n))
}

fn run_python(app: &AppHandle, dir: &Path, args: &[&str], limit: Duration) -> Result<RunOutput, String> {
    let py = crate::python_path(app);
    let start = Instant::now();
    let mut cmd = Command::new(&py);
    crate::platform::quiet(&mut cmd);
    let mut child = cmd
        .args(args)
        .current_dir(dir)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("MPLBACKEND", "Agg")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start {}: {e}. Set the Python path in Settings.", py.display()))?;
    let mut out = child.stdout.take().unwrap();
    let mut err = child.stderr.take().unwrap();
    let t_out = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = out.read_to_string(&mut s);
        s
    });
    let t_err = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = err.read_to_string(&mut s);
        s
    });
    let mut timed_out = false;
    loop {
        if child.try_wait().map_err(|e| e.to_string())?.is_some() {
            break;
        }
        if start.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            timed_out = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let clip = |s: String| if s.len() > 20_000 { s[s.len() - 20_000..].to_string() } else { s };
    Ok(RunOutput {
        stdout: clip(t_out.join().unwrap_or_default()),
        stderr: clip(t_err.join().unwrap_or_default()),
        timed_out,
        ms: start.elapsed().as_millis(),
    })
}

#[tauri::command]
pub async fn run_tests(app: AppHandle, code: String, tests: String) -> Result<RunOutput, String> {
    let dir = temp_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("code.py"), code).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("tests.py"), tests).map_err(|e| e.to_string())?;
    let harness = crate::resource(&app, "resources/harness.py", "harness.py");
    let h = harness.to_string_lossy().to_string();
    let res = run_python(&app, &dir, &[&h, "code.py", "tests.py"], Duration::from_secs(20));
    let _ = std::fs::remove_dir_all(&dir);
    res
}

#[tauri::command]
pub async fn run_scratch(app: AppHandle, code: String) -> Result<RunOutput, String> {
    let dir = temp_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("scratch.py"), code).map_err(|e| e.to_string())?;
    let res = run_python(&app, &dir, &["scratch.py"], Duration::from_secs(30));
    let _ = std::fs::remove_dir_all(&dir);
    res
}
