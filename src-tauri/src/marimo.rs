//! One local marimo server over the notebooks folder, bound to 127.0.0.1 with a random access token.
//! marimo is started as `python -m marimo` in its own process group (Unix) / with tree-kill (Windows), so stopping it
//! also stops every notebook kernel it started. Stopped when the window closes, when the app quits, and at next
//! launch if a crash left it running.

use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager};

use crate::platform;
use crate::AppState;

pub struct Server {
    child: Child,
    port: u16,
    token: String,
}

fn random_token() -> Result<String, String> {
    let mut buf = [0u8; 16];
    getrandom::fill(&mut buf).map_err(|e| format!("no secure randomness: {e}"))?;
    Ok(buf.iter().map(|b| format!("{b:02x}")).collect())
}

fn free_port() -> Result<u16, String> {
    let l = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    Ok(l.local_addr().map_err(|e| e.to_string())?.port())
}

fn encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'/' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn app_dir() -> PathBuf {
    platform::home().join(".study-quest")
}
fn pid_file() -> PathBuf {
    app_dir().join("marimo.pid")
}
fn log_file() -> PathBuf {
    app_dir().join("marimo.log")
}

// ---------------------------------------------------------------- process tree control

/// Put the child in its own group so the whole tree can be signalled (Unix); hide its console (Windows).
fn isolate(cmd: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000 | 0x0000_0200); // CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    }
}

fn alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        Command::new("kill").args(["-0", "--", &format!("-{pid}")]).stderr(Stdio::null()).status().map_or(false, |s| s.success())
    }
    #[cfg(windows)]
    {
        platform::quiet(Command::new("tasklist").args(["/FI", &format!("PID eq {pid}"), "/NH"]))
            .output()
            .map_or(false, |o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
    }
}

/// Ask the tree to exit, give it up to 3 s to save, then force it.
fn stop_tree(pid: u32) {
    #[cfg(unix)]
    let soft = |_: ()| { let _ = Command::new("kill").args(["-TERM", "--", &format!("-{pid}")]).stderr(Stdio::null()).status(); };
    #[cfg(unix)]
    let hard = |_: ()| { let _ = Command::new("kill").args(["-KILL", "--", &format!("-{pid}")]).stderr(Stdio::null()).status(); };
    #[cfg(windows)]
    let soft = |_: ()| { let _ = platform::quiet(Command::new("taskkill").args(["/PID", &pid.to_string(), "/T"])).status(); };
    #[cfg(windows)]
    let hard = |_: ()| { let _ = platform::quiet(Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"])).status(); };
    soft(());
    let t = Instant::now();
    while alive(pid) && t.elapsed() < Duration::from_secs(3) {
        std::thread::sleep(Duration::from_millis(100));
    }
    if alive(pid) {
        hard(());
    }
}

/// Is this pid still a marimo process (and not a recycled pid belonging to something else)?
fn is_marimo(pid: u32) -> bool {
    #[cfg(unix)]
    let out = Command::new("ps").args(["-o", "command=", "-p", &pid.to_string()]).output();
    #[cfg(windows)]
    let out = platform::quiet(Command::new("powershell").args([
        "-NoProfile", "-Command", &format!("(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"),
    ]))
    .output();
    matches!(&out, Ok(o) if String::from_utf8_lossy(&o.stdout).contains("marimo"))
}

/// At launch: if a previous run crashed and left marimo up, stop it.
pub fn cleanup_stale() {
    let Ok(text) = std::fs::read_to_string(pid_file()) else { return };
    if let Ok(pid) = text.trim().parse::<u32>() {
        if is_marimo(pid) {
            stop_tree(pid);
        }
    }
    let _ = std::fs::remove_file(pid_file());
}

fn log_tail() -> String {
    let s = std::fs::read_to_string(log_file()).unwrap_or_default();
    let lines: Vec<&str> = s.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(6)..].join("\n")
}

// ---------------------------------------------------------------- commands

/// Start (or reuse) the server and return the URL for `file`, a path relative to the notebooks folder.
#[tauri::command]
pub async fn start_marimo(app: AppHandle, file: String) -> Result<String, String> {
    crate::notebooks::seed(&app)?;
    let home = crate::notebooks::home_dir();
    let root = home.canonicalize().map_err(|_| format!("notebooks folder not found at {}", home.display()))?;
    let path = root.join(&file).canonicalize().map_err(|_| format!("notebook not found: {file}"))?;
    if !path.starts_with(&root) || path.extension().map_or(true, |e| e != "py") {
        return Err("invalid notebook path".into());
    }
    let state = app.state::<AppState>();
    let mut guard = state.marimo.lock().unwrap();
    let running = guard.as_mut().map_or(false, |s| matches!(s.child.try_wait(), Ok(None)));
    if !running {
        let py = crate::python_path(&app);
        let port = free_port()?;
        let token = random_token()?;
        std::fs::create_dir_all(app_dir()).map_err(|e| e.to_string())?;
        let log = std::fs::File::create(log_file()).map_err(|e| e.to_string())?;
        let mut cmd = Command::new(&py);
        cmd.args(["-m", "marimo", "edit", "--headless", "--host", "127.0.0.1", "--port", &port.to_string(),
                  "--token-password", &token, "--skip-update-check"])
            .arg(&home)                                       // plain path (no Windows \\?\ prefix)
            .current_dir(&home)
            .stdin(Stdio::null())
            .stdout(log.try_clone().map_err(|e| e.to_string())?)
            .stderr(log);
        isolate(&mut cmd);
        let child = cmd.spawn().map_err(|e| format!("could not start {} -m marimo: {e}. Set the Python path in Settings.", py.display()))?;
        let _ = std::fs::write(pid_file(), child.id().to_string());
        *guard = Some(Server { child, port, token });
        let start = Instant::now();
        loop {
            if TcpStream::connect(("127.0.0.1", port)).is_ok() {
                break;
            }
            let s = guard.as_mut().unwrap();
            if let Ok(Some(status)) = s.child.try_wait() {
                *guard = None;
                let tail = log_tail();
                return Err(if tail.contains("No module named marimo") {
                    format!("marimo isn't installed for {}. Run: {} -m pip install marimo", py.display(), py.display())
                } else {
                    format!("marimo exited early ({status}).\n{tail}")
                });
            }
            if start.elapsed() > Duration::from_secs(40) {
                let pid = s.child.id();
                *guard = None;
                stop_tree(pid);
                return Err("marimo did not start within 40 s".into());
            }
            std::thread::sleep(Duration::from_millis(150));
        }
    }
    let s = guard.as_ref().unwrap();
    let rel = platform::slash(path.strip_prefix(&root).unwrap());
    Ok(format!("http://127.0.0.1:{}/?file={}&access_token={}", s.port, encode(&rel), s.token))
}

#[tauri::command]
pub fn stop_marimo(app: AppHandle) {
    shutdown(&app);
}

/// Stop marimo and all its kernels, and tell the UI so open notebook views reset instead of showing a dead page.
pub fn shutdown(app: &AppHandle) {
    let server = app.state::<AppState>().marimo.lock().unwrap().take();
    if let Some(mut s) = server {
        stop_tree(s.child.id());
        let _ = s.child.wait();
        let _ = std::fs::remove_file(pid_file());
        let _ = app.emit("marimo-stopped", ());
    }
}
