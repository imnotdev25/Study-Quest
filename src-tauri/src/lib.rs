mod calendar;
mod marimo;
mod notebooks;
mod nudges;
pub mod platform;
mod runner;

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Manager, RunEvent, WindowEvent};

/// Whether a tray / menu-bar icon exists. Some Linux desktops have no tray: then closing the window quits the app.
static TRAY_OK: AtomicBool = AtomicBool::new(false);

#[derive(Default, Clone, serde::Deserialize)]
pub struct Config {
    pub python: Option<String>,
}

pub struct AppState {
    pub config: Mutex<Config>,
    pub marimo: Mutex<Option<marimo::Server>>,
    pub nudge: Arc<Mutex<nudges::NudgeState>>,
}

/// Python used for the playground and marimo: settings override, then the repo's venv, then the system Python.
pub fn python_path(app: &AppHandle) -> PathBuf {
    if let Some(p) = app.state::<AppState>().config.lock().unwrap().python.clone() {
        if !p.trim().is_empty() {
            return PathBuf::from(p.trim());
        }
    }
    let venv = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.venv-post");
    let dev = if cfg!(windows) { venv.join("Scripts").join("python.exe") } else { venv.join("bin").join("python") };
    if dev.exists() {
        return dev;
    }
    if cfg!(windows) {
        return PathBuf::from("python");
    }
    for p in ["/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"] {
        if PathBuf::from(p).exists() {
            return PathBuf::from(p);
        }
    }
    PathBuf::from("python3")
}

/// A bundled file: the source tree in development, the app's resources folder when packaged.
pub fn resource(app: &AppHandle, dev_rel: &str, bundled: &str) -> PathBuf {
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(dev_rel);
    if dev.exists() {
        return dev;
    }
    app.path().resource_dir().map(|d| d.join(bundled)).unwrap_or(dev)
}

fn runs(py: &PathBuf, args: &[&str]) -> bool {
    platform::quiet(Command::new(py).args(args)).output().map_or(false, |o| o.status.success())
}

#[tauri::command]
fn set_config(state: tauri::State<AppState>, config: Config) {
    *state.config.lock().unwrap() = config;
}

#[tauri::command]
async fn app_info(app: AppHandle) -> serde_json::Value {
    let py = python_path(&app);
    serde_json::json!({
        "python": py.to_string_lossy(),
        "pythonFound": runs(&py, &["--version"]),
        "marimoFound": runs(&py, &["-c", "import marimo"]),
        "notebooks": notebooks::home_dir().to_string_lossy(),
        "os": std::env::consts::OS,
    })
}

#[tauri::command]
fn set_tray(app: AppHandle, title: String) {
    if let Some(tray) = app.tray_by_id("main") {
        let _ = tray.set_title(Some(title.chars().take(24).collect::<String>()));
    }
}

/// Open a local marimo URL in the default browser. Only loopback URLs are allowed.
#[tauri::command]
fn open_local(url: String) -> Result<(), String> {
    if !url.starts_with("http://127.0.0.1:") {
        return Err("only local URLs can be opened".into());
    }
    platform::open(&url)
}

/// Text-to-speech for voice nudges (used by nudges.rs too).
pub fn speak(text: &str) {
    platform::speak(text);
}

/// Write a text file to ~/Downloads and show it. The web view can't do browser-style downloads.
#[tauri::command]
fn save_download(name: String, contents: String) -> Result<String, String> {
    let clean: String = name.chars().filter(|c| c.is_ascii_alphanumeric() || "-_.".contains(*c)).take(80).collect();
    if clean.is_empty() || clean.starts_with('.') {
        return Err("invalid file name".into());
    }
    let dir = platform::downloads();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(clean);
    std::fs::write(&path, contents).map_err(|e| e.to_string())?;
    platform::reveal(&path);
    Ok(path.to_string_lossy().to_string())
}

#[tauri::command]
fn say(text: String) {
    speak(&text);
}

fn show_main(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}

pub fn run() {
    let nudge = Arc::new(Mutex::new(nudges::NudgeState::default()));
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
        .manage(AppState { config: Mutex::new(Config::default()), marimo: Mutex::new(None), nudge: nudge.clone() })
        .invoke_handler(tauri::generate_handler![
            set_config,
            app_info,
            set_tray,
            open_local,
            save_download,
            say,
            runner::run_tests,
            runner::run_scratch,
            marimo::start_marimo,
            marimo::stop_marimo,
            calendar::add_calendar_event,
            calendar::delete_calendar_event,
            calendar::save_ics,
            nudges::sync_state,
            nudges::nudge_now,
            notebooks::list_notebooks,
            notebooks::create_notebook,
            notebooks::delete_notebook,
            notebooks::restore_examples,
            notebooks::reveal_notebooks,
        ])
        .setup(move |app| {
            let open = MenuItem::with_id(app, "open", "Open Study Quest", true, None::<&str>)?;
            let poke = MenuItem::with_id(app, "poke", "Poke me (send a nudge)", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open, &poke, &quit])?;
            // The tray is optional: on a Linux desktop without one, the app still runs (window close then quits).
            let tray = TrayIconBuilder::with_id("main")
                .icon(app.default_window_icon().unwrap().clone())
                .icon_as_template(false)
                .title("🦦")
                .tooltip("Study Quest")
                .menu(&menu)
                .show_menu_on_left_click(true)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "open" => show_main(app),
                    "poke" => nudges::fire(app, true),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .build(app);
            match tray {
                Ok(_) => TRAY_OK.store(true, Ordering::Relaxed),
                Err(e) => eprintln!("tray unavailable, closing the window will quit: {e}"),
            }
            marimo::cleanup_stale();                         // a crash may have left marimo running
            let _ = notebooks::seed(app.handle());           // ~/.study-quest/notebooks
            nudges::spawn_loop(app.handle().clone(), nudge.clone());
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                let app = window.app_handle().clone();
                if TRAY_OK.load(Ordering::Relaxed) {
                    // Keep running in the menu bar / system tray so nudges still fire.
                    api.prevent_close();
                    let _ = window.hide();
                }
                // Notebooks aren't visible any more: stop marimo and its kernels (off the main thread; it may wait).
                std::thread::spawn(move || marimo::shutdown(&app));
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building Study Quest");

    app.run(|app, event| match event {
        RunEvent::Exit => marimo::shutdown(app),
        #[cfg(target_os = "macos")]
        RunEvent::Reopen { .. } => show_main(app),
        _ => {}
    });
}
