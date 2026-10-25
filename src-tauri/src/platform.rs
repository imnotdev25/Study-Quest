//! OS helpers so the rest of the backend stays platform-neutral: macOS, Windows and Linux.
//! Every external program gets its input as separate arguments or environment variables, never pasted into a shell.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The user's home folder: $HOME on macOS/Linux, %USERPROFILE% on Windows.
pub fn home() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
}

pub fn downloads() -> PathBuf {
    home().join("Downloads")
}

/// On Windows, don't flash a console window for background processes. No-op elsewhere.
pub fn quiet(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd
}

/// Open a file, folder or URL with its default app.
pub fn open(target: &str) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    let mut c = Command::new("/usr/bin/open");
    #[cfg(windows)]
    let mut c = Command::new("explorer");
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut c = Command::new("xdg-open");
    c.arg(target).spawn().map(|_| ()).map_err(|e| e.to_string())
}

/// Show a file in Finder / Explorer / the file manager.
pub fn reveal(path: &Path) {
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("/usr/bin/open").arg("-R").arg(path).spawn();
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // explorer needs exactly /select,"path"; the path comes from our own folders (no quotes possible in Windows paths).
        let _ = Command::new("explorer").raw_arg(format!("/select,\"{}\"", path.display())).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = Command::new("xdg-open").arg(if path.is_dir() { path } else { path.parent().unwrap_or(path) }).spawn();
    }
}

/// Text-to-speech for voice nudges: `say` on macOS, System.Speech on Windows, spd-say / espeak on Linux.
pub fn speak(text: &str) {
    let clean: String = text.chars().filter(|c| c.is_alphanumeric() || " .,!?'-:".contains(*c)).take(240).collect();
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("/usr/bin/say").args(["-r", "190", "--", &clean]).spawn();
    }
    #[cfg(windows)]
    {
        let _ = quiet(
            Command::new("powershell")
                .args(["-NoProfile", "-Command",
                       "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($env:SQ_TEXT)"])
                .env("SQ_TEXT", &clean),
        )
        .spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if Command::new("spd-say").arg("--").arg(&clean).spawn().is_err() {
            let _ = Command::new("espeak").arg(&clean).spawn();
        }
    }
}

/// What the OS calls its bin, for messages.
pub const TRASH_NAME: &str = if cfg!(windows) { "Recycle Bin" } else { "Trash" };

/// Move a file to the system Trash / Recycle Bin. Returns false if the OS refused (caller then keeps a copy elsewhere).
pub fn trash(p: &Path) -> bool {
    #[cfg(target_os = "macos")]
    {
        // Finder is the normal way and keeps "Put Back".
        let finder = Command::new("/usr/bin/osascript")
            .args(["-e", "on run argv", "-e", "tell application \"Finder\" to delete (POSIX file (item 1 of argv) as alias)",
                   "-e", "end run"])
            .arg(p)
            .output();
        if matches!(&finder, Ok(o) if o.status.success()) && !p.exists() {
            return true;
        }
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if std::fs::rename(p, home().join(".Trash").join(format!("{stamp}_{name}"))).is_ok() {
            return true;
        }
    }
    #[cfg(windows)]
    {
        let r = quiet(
            Command::new("powershell")
                .args(["-NoProfile", "-Command",
                       "Add-Type -AssemblyName Microsoft.VisualBasic; [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($env:SQ_PATH, 'OnlyErrorDialogs', 'SendToRecycleBin')"])
                .env("SQ_PATH", p),
        )
        .output();
        if matches!(&r, Ok(o) if o.status.success()) && !p.exists() {
            return true;
        }
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let r = Command::new("gio").arg("trash").arg(p).output();
        if matches!(&r, Ok(o) if o.status.success()) && !p.exists() {
            return true;
        }
    }
    false
}

/// Paths shown to the frontend and used in URLs always use forward slashes.
pub fn slash(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}
