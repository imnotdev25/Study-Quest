//! macOS Calendar sync. Events go into a "Study Quest" calendar through JavaScript for Automation
//! (osascript). Arguments are passed as argv, never interpolated into the script, so titles can't inject code.
//! The first call triggers macOS's "allow Study Quest to control Calendar" prompt.

use std::process::Command;

const FIND_CAL: &str = r#"
function findCal(Cal, calName, create) {
  let cal = Cal.calendars.whose({ name: calName })[0];
  try { cal.name(); return cal; } catch (e) {}
  if (!create) return null;
  try {
    Cal.calendars.push(Cal.Calendar({ name: calName }));
    cal = Cal.calendars.whose({ name: calName })[0];
    cal.name();
    return cal;
  } catch (e2) {
    return Cal.calendars.whose({ writable: true })[0];
  }
}
"#;

const ADD_EVENT_JXA: &str = r#"
function run(argv) {
  const [title, notes, startMs, minutes, calName, alarm] = argv;
  const Cal = Application("Calendar");
  const cal = findCal(Cal, calName, true);
  const start = new Date(Number(startMs));
  const end = new Date(start.getTime() + Number(minutes) * 60000);
  const ev = Cal.Event({ summary: title, startDate: start, endDate: end, description: notes });
  cal.events.push(ev);
  if (alarm === "1") {
    try { ev.displayAlarms.push(Cal.DisplayAlarm({ triggerInterval: -10 })); } catch (e) {}
  }
  let uid = "";
  try { uid = ev.uid(); } catch (e) {}
  return JSON.stringify({ calendar: cal.name(), uid: uid });
}
"#;

// Deletes the event with this uid, plus every event with the same title starting within a minute of startMs
// (that second rule also catches events booked before uids were stored, and duplicates).
const DELETE_EVENT_JXA: &str = r#"
function run(argv) {
  const [uid, title, startMs, calName] = argv;
  const Cal = Application("Calendar");
  const cal = findCal(Cal, calName, false);
  if (!cal) return "0";
  const t0 = Number(startMs);
  let n = 0;
  if (uid) {
    try { const byId = cal.events.whose({ uid: uid }); for (let i = byId.length - 1; i >= 0; i--) { byId[i].delete(); n++; } } catch (e) {}
  }
  const same = cal.events.whose({ summary: title });
  for (let i = same.length - 1; i >= 0; i--) {
    try {
      if (Math.abs(same[i].startDate().getTime() - t0) < 60000) { same[i].delete(); n++; }
    } catch (e) {}
  }
  return String(n);
}
"#;

fn safe(s: &str, n: usize) -> String {
    s.chars().filter(|c| !c.is_control() || *c == '\n').take(n).collect()
}

/// Every event Study Quest writes is titled "🦦 <title>".
fn event_title(title: &str) -> String {
    format!("🦦 {}", safe(title, 120))
}

fn osascript(script: &str, args: &[String]) -> Result<String, String> {
    if !cfg!(target_os = "macos") {
        return Err("Calendar sync is macOS-only. Use Export .ics to add events to your calendar app.".into());
    }
    let full = format!("{FIND_CAL}\n{script}");
    let out = Command::new("/usr/bin/osascript")
        .args(["-l", "JavaScript", "-e", &full])
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
    if out.status.success() {
        return Ok(String::from_utf8_lossy(&out.stdout).trim().to_string());
    }
    let err = String::from_utf8_lossy(&out.stderr).to_string();
    if err.contains("-1743") || err.contains("Not authorized") {
        Err("macOS blocked Calendar access. Allow it in System Settings → Privacy & Security → Automation.".into())
    } else {
        Err(err.trim().to_string())
    }
}

/// Returns {"calendar": name, "uid": event uid ("" if Calendar didn't report one)}.
#[tauri::command]
pub async fn add_calendar_event(
    title: String,
    notes: String,
    start_ms: f64,
    minutes: u32,
    calendar: String,
    alarm: bool,
) -> Result<serde_json::Value, String> {
    let out = osascript(ADD_EVENT_JXA, &[
        event_title(&title),
        safe(&notes, 2000),
        format!("{}", start_ms as i64),
        minutes.clamp(5, 600).to_string(),
        safe(&calendar, 60),
        (if alarm { "1" } else { "0" }).to_string(),
    ])?;
    serde_json::from_str(&out).map_err(|_| format!("unexpected reply from Calendar: {out}"))
}

/// Delete a planned block from Calendar. Returns how many events were removed (0 if it was already gone).
#[tauri::command]
pub async fn delete_calendar_event(uid: String, title: String, start_ms: f64, calendar: String) -> Result<u32, String> {
    let out = osascript(DELETE_EVENT_JXA, &[
        safe(&uid, 200),
        event_title(&title),
        format!("{}", start_ms as i64),
        safe(&calendar, 60),
    ])?;
    out.parse::<u32>().map_err(|_| format!("unexpected reply from Calendar: {out}"))
}

/// Write an .ics to ~/Downloads and open it with the default calendar app (all platforms).
#[tauri::command]
pub fn save_ics(contents: String, name: String) -> Result<String, String> {
    let clean: String = name.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').take(60).collect();
    let dir = crate::platform::downloads();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{}.ics", if clean.is_empty() { "study-quest" } else { &clean }));
    std::fs::write(&path, contents).map_err(|e| e.to_string())?;
    crate::platform::open(&path.to_string_lossy())?;   // opens in Calendar / Outlook / GNOME Calendar
    Ok(path.to_string_lossy().to_string())
}
