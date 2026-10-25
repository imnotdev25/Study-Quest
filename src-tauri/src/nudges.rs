//! Nudge scheduler. Runs in Rust so reminders fire even when the window is closed to the menu bar.
//! The frontend pushes a small snapshot of progress and settings; messages come from src/nudges.json.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_notification::NotificationExt;

const MESSAGES: &str = include_str!("../../src/nudges.json");

#[derive(Clone, serde::Deserialize, serde::Serialize)]
struct Msg {
    t: String,
    b: String,
}

#[derive(Clone, Default, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    enabled: bool,
    last_study_ms: f64,
    streak: u32,
    today_minutes: u32,
    goal_minutes: u32,
    interval_min: u32,
    quiet_start: u32,
    quiet_end: u32,
    troll_level: u32,
    voice: bool,
    name: String,
    tz_offset_min: i32,
    first_run: bool,
    tray_title: String,
}

#[derive(Default)]
pub struct NudgeState {
    snap: Option<Snapshot>,
    last_nudge_ms: f64,
    seed: u64,
}

fn now_ms() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as f64
}

fn rand(st: &mut NudgeState) -> u64 {
    if st.seed == 0 {
        st.seed = now_ms() as u64 | 1;
    }
    st.seed ^= st.seed << 13;
    st.seed ^= st.seed >> 7;
    st.seed ^= st.seed << 17;
    st.seed
}

fn local_hour(s: &Snapshot) -> u32 {
    let local_min = (now_ms() / 60000.0) as i64 - s.tz_offset_min as i64;
    (local_min.rem_euclid(1440) / 60) as u32
}

fn in_quiet(s: &Snapshot) -> bool {
    let h = local_hour(s);
    if s.quiet_start == s.quiet_end {
        false
    } else if s.quiet_start < s.quiet_end {
        h >= s.quiet_start && h < s.quiet_end
    } else {
        h >= s.quiet_start || h < s.quiet_end
    }
}

/// Pick a category and a message, fill in the placeholders.
fn compose(st: &mut NudgeState) -> Option<Msg> {
    let s = st.snap.clone()?;
    let all: HashMap<String, Vec<Msg>> = serde_json::from_str(MESSAGES).ok()?;
    let idle_h = if s.last_study_ms > 0.0 { ((now_ms() - s.last_study_ms) / 3.6e6) as u32 } else { 0 };
    let cat = if s.first_run {
        "welcome"
    } else if s.streak > 0 && s.today_minutes == 0 && local_hour(&s) >= 19 {
        "streakDanger"
    } else if idle_h >= 48 {
        "comeback"
    } else if (rand(st) % 100) < (s.troll_level.min(3) * 30) as u64 {
        // Savage (2) and Unhinged (3) escalate to personal roasts: friends, crush, mom, screen time…
        let roast_pct = match s.troll_level { 2 => 45, 3 => 75, _ => 10 };
        if (rand(st) % 100) < roast_pct { "roast" } else { "troll" }
    } else {
        "gentle"
    };
    let list = all.get(cat)?;
    let m = list[(rand(st) % list.len() as u64) as usize].clone();
    let left = s.goal_minutes.saturating_sub(s.today_minutes).max(5);
    let fill = |x: &str| {
        x.replace("{streak}", &s.streak.to_string())
            .replace("{hours}", &idle_h.max(1).to_string())
            .replace("{left}", &left.to_string())
            .replace("{goal}", &s.goal_minutes.to_string())
            .replace("{name}", if s.name.is_empty() { "friend" } else { &s.name })
    };
    Some(Msg { t: fill(&m.t), b: fill(&m.b) })
}

/// Send one nudge now: native notification, in-app toast, optional voice.
pub fn fire(app: &AppHandle, forced: bool) {
    let state = app.state::<crate::AppState>().nudge.clone();
    let (msg, voice) = {
        let mut st = state.lock().unwrap();
        if st.snap.is_none() {
            st.snap = Some(Snapshot { enabled: true, goal_minutes: 20, troll_level: 1, ..Default::default() });
        }
        let Some(m) = compose(&mut st) else { return };
        st.last_nudge_ms = now_ms();
        let voice = st.snap.as_ref().map_or(false, |s| s.voice);
        (m, voice)
    };
    let _ = app.notification().builder().title(&msg.t).body(&msg.b).show();
    let _ = app.emit("nudge", &msg);
    let _ = forced;
    if voice {
        crate::speak(&format!("{}. {}", msg.t, msg.b));
    }
}

pub fn spawn_loop(app: AppHandle, state: Arc<Mutex<NudgeState>>) {
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(60));
        let due = {
            let st = state.lock().unwrap();
            match &st.snap {
                Some(s) if s.enabled && !in_quiet(s) => {
                    let goal_done = s.today_minutes >= s.goal_minutes && s.goal_minutes > 0;
                    let gap = now_ms() - st.last_nudge_ms.max(s.last_study_ms);
                    !goal_done && gap > s.interval_min.max(15) as f64 * 60000.0
                }
                _ => false,
            }
        };
        if due {
            fire(&app, false);
        }
    });
}

#[tauri::command]
pub fn sync_state(app: AppHandle, snapshot: Snapshot) {
    if let Some(tray) = app.tray_by_id("main") {
        let _ = tray.set_title(Some(snapshot.tray_title.chars().take(24).collect::<String>()));
    }
    let state = app.state::<crate::AppState>().nudge.clone();
    let mut st = state.lock().unwrap();
    if st.last_nudge_ms == 0.0 {
        st.last_nudge_ms = now_ms();                      // don't nudge the moment the app opens
    }
    st.snap = Some(snapshot);
}

#[tauri::command]
pub fn nudge_now(app: AppHandle) {
    fire(&app, true);
}
