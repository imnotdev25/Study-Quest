//! The app's own notebook folder: ~/.study-quest/notebooks
//!   examples copied from the bundle on start (never overwriting your edits), plus mine/ for notebooks you create.
//! Each notebook's first line may hold metadata: `# sq-meta: {"title": ..., "group": ..., "topic": ...}`.

use std::path::{Path, PathBuf};
use tauri::AppHandle;

use crate::platform;

pub fn home_dir() -> PathBuf {
    platform::home().join(".study-quest").join("notebooks")
}

fn copy_tree(src: &Path, dst: &Path, overwrite: bool) -> std::io::Result<usize> {
    std::fs::create_dir_all(dst)?;
    let mut n = 0;
    for e in std::fs::read_dir(src)? {
        let p = e?.path();
        let target = dst.join(p.file_name().unwrap());
        if p.is_dir() {
            n += copy_tree(&p, &target, overwrite)?;
        } else if p.extension().map_or(false, |x| x == "py") && (overwrite || !target.exists()) {
            std::fs::copy(&p, &target)?;
            n += 1;
        }
    }
    Ok(n)
}

/// Copy any bundled example that isn't in the home folder yet. Called at startup.
/// An example you haven't edited is refreshed when the bundled one gains its `# sq-meta:` line.
pub fn seed(app: &AppHandle) -> Result<usize, String> {
    let dst = home_dir();
    std::fs::create_dir_all(dst.join("mine")).map_err(|e| e.to_string())?;
    let src = crate::resource(app, "../notebooks", "notebooks");
    if !src.exists() {
        return Ok(0);
    }
    let mut n = copy_tree(&src, &dst, false).map_err(|e| e.to_string())?;
    for sub in ["interactive", "code"] {
        let Ok(entries) = std::fs::read_dir(src.join(sub)) else { continue };
        for e in entries.flatten() {
            let (from, to) = (e.path(), dst.join(sub).join(e.file_name()));
            let (Ok(new), Ok(old)) = (std::fs::read_to_string(&from), std::fs::read_to_string(&to)) else { continue };
            // Old copy = new file minus its first line: unedited, just missing metadata.
            if new.starts_with("# sq-meta: ") && !old.starts_with("# sq-meta: ") && new.splitn(2, '\n').nth(1) == Some(old.as_str()) {
                let _ = std::fs::write(&to, new);
                n += 1;
            }
        }
    }
    Ok(n)
}

/// Group for notebooks without a metadata line (e.g. copies seeded by an older build). Mirrors build_notebooks.py.
fn guess_group(stem: &str, rel: &str) -> &'static str {
    if rel.starts_with("mine/") {
        return "My notebooks";
    }
    match stem {
        "bandit_casino" | "ppo_clip" | "gridworld_values" | "grpo_advantages" | "kl_estimators" | "rl_bridge" => "Reinforcement Learning",
        "softmax_temperature" | "attention_heatmap" => "Transformers & LLMs",
        "kv_cache_calculator" | "speculative_decoding" | "inf_memory" | "serving_sim" | "paged_kv" | "quant" | "attn_kernels"
        | "spec_decode" | "distributed" | "prod" | "specialized" | "mini_engine" => "Inference",
        "gradient_descent" | "low_rank_svd" | "math_kit" | "intuitions" => "Math & Optimisation",
        "bayes_coin" | "central_limit" => "Probability & Statistics",
        "pass_at_k" => "Agents & Evals",
        "reward_hacking" | "calibration_abstain" | "truth_probe" | "judge_bias" | "prompt_injection_lab" | "weak_to_strong" => "AI Safety",
        "verifier_lab" | "sandbox_harness" | "rl_data_filter" => "RL Environments",
        s if s.starts_with("mf_") => "Math & Optimisation",
        s if s.starts_with("pr_") || s.starts_with("jt_") || s.starts_with("st_") => "Probability & Statistics",
        s if s.starts_with("ag_") => "Agents & Evals",
        _ => "Transformers & LLMs",
    }
}

fn meta(path: &Path, rel: &str) -> serde_json::Value {
    let first = std::fs::read_to_string(path).ok().and_then(|s| s.lines().next().map(str::to_string)).unwrap_or_default();
    let mut m: serde_json::Value = first
        .strip_prefix("# sq-meta: ")
        .and_then(|j| serde_json::from_str(j).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    let stem = path.file_stem().unwrap().to_string_lossy().to_string();
    let mine = rel.starts_with("mine/");
    let obj = m.as_object_mut().unwrap();
    obj.entry("title").or_insert(stem.replace('_', " ").into());
    obj.entry("emoji").or_insert(if mine { "✏️" } else { "📓" }.into());
    obj.entry("group").or_insert(guess_group(&stem, rel).into());
    obj.entry("kind").or_insert(if mine { "mine" } else if rel.starts_with("interactive/") { "interactive" } else { "code" }.into());
    obj.entry("desc").or_insert("".into());
    obj.entry("topic").or_insert("".into());
    obj.insert("file".into(), rel.into());
    obj.insert("mine".into(), mine.into());
    m
}

#[tauri::command]
pub fn list_notebooks(app: AppHandle) -> Result<Vec<serde_json::Value>, String> {
    seed(&app)?;
    let root = home_dir();
    let mut out = vec![];
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).map_err(|e| e.to_string())?.flatten() {
            let p = e.path();
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            if name.starts_with('.') || name == "__pycache__" || name == "__marimo__" {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().map_or(false, |x| x == "py") {
                let rel = platform::slash(p.strip_prefix(&root).unwrap());
                out.push(meta(&p, &rel));
            }
        }
    }
    Ok(out)
}

const TEMPLATE_HEAD: &str = "import marimo\n\n__generated_with = \"0.25.1\"\napp = marimo.App(width=\"medium\")\n\n\n@app.cell\ndef _():\n    import marimo as mo\n    import numpy as np\n    import matplotlib.pyplot as plt\n    return mo, np, plt\n\n\n";
const TEMPLATE_TAIL: &str = "if __name__ == \"__main__\":\n    app.run()\n";

fn template(kind: &str, title: &str, topic: &str) -> String {
    let intro = format!(
        "@app.cell\ndef _(mo):\n    mo.md(\"\"\"\n    # {title}\n\n    {}Markdown cells for notes, Python cells for code. Change anything and dependent cells re-run.\n    \"\"\")\n    return\n\n\n",
        if topic.is_empty() { String::new() } else { format!("Notes and experiments for **{topic}**. ") }
    );
    let body = match kind {
        "slider" => "@app.cell\ndef _(mo):\n    k = mo.ui.slider(1, 10, value=3, label=\"k\")\n    k\n    return (k,)\n\n\n@app.cell\ndef _(k, np, plt):\n    _x = np.linspace(0, 2 * np.pi, 200)\n    _fig, _ax = plt.subplots(figsize=(6, 3))\n    _ax.plot(_x, np.sin(k.value * _x))\n    _ax.set_title(f\"sin({k.value} x)\")\n    _fig\n    return\n\n\n",
        "torch" => "@app.cell\ndef _():\n    import torch\n    x = torch.randn(4, 8)\n    w = torch.randn(8, 3, requires_grad=True)\n    loss = (x @ w).softmax(-1)[:, 0].mean()\n    loss.backward()\n    w.grad.shape, float(loss)\n    return\n\n\n",
        _ => "@app.cell\ndef _(np):\n    x = np.arange(5)\n    x ** 2\n    return\n\n\n",
    };
    format!("{TEMPLATE_HEAD}{intro}{body}{TEMPLATE_TAIL}")
}

#[tauri::command]
pub fn create_notebook(title: String, group: String, topic: String, template_kind: String) -> Result<String, String> {
    let clean = |s: &str, n: usize| -> String {
        s.chars().filter(|c| c.is_alphanumeric() || " -_:,.()&?!+'".contains(*c)).take(n).collect::<String>().trim().to_string()
    };
    let title = { let t = clean(&title, 80); if t.is_empty() { "Untitled notebook".to_string() } else { t } };
    let group = { let g = clean(&group, 40); if g.is_empty() { "My notebooks".to_string() } else { g } };
    let topic = clean(&topic, 80);
    let slug: String = title.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    let slug = slug.trim_matches('_').chars().take(40).collect::<String>();
    let dir = home_dir().join("mine");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut name = format!("{}.py", if slug.is_empty() { "notebook" } else { &slug });
    let mut i = 2;
    while dir.join(&name).exists() {
        name = format!("{slug}_{i}.py");
        i += 1;
    }
    let meta = serde_json::json!({"title": title, "emoji": "✏️", "group": group, "kind": "mine", "desc": "", "topic": topic});
    std::fs::write(dir.join(&name), format!("# sq-meta: {meta}\n{}", template(&template_kind, &title, &topic)))
        .map_err(|e| e.to_string())?;
    Ok(format!("mine/{name}"))
}

/// Move one of your own notebooks to the Trash / Recycle Bin (recoverable). Examples can't be deleted.
/// Returns where it went: the Trash, or the app's own trash folder if the OS refused.
#[tauri::command]
pub async fn delete_notebook(file: String) -> Result<String, String> {
    let root = home_dir().canonicalize().map_err(|e| e.to_string())?;
    let p = root.join(&file).canonicalize().map_err(|_| format!("notebook not found: {file}"))?;
    if !p.starts_with(root.join("mine")) || p.extension().map_or(true, |x| x != "py") {
        return Err("only your own notebooks (in mine/) can be deleted".into());
    }
    if platform::trash(&p) {
        return Ok(platform::TRASH_NAME.into());
    }
    // Last resort: the app's own trash folder, so the notebook is never lost.
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let name = p.file_name().unwrap().to_string_lossy().to_string();
    let own = home_dir().parent().unwrap().join("trash");
    std::fs::create_dir_all(&own).map_err(|e| e.to_string())?;
    std::fs::rename(&p, own.join(format!("{stamp}_{name}"))).map_err(|e| format!("couldn't move it: {e}"))?;
    Ok(own.to_string_lossy().to_string())
}

#[tauri::command]
pub fn restore_examples(app: AppHandle) -> Result<usize, String> {
    let src = crate::resource(&app, "../notebooks", "notebooks");
    copy_tree(&src, &home_dir(), true).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn reveal_notebooks() -> Result<(), String> {
    let d = home_dir();
    std::fs::create_dir_all(&d).map_err(|e| e.to_string())?;
    platform::open(&d.to_string_lossy())
}
