use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug, Deserialize)]
pub struct RuntimeResult {
    pub python: String,
    pub action: String,
    #[serde(default)]
    pub report: String,
}

fn find_python() -> Result<PathBuf> {
    if let Some(configured) = env::var_os("DEMUCS_PYTHON") {
        let path = PathBuf::from(configured);

        if path.is_file() {
            return Ok(path);
        }

        bail!("DEMUCS_PYTHON does not exist: {}", path.display());
    }

    for name in ["python3", "python"] {
        if let Ok(output) = Command::new(name).arg("--version").output() {
            if output.status.success() {
                return Ok(PathBuf::from(name));
            }
        }
    }

    bail!("Python 3 is required to configure the Demucs runtime.");
}

pub fn setup(root: &Path) -> Result<RuntimeResult> {
    let python = find_python()?;
    let script = root.join("scripts/setup_demucs_runtime.py");

    if !script.is_file() {
        bail!("Runtime installer missing: {}", script.display());
    }

    println!("Bootstrap Python: {}", python.display());
    println!("Running Demucs runtime installer...");

    let output = Command::new(&python)
        .arg(&script)
        .arg("--ensure")
        .arg("--json")
        .arg("--app-root")
        .arg(root)
        .current_dir(root)
        .output()
        .context("Failed to execute Demucs runtime installer")?;

    if !output.status.success() {
        bail!(
            "Demucs runtime setup failed:\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let result: RuntimeResult = serde_json::from_slice(&output.stdout)
        .context("Invalid JSON returned by Demucs runtime installer")?;

    if !Path::new(&result.python).is_file() {
        bail!("Selected Python does not exist: {}", result.python);
    }

    println!("Runtime action: {}", result.action);
    println!("Selected Python: {}", result.python);

    if !result.report.is_empty() {
        println!("{}", result.report);
    }

    Ok(result)
}
