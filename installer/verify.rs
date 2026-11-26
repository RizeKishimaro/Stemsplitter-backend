use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

/// Initialize the runtime and verify the installation independently.
pub fn setup(root: &Path) -> Result<()> {
    let runtime = super::runtime::setup(root)?;
    setup_with_python(root, Path::new(&runtime.python))
}

/// Verify the installation using an existing Python interpreter.
pub fn setup_with_python(root: &Path, python: &Path) -> Result<()> {
    println!("\nDubPad installation verification");

    // ---------------------------------------------------------
    // 1. Check required application files.
    // ---------------------------------------------------------

    let worker: PathBuf = root.join("worker/demucs_worker.py");
    let beat_model: PathBuf = root.join("models/beat_this.onnx");
    let mel_model: PathBuf = root.join("models/mel_spectrogram.onnx");

    for (label, path) in [
        ("Demucs Worker", &worker),
        ("Beat This model", &beat_model),
        ("Mel-spectrogram model", &mel_model),
    ] {
        if !path.is_file() {
            bail!("Missing {}: {}", label, path.display());
        }

        let metadata = fs::metadata(path)?;

        if metadata.len() == 0 {
            bail!("{} is empty. Please check: {}", label, path.display());
        }

        println!(
            "[OK] {} ({:.2} MB)",
            label,
            metadata.len() as f64 / 1_048_576.0
        );
    }

    // ---------------------------------------------------------
    // 2. Check the selected Python interpreter.
    // ---------------------------------------------------------

    if !python.is_file() {
        bail!("Demucs Python interpreter not found: {}", python.display());
    }

    println!("[OK] Python interpreter: {}", python.display());

    // ---------------------------------------------------------
    // 3. Start the actual Demucs worker.
    // Run in CPU/offline mode to avoid depending on CUDA
    // or network access during installation verification.
    // ---------------------------------------------------------

    println!("\nStarting Demucs worker smoke test (CPU, offline)...");

    let mut child = Command::new(python)
        .arg(&worker)
        .current_dir(root)
        .env("DEMUCS_DEVICE", "cpu")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to start Demucs worker")?;

    // Send shutdown immediately.
    // The worker loads HTDemucs first, emits its ready event,
    // then processes the queued shutdown command.
    {
        let stdin = child
            .stdin
            .as_mut()
            .context("Could not open worker stdin")?;

        stdin
            .write_all(b"{\"type\":\"shutdown\"}\n")
            .context("Could not send worker shutdown command")?;
    }

    // Wait for the worker to finish loading and shutting down.
    let output = child
        .wait_with_output()
        .context("Failed waiting for Demucs worker")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // ---------------------------------------------------------
    // 4. Check the worker's JSON-lines readiness event.
    // ---------------------------------------------------------

    let ready = stdout.lines().any(|line| {
        serde_json::from_str::<Value>(line)
            .ok()
            .and_then(|message| {
                message
                    .get("type")
                    .and_then(Value::as_str)
                    .map(|kind| kind == "ready")
            })
            .unwrap_or(false)
    });

    if !ready {
        bail!(
            "Demucs worker did not become ready.\n\nstdout:\n{}\n\nstderr:\n{}",
            stdout,
            stderr
        );
    }

    // ---------------------------------------------------------
    // 5. Check the worker's exit status.
    // ---------------------------------------------------------

    if !output.status.success() {
        bail!(
            "Demucs worker exited unsuccessfully: {}\n{}",
            output.status,
            stderr
        );
    }

    println!("[OK] Demucs worker loaded HTDemucs and reported ready");
    println!("[OK] Worker shut down successfully");

    println!("\nAll verification checks passed.");

    Ok(())
}
