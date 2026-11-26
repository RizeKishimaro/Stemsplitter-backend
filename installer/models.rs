use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, process::Command};
use tokio::{fs::File, io::AsyncWriteExt};

const BEAT_THIS_RELEASE: &str =
    "https://github.com/danigb/beat-this-rs/releases/download/model-large";

const MEL_MODEL_URL: &str =
    "https://raw.githubusercontent.com/danigb/beat-this-rs/main/models/mel_spectrogram.onnx";

async fn download_file(url: &str, destination: &Path) -> Result<()> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }

    let temporary = destination.with_extension(format!(
        "{}.part",
        destination
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("download")
    ));

    let client = reqwest::Client::builder()
        .user_agent("DubPad-Setup/1.0")
        .build()?;

    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("Request failed: {url}"))?
        .error_for_status()
        .with_context(|| format!("Download returned an HTTP error: {url}"))?;

    let total = response.content_length().unwrap_or(0);

    let progress = if total > 0 {
        ProgressBar::new(total)
    } else {
        ProgressBar::new_spinner()
    };

    progress.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} {msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes}",
        )?
        .progress_chars("##-"),
    );

    progress.set_message(
        destination
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
    );

    let mut file = File::create(&temporary).await?;
    let mut stream = response.bytes_stream();

    let result: Result<()> = async {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Download stream failed")?;
            file.write_all(&chunk).await?;
            progress.inc(chunk.len() as u64);
        }

        file.flush().await?;
        file.sync_all().await?;
        Ok(())
    }
    .await;

    if let Err(error) = result {
        drop(file);
        let _ = tokio::fs::remove_file(&temporary).await;
        progress.abandon_with_message("Download failed");
        return Err(error);
    }

    drop(file);

    let metadata = tokio::fs::metadata(&temporary).await?;

    if metadata.len() == 0 {
        tokio::fs::remove_file(&temporary).await?;
        bail!("Downloaded file is empty: {url}");
    }

    tokio::fs::rename(&temporary, destination).await?;

    progress.finish_with_message("Downloaded");

    Ok(())
}

async fn download_text(url: &str) -> Result<String> {
    let response = reqwest::get(url).await?.error_for_status()?;

    Ok(response.text().await?)
}

fn parse_checksum(text: &str) -> Result<String> {
    let checksum = text
        .split_whitespace()
        .next()
        .context("Checksum file is empty")?;

    if checksum.len() != 64 || !checksum.chars().all(|c| c.is_ascii_hexdigit()) {
        bail!("Invalid SHA-256 checksum format");
    }

    Ok(checksum.to_ascii_lowercase())
}

async fn sha256(path: &Path) -> Result<String> {
    let contents = tokio::fs::read(path).await?;
    let digest = Sha256::digest(&contents);

    Ok(format!("{digest:x}"))
}

async fn download_beat_model(models_dir: &Path) -> Result<()> {
    let destination = models_dir.join("beat_this.onnx");
    let checksum_url = format!("{BEAT_THIS_RELEASE}/beat_this.onnx.sha256");

    println!("\nBeat This model");

    let checksum_text = download_text(&checksum_url).await?;
    let expected = parse_checksum(&checksum_text)?;

    if destination.is_file() {
        let actual = sha256(&destination).await?;

        if actual == expected {
            println!("Existing Beat This model is valid. Reusing it.");
            return Ok(());
        }

        println!("Existing Beat This model failed checksum. Replacing it.");
    }

    download_file(&format!("{BEAT_THIS_RELEASE}/beat_this.onnx"), &destination).await?;

    let actual = sha256(&destination).await?;

    if actual != expected {
        tokio::fs::remove_file(&destination).await?;
        bail!("Beat This model SHA-256 verification failed");
    }

    println!("Beat This model checksum verified.");

    Ok(())
}

async fn download_mel_model(models_dir: &Path) -> Result<()> {
    let destination = models_dir.join("mel_spectrogram.onnx");

    println!("\nBeat This mel-spectrogram model");

    if destination.is_file() && tokio::fs::metadata(&destination).await?.len() > 0 {
        println!("Existing mel model found. Reusing it.");
        return Ok(());
    }

    download_file(MEL_MODEL_URL, &destination).await?;

    Ok(())
}

fn download_htdemucs(python: &Path) -> Result<()> {
    println!("\nHTDemucs model");

    let code = r#"
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download, try_to_load_from_cache

repo = "adefossez/HTDemucs"

snapshot = snapshot_download(repo_id=repo)

config = try_to_load_from_cache(repo, "htdemucs.yaml")

if not isinstance(config, str) or not Path(config).is_file():
    raise RuntimeError("HTDemucs config missing after download")

data = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
signatures = data.get("models", [])

if not signatures:
    raise RuntimeError("No model signatures in HTDemucs config")

for signature in signatures:
    checkpoint = try_to_load_from_cache(repo, f"{signature}.safetensors")

    if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
        raise RuntimeError(f"HTDemucs checkpoint missing: {signature}")

print(f"HTDemucs snapshot: {snapshot}")
print(f"Verified checkpoints: {len(signatures)}")
"#;

    let output = Command::new(python)
        .arg("-c")
        .arg(code)
        .output()
        .context("Could not launch Hugging Face model downloader")?;

    if !output.status.success() {
        bail!(
            "HTDemucs download failed:\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    println!("{}", String::from_utf8_lossy(&output.stdout));

    Ok(())
}

pub async fn setup(root: &Path) -> Result<()> {
    let runtime = super::runtime::setup(root)?;
    setup_with_python(root, Path::new(&runtime.python)).await
}

pub async fn setup_with_python(root: &Path, python: &Path) -> Result<()> {
    if !python.is_file() {
        bail!("Demucs Python interpreter not found: {}", python.display());
    }

    let models_dir = root.join("models");
    fs::create_dir_all(&models_dir)?;

    download_htdemucs(python)?;

    download_mel_model(&models_dir).await?;
    download_beat_model(&models_dir).await?;

    println!("\nAll inference models are ready.");

    Ok(())
}
