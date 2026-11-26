use std::{
    io,
    path::{Component, Path, PathBuf},
};

use tokio::{fs, io::AsyncWriteExt};
use uuid::Uuid;

const STEM_NAMES: [&str; 4] = ["drums", "bass", "other", "vocals"];

pub fn projects_root(upload_dir: &Path) -> PathBuf {
    upload_dir
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("projects")
}

pub fn stem_storage_key(project_id: Uuid, stem_name: &str) -> Result<String, String> {
    if !STEM_NAMES.contains(&stem_name) {
        return Err(format!("unsupported stem name: {stem_name}"));
    }
    Ok(format!("{project_id}/stems/{stem_name}.wav"))
}

pub fn original_storage_key(project_id: Uuid, _asset_id: Uuid, extension: &str) -> String {
    format!("{project_id}/original.{extension}")
}

pub fn resolve_storage_key(root: &Path, key: &str) -> Result<PathBuf, String> {
    let relative = Path::new(key);
    if relative.is_absolute()
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err("invalid project storage key".into());
    }
    Ok(root.join(relative))
}

/// Copy into a same-directory temporary file, sync it, then publish without
/// replacing an existing project asset. The returned path is safe to expose
/// only after its database reference is committed.
pub async fn atomic_copy_new(source: &Path, destination: &Path) -> Result<(), String> {
    let parent = destination
        .parent()
        .ok_or_else(|| "project asset destination has no parent directory".to_owned())?;
    fs::create_dir_all(parent)
        .await
        .map_err(|error| format!("create project asset directory: {error}"))?;
    let temp = parent.join(format!(".{}.{}.tmp", Uuid::new_v4(), Uuid::new_v4()));
    let copy_result = async {
        let mut input = fs::File::open(source)
            .await
            .map_err(|error| format!("open processing output {}: {error}", source.display()))?;
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .await
            .map_err(|error| format!("create temporary project asset: {error}"))?;
        tokio::io::copy(&mut input, &mut output)
            .await
            .map_err(|error| format!("copy project asset: {error}"))?;
        output
            .flush()
            .await
            .map_err(|error| format!("flush project asset: {error}"))?;
        output
            .sync_all()
            .await
            .map_err(|error| format!("sync project asset: {error}"))?;
        drop(output);
        fs::hard_link(&temp, destination)
            .await
            .map_err(|error| match error.kind() {
                io::ErrorKind::AlreadyExists => format!(
                    "refusing to overwrite existing project asset {}",
                    destination.display()
                ),
                _ => format!("publish project asset atomically: {error}"),
            })?;
        Ok::<(), String>(())
    }
    .await;
    let remove_result = fs::remove_file(&temp).await;
    if let Err(error) = copy_result {
        let _ = remove_result;
        return Err(error);
    }
    remove_result.map_err(|error| format!("remove temporary project asset: {error}"))?;
    Ok(())
}

pub fn validate_stem_wav(
    path: &Path,
    sample_rate: u32,
    channels: u16,
    frame_count: u64,
) -> Result<u64, String> {
    let reader = hound::WavReader::open(path)
        .map_err(|error| format!("open separated stem {}: {error}", path.display()))?;
    let spec = reader.spec();
    let frames = u64::from(reader.duration());
    if spec.sample_rate != sample_rate || spec.channels != channels || frames != frame_count {
        return Err(format!(
            "stem {} metadata mismatch: got {} Hz, {} channels, {frames} frames; expected {sample_rate} Hz, {channels} channels, {frame_count} frames",
            path.display(),
            spec.sample_rate,
            spec.channels
        ));
    }
    std::fs::metadata(path)
        .map(|metadata| metadata.len())
        .map_err(|error| format!("read stem file size: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_keys_are_project_scoped_and_reject_path_traversal() {
        let project = Uuid::new_v4();
        assert_eq!(
            stem_storage_key(project, "drums").unwrap(),
            format!("{project}/stems/drums.wav")
        );
        assert!(stem_storage_key(project, "../drums").is_err());
        assert!(resolve_storage_key(Path::new("/tmp/projects"), "../outside").is_err());
        assert!(resolve_storage_key(Path::new("/tmp/projects"), "/outside").is_err());
    }

    #[tokio::test]
    async fn atomic_persistence_publishes_complete_bytes_and_refuses_replacement() {
        let root = std::env::temp_dir().join(format!("dubpad-storage-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).await.unwrap();
        let source = root.join("staged.wav");
        let destination = root.join("project").join("drums.wav");
        fs::write(&source, b"complete wave bytes").await.unwrap();

        atomic_copy_new(&source, &destination).await.unwrap();
        assert_eq!(
            fs::read(&destination).await.unwrap(),
            b"complete wave bytes"
        );
        fs::write(&source, b"different retry bytes").await.unwrap();
        assert!(atomic_copy_new(&source, &destination).await.is_err());
        assert_eq!(
            fs::read(&destination).await.unwrap(),
            b"complete wave bytes"
        );

        fs::remove_dir_all(root).await.unwrap();
    }

    #[test]
    fn stem_validation_checks_sample_rate_channels_and_frame_count() {
        let root = std::env::temp_dir().join(format!("dubpad-wav-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("stem.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for _ in 0..8 {
            writer.write_sample(0_i16).unwrap();
        }
        writer.finalize().unwrap();

        assert!(validate_stem_wav(&path, 48_000, 2, 4).is_ok());
        assert!(validate_stem_wav(&path, 48_000, 2, 3).is_err());
        assert!(validate_stem_wav(&path, 44_100, 2, 4).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
