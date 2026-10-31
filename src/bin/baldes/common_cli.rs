//! Shared solution-file and BKS helpers.
use std::path::Path;

pub(super) fn write_checkpoint(path: &Path, text: &str) -> Result<(), String> {
    let filename = path
        .file_name()
        .ok_or("solution output needs a filename")?
        .to_string_lossy();
    let temporary = path.with_file_name(format!(".{filename}.{}.tmp", std::process::id()));
    std::fs::write(&temporary, text).map_err(|e| format!("write checkpoint: {e}"))?;
    std::fs::rename(&temporary, path).map_err(|e| format!("rename checkpoint: {e}"))
}

pub(super) fn download_bks(name: &str) -> Result<String, String> {
    // Bundle the dependency-free downloader so installed binaries do not depend
    // on a source checkout. Python performs HTTPS certificate verification.
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let directory = std::env::temp_dir().join(format!("baldes-bks-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&directory).map_err(|e| format!("create BKS temporary directory: {e}"))?;
    let path = directory.join("seed.sol");
    let result = (|| {
        let output = std::process::Command::new("python3")
            .arg("-c")
            .arg(include_str!("../../../scripts/setup/get_bks.py"))
            .arg(name)
            .arg("-o")
            .arg(&path)
            .output()
            .map_err(|e| format!("--get-bks requires Python 3: {e}"))?;
        if !output.status.success() {
            return Err(format!(
                "BKS download failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        eprint!(
            "{}",
            String::from_utf8_lossy(&output.stdout)
                .replace(&path.to_string_lossy().to_string(), "temporary seed")
        );
        std::fs::read_to_string(&path).map_err(|e| e.to_string())
    })();
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&directory);
    result
}
