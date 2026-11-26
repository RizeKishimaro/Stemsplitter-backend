use anyhow::Result;
use std::path::Path;

pub async fn setup(root: &Path) -> Result<()> {
    println!("\n================================");
    println!("       DubPad Installation");
    println!("================================");

    // 1. Detect platform.
    println!("\n[1/4] Detecting platform...");
    super::platform::print();

    // 2. Prepare runtime once.
    println!("\n[2/4] Preparing Python runtime...");
    let runtime = super::runtime::setup(root)?;
    let python = std::path::Path::new(&runtime.python);

    // 3. Download inference models.
    println!("\n[3/4] Preparing inference models...");
    super::models::setup_with_python(root, python).await?;

    // 4. Verify installation.
    println!("\n[4/4] Verifying installation...");
    super::verify::setup_with_python(root, python)?;

    println!("\n================================");
    println!("  DubPad installation complete!");
    println!("================================");

    Ok(())
}
