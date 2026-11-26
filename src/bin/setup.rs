#[path = "../../installer/mod.rs"]
mod installer;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "dubpad-setup", version, about = "DubPad installation manager")]
struct Cli {
    #[arg(long, global = true, default_value = ".")]
    app_root: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Provision or reuse the Demucs Python runtime.
    Runtime,

    /// Download and verify inference models.
    Models,

    /// Verify runtime, models, and worker startup.
    Verify,

    /// Run the complete DubPad installation.
    Install,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = cli.app_root.canonicalize()?;

    println!("DubPad Setup");
    println!("Application root: {}", root.display());

    match cli.command {
        Commands::Runtime => {
            installer::runtime::setup(&root)?;
        }
        Commands::Models => {
            installer::models::setup(&root).await?;
        }
        Commands::Verify => {
            installer::verify::setup(&root)?;
        }
        Commands::Install => {
            installer::install::setup(&root).await?;
        }
    }

    Ok(())
}
