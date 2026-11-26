mod config;
mod error;
mod models;
mod routes;
mod services;
mod state;

use core::fmt;
use std::{net::SocketAddr, sync::Arc};

use axum::{
    Router,
    http::{HeaderValue, Method},
};
use sqlx::postgres::PgPoolOptions;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tower_http::cors::CorsLayer;
use tracing_subscriber::EnvFilter;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::{
    config::Config,
    routes::event_channel,
    services::{job_manager::JobManager, storage},
    state::AppState,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("soniclab_backend=info,tower_http=info")),
        )
        .json()
        .init();

    let config = Arc::new(Config::from_env().map_err(std::io::Error::other)?);
    // RTen reads this once when its shared Rayon pool is first initialized.
    // Bound its CPU workers independently from Tokio's blocking pool.
    unsafe { std::env::set_var("RTEN_NUM_THREADS", config.beat_this_threads.to_string()) };
    tracing::info!(
        threads = config.beat_this_threads,
        "configured Beat This RTen CPU workers"
    );
    storage::prepare_root(&config.upload_dir).await?;
    let pool = PgPoolOptions::new()
        .max_connections(config.database_max_connections)
        .connect(&config.database_url)
        .await?;
    sqlx::migrate!("./migrations").run(&pool).await?;

    let events = event_channel();
    let manager = Arc::new(JobManager::new(
        pool.clone(),
        events.clone(),
        config.job_poll_interval,
        config.clone(),
    ));
    let requeued = manager.recover_interrupted().await?;
    tracing::info!(requeued_jobs = requeued, "recovered interrupted jobs");
    let state = AppState {
        db: pool,
        config: config.clone(),
        jobs: manager.clone(),
        events,
    };
    let origins: Vec<HeaderValue> = config
        .cors_allow_origins
        .iter()
        .filter_map(|origin| origin.parse().ok())
        .collect();
    let cors = CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(tower_http::cors::Any);
    let app: Router = axum::Router::new()
        .nest("/api/v1", routes::router(state, config.max_upload_bytes))
        .merge(SwaggerUi::new("/swagger-ui").url("/api/v1/openapi.json", routes::ApiDoc::openapi()))
        .layer(cors);
    let listener: TcpListener = TcpListener::bind(config.address).await?;
    println!("{}", config.address);

    tracing::info!(address = %config.address, concurrency = config.job_concurrency, "DubPad API listening");
    serve(listener, app, manager).await?;
    Ok(())
}

async fn serve(
    listener: TcpListener,
    app: axum::Router,
    manager: Arc<JobManager>,
) -> Result<(), std::io::Error> {
    let shutdown = CancellationToken::new();
    let worker_shutdown = shutdown.clone();
    let workers = tokio::spawn(async move {
        manager.run(worker_shutdown).await;
    });
    let signal = async move {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutdown signal received");
        shutdown.cancel();
    };
    let result = axum::serve(listener, app)
        .with_graceful_shutdown(signal)
        .await;
    let _ = workers.await;
    result
}

#[allow(dead_code)]
fn _bind_address_type(_: SocketAddr) {}
