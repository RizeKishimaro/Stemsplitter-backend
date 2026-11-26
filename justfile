
set dotenv-load := true

backend := "."

# Frontend repository (override with FRONTEND_DIR if needed)
frontend := env_var_or_default("FRONTEND_DIR", "../soniclab-web")

default:
    @just --list

# Rust checks
check-backend:
    cargo fmt --check
    cargo check
    cargo test

# Frontend checks
check-frontend:
    cd {{frontend}} && pnpm lint
    cd {{frontend}} && pnpm build

# Complete local verification
check: check-backend check-frontend

# Run backend
dev-backend:
    cargo run

# Run frontend
dev-frontend:
    cd {{frontend}} && pnpm dev

# Run frontend end-to-end tests
test-e2e:
    cd {{frontend}} && pnpm test:e2e

# Explicitly build both projects
build:
    cargo build --release
    cd {{frontend}} && pnpm build

