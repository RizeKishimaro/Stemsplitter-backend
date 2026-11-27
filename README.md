
# DubPad Backend

The DubPad backend is a Rust API service for audio separation and beat/downbeat analysis.

It accepts audio uploads, runs a persistent HTDemucs worker, analyzes audio using Beat This!, and stores complete synchronized audio stems and analysis metadata.

The backend uses:

- **Rust** — API, job orchestration, storage, and beat analysis.
- **Axum** — HTTP API and WebSocket communication.
- **PostgreSQL** — Persistent projects, jobs, and analysis metadata.
- **Python** — Persistent HTDemucs inference worker.
- **PyTorch / Demucs** — Audio stem separation.
- **Beat This! / RTen** — Beat and downbeat detection.
- **Filesystem storage** — Original audio and generated WAV stems.

The frontend performs section slicing locally using the original sample coordinates and saved beat-grid information. The backend stores complete audio files rather than generating chunk files.

---

## Installation

### Requirements

- Rust toolchain and Cargo
- Python 3
- PostgreSQL
- Git

### 1. Clone the repository

```bash
git clone <repository-url>
cd soniclab-backend
```

### 2. Configure the database

Create a PostgreSQL database and configure the connection string in `.env`.

```bash
cp .env.example .env
```

Set:

```dotenv
DATABASE_URL=postgres://username:password@localhost:5432/dubpad
```

### 3. Run the DubPad setup manager

```bash
cargo run --bin dubpad-setup -- --app-root . install
```

The installer automatically:

1. Detects the operating system and architecture.
2. Locates and validates an existing compatible Demucs Python environment.
3. Reuses a compatible environment or creates a managed environment.
4. Installs the pinned Python dependencies when necessary.
5. Downloads and prepares the HTDemucs model.
6. Downloads the Beat This! model files.
7. Verifies the required worker and model files.
8. Starts the Demucs worker in offline mode to verify successful initialization.

Existing unmanaged Python environments are not modified.

### 4. Start the API

```bash
cargo run --bin dubpad-api
```

The API uses the configured `HOST` and `PORT`.

Default address:

```text
http://127.0.0.1:8787
```

Swagger UI:

```text
http://127.0.0.1:8787/swagger-ui
```

### Setup manager commands

```bash
# Complete installation
cargo run --bin dubpad-setup -- --app-root . install

# Prepare Python runtime
cargo run --bin dubpad-setup -- --app-root . runtime

# Download and verify models
cargo run --bin dubpad-setup -- --app-root . models

# Verify installation and worker startup
cargo run --bin dubpad-setup -- --app-root . verify
```

---

## Configuration

Environment variables are loaded by the backend configuration.

| Variable | Default | Description |
|---|---|---|
| `HOST` | `127.0.0.1` | API bind interface |
| `PORT` | `8787` | HTTP port |
| `DATABASE_URL` | Required | PostgreSQL connection string |
| `DATABASE_MAX_CONNECTIONS` | `10` | SQLx connection pool limit |
| `UPLOAD_DIR` | `./data/uploads` | Temporary upload storage |
| `AUDIO_JOB_TTL_SECS` | `86400` | Retention period for temporary uploads and terminal job metadata |
| `PROJECT_TTL_SECS` | `36000` | Sliding lifetime for server-managed projects (10 hours) |
| `MAX_UPLOAD_BYTES` | `1073741824` | Maximum upload size (1 GiB) |
| `JOB_CONCURRENCY` | `1` | Job concurrency; must remain one with the single-model worker |
| `JOB_POLL_INTERVAL_MS` | `500` | Durable queue polling interval |
| `DEMUCS_PYTHON` | Auto-detected | Explicit Python interpreter path |
| `DEMUCS_RUNTIME_DIR` | `data/runtime/demucs` | Managed Python environment directory |
| `DEMUCS_WORKER_SCRIPT` | `worker/demucs_worker.py` | Python worker script |
| `DUBPAD_APP_ROOT` | Compiled application root | Explicit application root |
| `DEMUCS_DEVICE` | `auto` | Processing device: `auto`, `cuda`, or `cpu` |
| `DEMUCS_ALLOW_CPU_FALLBACK` | `false` | Allow CPU fallback when automatic CUDA selection fails |
| `DEMUCS_STARTUP_TIMEOUT_SECS` | `600` | Worker model startup timeout |
| `DEMUCS_CANCEL_TIMEOUT_SECS` | `60` | Cancellation acknowledgement timeout |
| `BEAT_THIS_MEL_MODEL` | `models/mel_spectrogram.onnx` | Beat This! mel-spectrogram model path |
| `BEAT_THIS_MODEL` | `models/beat_this.onnx` | Beat This! beat/downbeat model path |
| `BEAT_THIS_THREADS` | Up to `16` | RTen inference worker threads |
| `BEAT_THIS_MODEL_VERSION` | Unset | Optional model version label |
| `CORS_ALLOW_ORIGINS` | `http://localhost:5173` | Comma-separated allowed browser origins |
| `RUST_LOG` | `soniclab_backend=info,tower_http=info` | Tracing filter |

`SONICLAB_TEST_DATABASE_URL` is an optional, separate database URL used only by the explicit persistence integration test.

The integration test uses a transaction and rolls back its changes. It requires a disposable test database with migrations already applied.

Do not point development or test environments at a production database.

---

## Runtime and Model Management

### Python runtime

The setup manager searches for a compatible Python environment before creating a new one.

The runtime setup checks:

1. `DEMUCS_PYTHON`, if configured.
2. Common existing virtual environment locations.
3. Python interpreters available through `PATH`.
4. The managed runtime directory.

A candidate interpreter must pass the Demucs and worker dependency checks.

If a compatible interpreter is found, it is reused without modifying the environment.

If no compatible interpreter exists, the installer creates an isolated environment under:

```text
data/runtime/demucs/
```

The managed environment is marked with:

```text
.dubpad-managed-runtime.json
```

The pinned worker dependencies are defined in:

```text
worker/requirements-demucs.txt
```

### HTDemucs

The installer downloads the HTDemucs model snapshot into the Hugging Face cache.

The worker verifies that the model configuration and required checkpoint files are available before loading the model.

After installation, the worker runs with Hugging Face offline mode enabled and blocks the Demucs `torch.hub` download fallback.

Model downloads are performed during installation, not during normal worker startup.

### Beat This!

Beat This! performs beat and downbeat detection through the Rust integration using `beat-this` 1.1.0 and its pure-Rust RTen runtime.

The model files are stored under:

```text
models/
├── beat_this.onnx
└── mel_spectrogram.onnx
```

The `.onnx` files are model formats. The backend does not require the ONNX Runtime shared library.

The installer downloads and verifies the required model files.

Model weights are not committed to the source repository. Verify upstream model-weight distribution rights before redistributing them.

### Device configuration

On Linux with a supported NVIDIA CUDA installation, the existing PyTorch environment can use CUDA.

For macOS, including Apple Silicon and Intel systems, use:

```dotenv
DEMUCS_DEVICE=cpu
```

The default `auto` mode does not silently fall back to CPU. To explicitly allow CPU fallback:

```dotenv
DEMUCS_ALLOW_CPU_FALLBACK=true
```

---

## API v1

OpenAPI specification:

```text
/api/v1/openapi.json
```

Swagger UI:

```text
/swagger-ui
```

Errors use the following structure:

```json
{
  "error": {
    "code": "...",
    "message": "..."
  }
}
```

### Health and projects

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health` | Liveness response |
| POST | `/api/v1/projects` | Create a project |
| GET | `/api/v1/projects/:project_id` | Retrieve project metadata |
| PUT | `/api/v1/projects/:project_id` | Save validated project arrangement and timeline state |
| GET | `/api/v1/projects/:project_id/analysis` | Retrieve saved beat-grid JSON |
| GET | `/api/v1/projects/:project_id/stems/:stem_name` | Stream a persistent WAV stem |
| GET | `/api/v1/projects/:project_id/audio/:asset_id` | Stream original project audio (Range supported) |
| GET | `/api/v1/projects/:project_id/segments` | Retrieve segment manifest, including progressive readiness |
| GET | `/api/v1/projects/:project_id/segments/:segment_id/:stem_name` | Stream one ready stem segment |
| GET | `/api/v1/system/hardware` | Retrieve the Demucs backend validated at startup |
| POST | `/api/v1/projects/:project_id/audio` | Upload original audio |
| GET | `/api/v1/projects/:project_id/audio` | List audio metadata |

Project creation accepts:

```json
{
  "name": "My Project",
  "description": "Optional project description"
}
```

Audio upload uses multipart form data:

- `file`
- `sample_rate`
- `channels`
- `frame_count`

### Jobs and separation

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/projects/:project_id/jobs` | Create a separation job |
| GET | `/api/v1/jobs/:job_id` | Retrieve job status |
| POST | `/api/v1/jobs/:job_id/cancel` | Cancel a queued or running job |
| GET | `/api/v1/jobs/:job_id/events` | WebSocket progress events |
| GET | `/api/v1/jobs/:job_id/stems` | List generated stems |
| GET | `/api/v1/jobs/:job_id/stems/:stem_name` | Stream a generated WAV stem |

Job creation supports:

```json
{
  "analysis_strategy": "drums_first",
  "beats_per_bar": 4
}
```

Supported analysis strategies:

- `mix_first`
- `drums_first`

### Beat analysis

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/jobs/:job_id/beat-analysis` | Queue Beat This! analysis |
| GET | `/api/v1/jobs/:job_id/beat-analysis` | Retrieve analysis status and result |
| GET | `/api/v1/jobs/:job_id/analysis` | Retrieve complete saved beat-grid JSON |

Beat analysis is a separate job from HTDemucs separation.

It does not enter or modify the Demucs queue.

The detector instance is cached and reused. WAV decoding and CPU inference run in a blocking worker thread.

The service records model loading, WAV validation, decoding, mel processing, beat inference, peak decoding, and total execution timings.

---

## Audio Processing Pipeline

The backend supports two analysis strategies.

### Mix-first

```text
Original Audio
      |
      v
Beat This! Analysis
      |
      v
HTDemucs Separation
      |
      v
Complete Stem Storage
      |
      v
Saved Beat Grid
```

Mix-first analyzes the uploaded input before separation.

Currently, mix-first requires an input WAV readable by `hound`. Compressed audio formats should use drums-first until a Rust decoder is added.

### Drums-first

```text
Original Audio
      |
      v
HTDemucs Separation
      |
      v
Drums Stem
      |
      v
Beat This! Analysis
      |
      v
Saved Beat Grid
```

Drums-first performs analysis against the completed drums stem.

### Stem output

The backend generates four synchronized WAV stems:

- Drums
- Bass
- Other
- Vocals

The original audio is retained as well.

Each output is restored to the original sample rate and exact frame count.

The worker validates the decoded input against uploaded metadata and rejects unsupported channel layouts other than mono and stereo.

### Storage

PostgreSQL stores:

- Project metadata
- Job metadata
- Stem storage keys
- Versioned beat-grid JSON
- Beat analysis records

Audio data is stored on disk rather than in PostgreSQL.

Default storage layout:

```text
data/
├── uploads/
├── projects/
└── runtime/
    └── demucs/
```

Successful project assets are retained. Temporary uploads and terminal job metadata are subject to the configured retention policy.

---

## Job Queue and Worker Lifecycle

The job queue is durable and PostgreSQL-backed.

A single Rust supervisor owns one persistent Python worker and claims jobs using:

```sql
FOR UPDATE SKIP LOCKED
```

The worker is kept alive between jobs to avoid repeatedly loading the HTDemucs model.

### Worker lifecycle

1. Rust starts the Python worker.
2. Python validates and loads HTDemucs.
3. The worker reports `ready`.
4. Rust submits separation jobs through JSON-lines communication.
5. The worker emits progress events.
6. Completed outputs are finalized and stored.
7. The worker remains alive for subsequent jobs.

### Recovery

If the worker crashes:

- The active attempt is marked failed.
- The supervisor restarts the worker with backoff.
- Queued jobs remain in PostgreSQL.

When the API restarts, interrupted jobs left in `RUNNING` state are requeued.

### Cancellation

Queued jobs can be cancelled directly.

Running jobs receive a cancellation request over the worker's JSON-lines stdin.

The worker acknowledges cancellation at a Demucs segment callback.

If acknowledgement times out, Rust kills the worker and marks the job `FAILED` rather than incorrectly reporting it as cancelled.

WebSocket events are transient. Clients should reconnect and retrieve the durable job state when necessary.

---

## Audio Storage and HTTP Range Requests

Successful projects retain the original audio and complete generated WAV stems.

Audio routes stream files in 64 KiB blocks.

Supported byte-range requests include:

- Closed ranges
- Open-ended ranges
- Suffix ranges

Successful partial requests return:

```text
206 Partial Content
```

With appropriate headers:

```text
Content-Range
Accept-Ranges
Content-Length
```

Invalid or unsatisfiable ranges return:

```text
416 Range Not Satisfiable
```

Project stem paths are resolved from database storage keys and checked against the project storage root.

### Stem metadata

Stem listing returns:

- `job_id`
- `job_status`
- `stems`

Each stem contains:

- `name`
- `sample_rate`
- `channels`
- `frame_count`
- `duration_seconds`
- `audio_format`
- `file_size_bytes`

The audio format is WAV.

---

## Beat Grid and Timing Contract

The backend defines the inference timing boundary through:

```text
models::audio_asset::InferenceTimingManifest
```

Integer source sample positions are authoritative.

`SampleRange` represents:

- Exact section core bounds
- Expanded processing bounds, including overlap
- Exact trim offsets back to the core region

Seconds are derived from sample positions and sample rate.

Output sample rate, frame count, stem source mapping, and measured processing offsets are represented explicitly.

Offsets remain absent until measured.

Implementations must preserve the frontend's overlap/core-region trimming behavior and original sample coordinates.

Do not independently round stem boundaries, silently stretch audio, or invent processing offsets.

### Beat-grid timestamps

The complete detector result is mapped into a versioned beat grid.

- Beat timestamps remain on the original song timeline.
- Sample positions are frame indices at the original sample rate.
- `sample_position` is calculated using `round(time_seconds * original_sample_rate)`.
- BPM is derived from detected beat intervals.
- `beat_number_in_bar` resets at detected downbeats.

The detector does not provide a reliable time signature or confidence value, so these fields are omitted.

---

## Frontend Integration

The frontend can request either mix-first or drums-first separation.

After successful processing, it can retrieve:

1. Project metadata
2. Saved beat-grid analysis
3. Original audio
4. Drums WAV
5. Bass WAV
6. Other WAV
7. Vocals WAV

The frontend performs section slicing locally using the original frame positions from the beat grid.

Backend chunk generation is disabled. The backend does not create new chunk records or chunk WAV files.

The frontend uses:

```dotenv
VITE_RUST_API_URL=http://127.0.0.1:8787
```

when the Rust backend is hosted separately from the frontend.

---

## Development

### Run the API

```bash
cargo run --bin dubpad-api
```

### Run the installer

```bash
cargo run --bin dubpad-setup -- --app-root . install
```

### Run tests

```bash
cargo test
```

`cargo test` does not read `DATABASE_URL` or apply migrations.

The optional persistence integration test uses `SONICLAB_TEST_DATABASE_URL`. It requires a disposable test database with migrations already applied.

### Build release binaries

```bash
cargo build --release --bin dubpad-api --bin dubpad-setup
```

---

## Licensing and Model Distribution

The Rust backend and its dependencies are subject to their respective licenses.

The Beat This! integration uses `beat-this` 1.1.0 and its pure-Rust RTen runtime.

Model weights are not included in the source repository. HTDemucs is downloaded through the Hugging Face model cache, while the Beat This! model files are provisioned by the setup manager.

Verify upstream model-weight licensing and redistribution rights before packaging or distributing model assets.
