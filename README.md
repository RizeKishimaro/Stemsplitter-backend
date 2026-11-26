# DubPad API

This service is the DubPad Rust API. It accepts audio, runs the persistent HTDemucs worker, analyzes the drums (or original mix when configured) with Beat This, and stores the original plus four complete synchronized stems under project storage. PostgreSQL stores project/job metadata, stem storage keys, and the versioned beat-grid JSON; WAV data is stored on disk, not in PostgreSQL. Backend chunk generation is disabled: the frontend slices the full stems locally using frame positions from the beat grid.

## Run locally

Install Rust and PostgreSQL, create an empty development database, copy `.env.example` to `.env`, set `DATABASE_URL`, then run `cargo run` from this directory. On startup the service applies the embedded SQLx migrations, creates the upload directory, removes incomplete `.uploading` files, and requeues jobs left in `RUNNING` state. Do not point development or tests at a production database. `cargo test` never reads `DATABASE_URL` or applies migrations.

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | Bind interface |
| `PORT` | `8787` | HTTP port |
| `DATABASE_URL` | required | PostgreSQL connection string |
| `DATABASE_MAX_CONNECTIONS` | `10` | SQLx pool limit |
| `UPLOAD_DIR` | `./data/uploads` | Upload staging directory; project assets are stored in sibling `data/projects` |
| `AUDIO_JOB_TTL_SECS` | `86400` | Retention limit for temporary uploads and terminal job metadata; successful project assets are retained |
| `MAX_UPLOAD_BYTES` | `1073741824` | Maximum streamed upload size (1 GiB) |
| `JOB_CONCURRENCY` | `1` | Must remain one while the single-model worker is enabled |
| `JOB_POLL_INTERVAL_MS` | `500` | Durable job queue poll interval |
| `DEMUCS_PYTHON` | `/home/rizekishimaro/.venvs/demucs/bin/python` | Existing Demucs environment executable |
| `DEMUCS_WORKER_SCRIPT` | `./worker/demucs_worker.py` | Persistent JSON-lines inference worker |
| `DEMUCS_DEVICE` | `auto` | `auto`, `cuda`, or explicit `cpu`; `cuda` fails startup when unavailable |
| `DEMUCS_ALLOW_CPU_FALLBACK` | `false` | Must be explicitly enabled to permit CPU when `auto` cannot see CUDA |
| `DEMUCS_STARTUP_TIMEOUT_SECS` | `600` | Model startup timeout |
| `DEMUCS_CANCEL_TIMEOUT_SECS` | `60` | Grace period for segment-boundary cancellation before killing the child |
| `BEAT_THIS_MEL_MODEL` | `./models/mel_spectrogram.onnx` | Path to Beat This! mel ONNX model |
| `BEAT_THIS_MODEL` | `./models/beat_this.onnx` | Path to Beat This! beat/downbeat ONNX model |
| `BEAT_THIS_THREADS` | up to `16` | RTen inference worker threads, capped by the machine's available parallelism |
| `BEAT_THIS_MODEL_VERSION` | unset | Optional known checkpoint/version label; never inferred from a filename |
| `CORS_ALLOW_ORIGINS` | `http://localhost:5173` | Comma-separated allowed browser origins |
| `RUST_LOG` | `soniclab_backend=info,tower_http=info` | Tracing filter |

`SONICLAB_TEST_DATABASE_URL` is an optional, separate URL used only by the explicit persistence integration test. The test wraps its insert/read in a transaction and rolls it back. It must reference a disposable test database with the migration already applied; the test itself does not run migrations.

## API v1

OpenAPI JSON is served at `/api/v1/openapi.json` and the Swagger UI at `/swagger-ui`.

Errors use `{ "error": { "code": "...", "message": "..." } }`.

| Method and path | Behavior |
| --- | --- |
| `GET /api/v1/health` | Liveness response |
| `POST /api/v1/projects` | Create `{ "name": "...", "description": "..." }` |
| `GET /api/v1/projects/:project_id` | Project, audio/job metadata, persistent stem availability, and saved beat grid |
| `GET /api/v1/projects/:project_id/analysis` | Retrieve the saved beat-grid JSON when reopening a project |
| `GET /api/v1/projects/:project_id/stems/:stem_name` | Stream a complete persistent WAV stem; supports HTTP byte ranges |
| `POST /api/v1/projects/:project_id/audio` | Multipart upload: `file`, `sample_rate`, `channels`, `frame_count` |
| `GET /api/v1/projects/:project_id/audio` | List audio metadata |
| `POST /api/v1/projects/:project_id/jobs` | Create a separate job with `analysis_strategy` (`mix_first` or `drums_first`) and optional `beats_per_bar` (default `4`) |
| `GET /api/v1/jobs/:job_id` | Read durable job state |
| `POST /api/v1/jobs/:job_id/cancel` | Cancel a queued or running job |
| `GET /api/v1/jobs/:job_id/events` | WebSocket; sends current state then matching progress events |
| `GET /api/v1/jobs/:job_id/stems` | List WAV stems and sample metadata for a successful job |
| `GET /api/v1/jobs/:job_id/stems/:stem_name` | Stream `drums`, `bass`, `other`, or `vocals` WAV; supports a single `Range: bytes=start-end` request and returns `206` / `Content-Range` when partial |
| `POST /api/v1/jobs/:job_id/beat-analysis` | Queue a separate Beat This! analysis against the successful job's drums WAV (202 Accepted) |
| `GET /api/v1/jobs/:job_id/beat-analysis` | Read the newest analysis record and its status/result |
| `GET /api/v1/jobs/:job_id/analysis` | Retrieve the complete saved beat-grid JSON |

Beat analysis has its own `beat_analyses` table and does not enter or alter the HTDemucs queue. Detector instances are cached and reused. CPU inference and WAV decode run in a blocking worker thread. The service records model-load, WAV validation/decode, stream push/mel, beat inference, peak decode, and total elapsed timings in analysis metadata/logs. RTen's Rayon pool is bounded by `BEAT_THIS_THREADS` (default up to 16 and capped by available parallelism) to avoid exceeding the machine's CPU parallelism alongside the sequential Demucs worker. Detector timestamps stay in seconds as returned, and `sample_position` is `round(time_seconds * original_sample_rate)`. BPM is derived from detected beat intervals. `beat_number_in_bar` resets at detected downbeats but does not imply a time signature. Meter and confidence are omitted because the crate does not provide them.

The integration uses `beat-this` 1.1.0 with its pure-Rust `rten` runtime and WAV/format decoding feature. It does not require the ONNX Runtime shared library. The published crate does not bundle models. Model weights are deliberately not included in this repository; deployment must supply both model files through the configured paths. The Rust port and original implementation declare MIT licensing, but the upstream release materials do not clearly establish separate model-weight distribution terms, so verify rights for the chosen checkpoint before redistributing model assets. A cached detector is reused for pipeline and explicit analysis jobs.

Stem listing returns `job_id`, `job_status`, and a `stems` array. Each stem includes `name`, `sample_rate`, `channels`, `frame_count`, `duration_seconds`, `audio_format` (`wav`), and `file_size_bytes`. The project endpoint also exposes persistent stem URLs and frame metadata. Audio routes stream files in 64 KiB blocks and support closed, open-ended, and suffix byte ranges; successful ranges return `206` with `Content-Range`, `Accept-Ranges`, and `Content-Length`. Invalid or unsatisfiable ranges return `416`. Project stem paths are resolved from database storage keys and checked against the project storage root.

Upload size is bounded both at the HTTP body layer and while streaming to a temporary upload file. At inference time the worker checks decoded sample rate, channel count, and frame count against uploaded metadata, rejects channel layouts other than mono or stereo, restores each output to the original sample rate and exact frame count, and records explicit tail adjustment metadata. On successful finalization, each full stem and the original are copied to same-directory temporary files, synced, and published atomically without replacing existing project assets. Stem references, original path, beat-grid JSON, and successful job state are committed together. Temporary inference output is removed after commit; failed output remains available for recovery/TTL cleanup. Existing legacy chunk tables and files are left untouched, but no active route or separation code creates new chunk records or WAVs.

The queue is durable in PostgreSQL, not an in-memory work queue. A single Rust supervisor owns one persistent Python child/model and claims jobs using `FOR UPDATE SKIP LOCKED`. Worker crashes fail the active attempt and the supervisor restarts with backoff; queued jobs remain in PostgreSQL. At API restart, interrupted `RUNNING` jobs are requeued. Running cancellation sets `cancel_requested`, sends a command over the worker's JSON-lines stdin, and marks the row `CANCELLED` only after the worker acknowledges cancellation at a Demucs segment callback. If acknowledgement times out, Rust kills the worker and marks the job `FAILED`, never falsely `CANCELLED`. The WebSocket stream is transient; clients can reconnect and fetch durable state.

The worker first verifies the HTDemucs config and every checkpoint file are already present in the Hugging Face cache. It sets Hugging Face offline mode and blocks Demucs' `torch.hub` download fallback. Startup emits the actual device and CUDA visibility. On the current managed shell, the requested venv has CUDA-enabled PyTorch but the process cannot communicate with the NVIDIA driver, so `torch.cuda.is_available()` is false; with the default no-fallback policy the worker correctly refuses readiness. Run the API from the normal terminal where the RTX 4080 is visible, or explicitly configure CPU if that is intended.

## Timing and browser pipeline boundary

`models::audio_asset::InferenceTimingManifest` defines the intended inference boundary. Integer source sample positions are authoritative. `SampleRange` carries exact section core bounds, expanded processing bounds (including overlap), and exact trim offsets back to the core. Seconds are derived from samples and sample rate. Output sample rate/frame count, each stem's source mapping, and measured processing offsets are represented explicitly; offsets remain absent until measured. Implementations must preserve the frontend's overlap/core-region trimming behavior and original sample coordinates, never independently round stem boundaries, stretch audio silently, or invent offsets.

The browser can request either a mix-first or drums-first HTDemucs job. Mix-first analyzes the uploaded input before separation; drums-first analyzes the completed drums output. The complete detector result is mapped to a versioned beat grid whose timestamps remain on the original song timeline and whose sample positions are frame indices at the original sample rate. The full files are available at the project stem endpoints after the job succeeds. The frontend can fetch the project metadata, analysis, and four WAVs and perform all section slicing locally. Mix-first currently requires an input WAV readable by `hound`; compressed formats should use drums-first until a Rust decoder is added. The web app uses `VITE_RUST_API_URL` when the Rust backend is hosted separately from the frontend.
