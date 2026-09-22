# MLflow with Docker Compose (PostgreSQL + RustFS)

This directory provides a **Docker Compose** setup for running **MLflow** locally with a **PostgreSQL** backend store and **RustFS** (S3-compatible) artifact storage, following MLflow's current reference Compose setup. It's intended for quick evaluation and local development.

---

## Overview

- **MLflow Tracking Server** — exposed on your host (default `http://localhost:5000`).
- **PostgreSQL** — persists MLflow's metadata (experiments, runs, params, metrics).
- **RustFS** — stores run artifacts via an S3-compatible API (console on `http://localhost:9001`).
- **create-bucket** — one-shot AWS CLI container that creates the artifact bucket if it doesn't exist yet.

All published ports (PostgreSQL, RustFS, MLflow) are bound to `127.0.0.1` only. From another machine, use an SSH tunnel, e.g. `ssh -L 5000:127.0.0.1:5000 <host>`.

Compose automatically reads configuration from a local `.env` file in this directory.

---

## Prerequisites

- **Git**
- **Docker** and **Docker Compose**
  - Windows/macOS: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - Linux: Docker Engine + the `docker compose` plugin

Verify your setup:

```bash
docker --version
docker compose version
```

---

## 1. Clone the Repository

```bash
git clone https://github.com/mlflow/mlflow.git
cd docker-compose
```

---

## 2. Configure Environment

Copy the example environment file and modify as needed:

```bash
cp .env.dev.example .env
```

The `.env` file defines container image tags, ports, credentials, and storage configuration. Open it and review values before starting the stack.

**Common variables** :

- **MLflow**
  - `MLFLOW_PORT=5000` — host port for the MLflow UI/API
  - `MLFLOW_ARTIFACTS_DESTINATION=s3://mlflow/` — artifact store URI
  - `MLFLOW_S3_ENDPOINT_URL=http://storage:9000` — S3 endpoint (inside the Compose network)
  - `MLFLOW_HOSTS` — comma-separated hosts accepted by `--allowed-hosts`
  - `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS` — origins accepted by `--cors-allowed-origins` (required since MLflow 3.10)
- **PostgreSQL**
  - `POSTGRES_USER=mlflow`
  - `POSTGRES_PASSWORD` — set your own
  - `POSTGRES_DB=mlflow`
  - `PGPORT=5432`
- **RustFS (S3-compatible)**
  - `AWS_ACCESS_KEY_ID` — RustFS access key, also used by MLflow and the bucket job
  - `AWS_SECRET_ACCESS_KEY` — RustFS secret key; set your own
  - `AWS_DEFAULT_REGION=us-east-1`
  - `S3_BUCKET=mlflow` — bucket created on startup
  - `RUSTFS_CONSOLE_ENABLE=true` — optional; set to `false` to disable the web console

---

## 3. Launch the Stack

```bash
docker compose up -d
```

This:

- Builds/pulls images as needed
- Creates a user-defined network
- Starts **postgres**, **storage** (RustFS), **create-bucket**, and **mlflow** containers

Check status:

```bash
docker compose ps
```

View logs (useful on first run):

```bash
docker compose logs -f
```

---

## 4. Access MLflow

Open the MLflow UI:

- **URL**: `http://localhost:5000` (or the port set in `.env`)

You can now create experiments, run training scripts, and log metrics, parameters, and artifacts to this local MLflow instance.

---

## 5. Shutdown

To stop and remove the containers and network:

```bash
docker compose down
```

> Data is preserved in Docker **volumes**. To remove volumes as well (irreversible), run:
>
> ```bash
> docker compose down -v
> ```

---

## Tips & Troubleshooting

- **Verify connectivity**  
  If MLflow can't write artifacts, confirm your S3 settings:

  - `MLFLOW_ARTIFACTS_DESTINATION` points to your bucket (e.g., `s3://mlflow/`)
  - `MLFLOW_S3_ENDPOINT_URL` is reachable from the MLflow container (`http://storage:9000`)
  - The bucket job finished successfully: `docker compose logs create-bucket`

- **Resetting the environment**  
  If you want a clean slate, stop the stack and remove volumes:

  ```bash
  docker compose down -v
  docker compose up -d
  ```

- **Logs**

  - MLflow server: `docker compose logs -f mlflow`
  - PostgreSQL: `docker compose logs -f postgres`
  - RustFS: `docker compose logs -f storage`

- **Port conflicts**  
  If `5000` (or any other port) is in use, change it in `.env` and restart:
  ```bash
  docker compose down
  docker compose up -d
  ```

---

## How It Works (at a Glance)

- MLflow uses **PostgreSQL** as the _backend store_ for experiment/run metadata.
- MLflow uses **RustFS** as the _artifact store_ via S3 APIs, and proxies artifact access for clients (`--serve-artifacts`).
- Docker Compose wires services on a shared network; MLflow talks to PostgreSQL and RustFS by service name (`postgres`, `storage`).

---

## Upgrading MLflow

`MLFLOW_VERSION` in `.env` is pinned on purpose. A newer MLflow release may ship database schema changes, and the server then refuses to start with `Detected out-of-date database schema`. To upgrade:

```bash
# 1. Set the new MLFLOW_VERSION in .env, then pull it
docker compose pull mlflow

# 2. Back up the database (backups/ is git-ignored)
docker compose up -d --wait postgres
mkdir -p backups
docker exec mlflow-postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' \
  > "backups/mlflow-$(date +%Y%m%d-%H%M%S).dump"

# 3. Migrate the schema with the new image
docker compose run --rm --no-deps --entrypoint /bin/bash mlflow -c \
  'pip install -q psycopg2-binary && mlflow db upgrade "$MLFLOW_BACKEND_STORE_URI"'

# 4. Start the stack
docker compose up -d
```

To roll back, restore the dump with `pg_restore --clean` and go back to the previous `MLFLOW_VERSION`.

---

## Migrating from the Previous MinIO Setup

Earlier versions of this stack used MinIO with the `minio-data` volume and `MINIO_*` variables.

- **Variables:** rename `MINIO_ROOT_USER` → `AWS_ACCESS_KEY_ID`, `MINIO_ROOT_PASSWORD` → `AWS_SECRET_ACCESS_KEY`, `MINIO_BUCKET` → `S3_BUCKET`; drop `MINIO_HOST`/`MINIO_PORT`; set `MLFLOW_S3_ENDPOINT_URL=http://storage:9000`. Compare with `.env.dev.example`.
- **Artifacts:** RustFS starts with an empty `storage-data` volume. Run metadata in PostgreSQL is kept, but existing artifacts stay in `minio-data` until you copy them over (e.g., start the old MinIO once and `aws s3 sync` its bucket into RustFS).
- **Old container:** run `docker compose up -d --remove-orphans` to remove the leftover `mlflow-minio` container.

---

## Next Steps

- Point your training scripts to this server:
  ```bash
  export MLFLOW_TRACKING_URI=http://localhost:5000
  ```
- Start logging runs with `mlflow.start_run()` (Python) or the MLflow CLI.
- Customize the `.env` and `docker-compose.yml` to fit your local workflow (e.g., change image tags, add volumes, etc.).

---

**You now have a fully local MLflow stack with persistent metadata and artifact storage—ideal for development and experimentation.**
