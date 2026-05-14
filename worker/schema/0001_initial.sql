CREATE TABLE IF NOT EXISTS submissions (
    id                  TEXT PRIMARY KEY,
    submitted_at        TEXT NOT NULL,
    received_at         TEXT NOT NULL,
    schema_version      INTEGER NOT NULL DEFAULT 1,
    workload            TEXT NOT NULL,
    engine              TEXT NOT NULL,
    engine_version      TEXT NOT NULL,
    profile             TEXT NOT NULL,
    mode                TEXT NOT NULL,
    model_name          TEXT NOT NULL,
    model_source        TEXT NOT NULL,
    model_source_ref    TEXT NOT NULL,
    model_file_hash     TEXT,
    gpu_vendor          TEXT NOT NULL,
    gpu_name            TEXT NOT NULL,
    gpu_chip            TEXT NOT NULL,
    vram_gb             REAL NOT NULL,
    runtime             TEXT NOT NULL,
    os                  TEXT NOT NULL,
    kernel              TEXT NOT NULL,
    tokens_per_second   REAL,
    throttling_detected INTEGER NOT NULL DEFAULT 0,
    runs                INTEGER NOT NULL,
    warmup_runs         INTEGER NOT NULL,
    integrity_hash      TEXT,
    anonymous_id        TEXT NOT NULL,
    raw_yaml            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sub_workload_profile ON submissions(workload, profile);
CREATE INDEX IF NOT EXISTS idx_sub_gpu_chip         ON submissions(gpu_chip);
CREATE INDEX IF NOT EXISTS idx_sub_submitted_at     ON submissions(submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_sub_anonymous_id     ON submissions(anonymous_id);
