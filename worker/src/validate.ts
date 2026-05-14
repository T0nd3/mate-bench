const REQUIRED_TOP = [
  "schema_version", "submitted_at", "mode", "workload", "engine",
  "engine_version", "model", "profile", "test_set", "engine_config",
  "system", "bench", "measurement", "submitter",
] as const;

const VALID_MODES    = new Set(["closed", "open"]);
const VALID_PROFILES = new Set(["quick", "standard", "full"]);
const VALID_VENDORS  = new Set(["amd", "nvidia", "intel", "cpu"]);
const VALID_SOURCES  = new Set(["ollama", "huggingface", "local"]);

export function validate(data: Record<string, unknown>): string | null {
  for (const field of REQUIRED_TOP) {
    if (data[field] === undefined || data[field] === null)
      return `Missing required field: ${field}`;
  }

  if (!VALID_MODES.has(data.mode as string))
    return `Invalid mode: ${data.mode}`;

  if (!VALID_PROFILES.has(data.profile as string))
    return `Invalid profile: ${data.profile}`;

  const model = data.model as Record<string, unknown>;
  if (!model?.name || !model?.source || !model?.source_ref)
    return "model must have name, source, source_ref";
  if (!VALID_SOURCES.has(model.source as string))
    return `Invalid model.source: ${model.source}`;

  const system = data.system as Record<string, unknown>;
  const sysRequired = ["gpu_vendor", "gpu_name", "gpu_chip", "vram_gb", "runtime", "driver", "os", "kernel"];
  for (const f of sysRequired) {
    if (system?.[f] === undefined || system?.[f] === null)
      return `Missing system.${f}`;
  }
  if (!VALID_VENDORS.has(system.gpu_vendor as string))
    return `Invalid system.gpu_vendor: ${system.gpu_vendor}`;

  const m = data.measurement as Record<string, unknown>;
  if (!m) return "Missing measurement";
  if (typeof m.runs !== "number" || m.runs < 1)
    return "measurement.runs must be a positive integer";
  if (typeof m.warmup_runs !== "number" || m.warmup_runs < 0)
    return "measurement.warmup_runs must be >= 0";
  if (typeof m.throttling_detected !== "boolean")
    return "measurement.throttling_detected must be a boolean";

  const sub = data.submitter as Record<string, unknown>;
  if (!sub?.anonymous_id || typeof sub.anonymous_id !== "string")
    return "submitter.anonymous_id is required";

  return null;
}
