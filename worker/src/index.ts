import { parse as parseYaml } from "yaml";
import { validate } from "./validate";

export interface Env {
  DB: D1Database;
}

const JSON_HEADERS = {
  "Content-Type": "application/json",
  "Access-Control-Allow-Origin": "*",
};

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          ...JSON_HEADERS,
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        },
      });
    }

    if (request.method === "GET" && url.pathname === "/health") {
      return Response.json({ status: "ok", schema_version: 1 });
    }

    if (request.method === "POST" && url.pathname === "/submit/cli") {
      return handleSubmit(request, env);
    }

    return Response.json({ error: "Not found" }, { status: 404, headers: JSON_HEADERS });
  },
};

async function handleSubmit(request: Request, env: Env): Promise<Response> {
  let body: string;
  try {
    body = await request.text();
  } catch {
    return err(400, "Failed to read request body");
  }

  if (!body.trim()) return err(400, "Empty request body");

  let data: Record<string, unknown>;
  try {
    data = parseYaml(body) as Record<string, unknown>;
  } catch {
    return err(400, "Invalid YAML");
  }

  if (!data || typeof data !== "object") return err(400, "YAML must be a mapping");

  const validationError = validate(data);
  if (validationError) return err(422, validationError);

  const id = crypto.randomUUID();
  const receivedAt = new Date().toISOString();

  const model  = data.model  as Record<string, unknown>;
  const system = data.system as Record<string, unknown>;
  const meas   = data.measurement as Record<string, unknown>;
  const sub    = data.submitter   as Record<string, unknown>;
  const integ  = data.integrity   as Record<string, unknown> | undefined;

  const median = meas.median as Record<string, unknown> | undefined;
  const tps = typeof median?.tokens_per_second === "number"
    ? median.tokens_per_second
    : null;

  try {
    await env.DB.prepare(`
      INSERT INTO submissions (
        id, submitted_at, received_at, schema_version,
        workload, engine, engine_version, profile, mode,
        model_name, model_source, model_source_ref, model_file_hash,
        gpu_vendor, gpu_name, gpu_chip, vram_gb, runtime, os, kernel,
        tokens_per_second, throttling_detected, runs, warmup_runs,
        integrity_hash, anonymous_id, raw_yaml
      ) VALUES (
        ?, ?, ?, ?,
        ?, ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?
      )
    `).bind(
      id, data.submitted_at, receivedAt, data.schema_version ?? 1,
      data.workload, data.engine, data.engine_version, data.profile, data.mode,
      model.name, model.source, model.source_ref, model.file_hash ?? null,
      system.gpu_vendor, system.gpu_name, system.gpu_chip,
      system.vram_gb, system.runtime, system.os, system.kernel,
      tps, meas.throttling_detected ? 1 : 0,
      meas.runs, meas.warmup_runs,
      integ?.hash ?? null, sub.anonymous_id, body,
    ).run();
  } catch (e) {
    console.error("D1 insert failed:", e);
    return err(500, "Database error");
  }

  return Response.json({ id, received_at: receivedAt }, { status: 201, headers: JSON_HEADERS });
}

function err(status: number, message: string): Response {
  return Response.json({ error: message }, { status, headers: JSON_HEADERS });
}
