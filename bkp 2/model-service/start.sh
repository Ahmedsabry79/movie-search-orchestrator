#!/usr/bin/env bash
set -euo pipefail

EMBEDDING_PORT="${EMBEDDING_PORT:-8001}"
GENERATOR_PORT="${GENERATOR_PORT:-8002}"
GATEWAY_PORT="${GATEWAY_PORT:-8000}"

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  kill 0 >/dev/null 2>&1 || true
  exit "$code"
}
trap cleanup EXIT INT TERM

embedding_args=(
  vllm serve "${EMBEDDING_MODEL:-BAAI/bge-m3}"
  --host 127.0.0.1
  --port "$EMBEDDING_PORT"
  --runner pooling
  --served-model-name "${EMBEDDING_SERVED_MODEL_NAME:-BAAI/bge-m3}"
  --gpu-memory-utilization "${EMBEDDING_GPU_MEMORY_UTILIZATION:-0.10}"
  --max-model-len "${EMBEDDING_MAX_MODEL_LEN:-8192}"
  --max-num-seqs "${EMBEDDING_MAX_NUM_SEQS:-16}"
  --api-key "${MODEL_API_KEY:-movie-agent-local}"
)

generator_args=(
  vllm serve "${GENERATOR_MODEL:-cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit}"
  --host 127.0.0.1
  --port "$GENERATOR_PORT"
  --served-model-name "${GENERATOR_SERVED_MODEL_NAME:-movie-agent-generator}"
  --gpu-memory-utilization "${GENERATOR_GPU_MEMORY_UTILIZATION:-0.80}"
  --max-model-len "${GENERATOR_MAX_MODEL_LEN:-32768}"
  --max-num-seqs "${GENERATOR_MAX_NUM_SEQS:-16}"
  --api-key "${MODEL_API_KEY:-movie-agent-local}"
  --enable-auto-tool-choice
)

if [[ -n "${GENERATOR_TOOL_CALL_PARSER:-}" ]]; then
  generator_args+=(--tool-call-parser "${GENERATOR_TOOL_CALL_PARSER}")
fi

if [[ -n "${GENERATOR_REASONING_PARSER:-}" ]]; then
  generator_args+=(--reasoning-parser "${GENERATOR_REASONING_PARSER}")
fi

# Optional escape hatch for model-specific vLLM switches.
if [[ -n "${GENERATOR_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${GENERATOR_EXTRA_ARGS} )
  generator_args+=("${extra[@]}")
fi

printf 'Starting embedding backend: %q ' "${embedding_args[@]}"; printf '\n'
"${embedding_args[@]}" &
EMBED_PID=$!

printf 'Starting generator backend: %q ' "${generator_args[@]}"; printf '\n'
"${generator_args[@]}" &
GEN_PID=$!

export EMBEDDING_UPSTREAM="http://127.0.0.1:${EMBEDDING_PORT}"
export GENERATOR_UPSTREAM="http://127.0.0.1:${GENERATOR_PORT}"
export GATEWAY_PORT
python3 /srv/model-service/gateway.py &
GATEWAY_PID=$!

# Fail the container if any of the three processes exits.
wait -n "$EMBED_PID" "$GEN_PID" "$GATEWAY_PID"
