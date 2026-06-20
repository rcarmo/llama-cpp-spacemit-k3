#!/usr/bin/env bash
set -euo pipefail

# Free any stale TCM blocks before launch (keeps IME2/TCM enabled).
if [ -x /home/me/bin/tcm-cleanup ]; then
  LD_LIBRARY_PATH=/home/me/lib:/usr/lib /home/me/bin/tcm-cleanup || true
fi

mkdir -p /home/me/gemma-e2b-slot-cache

# Gemma 4 E2B QAT + native MTP drafter — sane high-performance interactive profile.
#
# Sizing rationale (Milk-V SpaceMIT K3, 8 perf cores, IME2/TCM):
#   - ctx 32768: prefill cost on RISC-V degrades with prompt length; 32K is the
#     practical interactive ceiling and prevents a single huge prompt from
#     wedging the one slot for many minutes.
#   - threads 8: matches the 8 perf cores and the 8-block TCM pool (t<8 segfaults,
#     t>8 contends with efficiency cores / TCM blocks).
#   - f16 KV: ~8.4 tok/s vs ~6.7 tok/s for q8_0 on this board; KV is small for E2B.
#   - batch 2048 / ubatch 1024: high prefill throughput for the IME2 kernels.
#   - prompt cache: shared prefixes (system prompt + files) are free after the
#     first request — the key win for iterative coding.
exec /home/me/src/llama-cpp-ff-gemma4-mtp-ime2-work/build/bin/llama-server \
  --model /home/me/models/gguf-misc/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf \
  --model-draft /home/me/models/gguf-misc/mtp-gemma-4-E2B-it-qat-Q4_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  --alias gemma4-e2b-qat \
  --host 0.0.0.0 --port 8080 \
  --threads 8 --threads-batch 8 \
  --batch-size 2048 --ubatch-size 1024 \
  --ctx-size 32768 \
  --cache-type-k f16 --cache-type-v f16 \
  --parallel 1 --jinja \
  --cache-prompt \
  --slot-save-path /home/me/gemma-e2b-slot-cache \
  --reasoning off --reasoning-format none \
  --no-warmup
