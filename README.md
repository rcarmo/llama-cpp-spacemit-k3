# llama.cpp SpaceMIT/TurboQuant K3 fork

## Gemma 4 E2B QAT — Sane High-Performance Interactive Profile

Fast, low-RAM interactive/coding server. Use when context capacity matters less
than latency and throughput:

```sh
scripts/run-gemma-e2b-qat-server.sh
```

Validated live on Milk-V SpaceMIT K3:

| Setting | Value |
|---|---|
| Model | `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` |
| Draft model | `mtp-gemma-4-E2B-it-qat-Q4_0.gguf` |
| Context | `32768` |
| Threads | `8` / `threads-batch=8` |
| Batch / uBatch | `2048` / `1024` |
| KV cache | `f16` K/V |
| MTP | enabled, `draft-mtp`, `n_max=4` |
| Prompt cache | enabled, 8 GiB budget |

Measured smoke (cold): **~82 tok/s prefill, ~27.5 tok/s generation, MTP acceptance 0.80**.
Warm request with shared prefix: prompt-cache hit, ~1 s round trip.

Sizing rationale:

- **32K context, not 128K.** Prompt-processing (prefill) cost on this RISC-V board
  degrades with prompt length, and the server runs a single slot. A 90K+ token
  prompt takes many minutes to prefill and blocks every other request until it
  finishes. 32K is the practical interactive ceiling.
- **threads 8** matches the 8 performance cores and the 8-block TCM pool (t<8
  segfaults; t>8 contends with efficiency cores / TCM blocks).
- **f16 KV** is ~8.4 tok/s vs ~6.7 tok/s for q8_0 on this board; KV is small for E2B.
- **Prompt cache** makes a repeated system-prompt + file prefix free after the
  first request.
- `--cache-reuse` (KV shifting) is auto-disabled by llama-server under `draft-mtp`,
  so it is omitted.

## Gemma 4 26B A4B QAT — Optimal Server Configuration

Based on llama-bench sweep (96 runs) and context-size testing:

```sh
/home/me/run-gemma-26b-a4b-qat-server.sh
# or:
llama-server \
  --model gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  --model-draft mtp-gemma-4-26B-A4B-it-qat-Q4_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  --threads 8 --ubatch-size 256 \
  --ctx-size 16384 --cache-type-k f16 --cache-type-v f16 \
  --reasoning off --reasoning-format deepseek
```

**Why these settings:**
- `t=8`: only the 8 A100/IME2 preferred cores — t≥12 causes TCM contention (SIGABRT)
- `ctx=16384`: tg tok/s is flat 4K→16K (~8.5 tok/s); drops at 32K and OOMs at 64K
- `ctk=f16`: better pp AND tg than q8_0 for this MoE model; q8_0 is consistently ~1.8 tok/s slower
- `ub=256`: no meaningful difference vs 128/512/1024 for this model
- `reasoning-format deepseek`: prevents 26B from leaking `<|channel>thought` into response body

**Context size vs tg tok/s (26B A4B, t=8 ub=256):**

| ctx | ctk | tg tok/s | mem |
|---|---|---:|---|
| 4K | f16 | 8.46 | 820 MiB |
| 4K | q8_0 | 6.66 | 745 MiB |
| 8K | f16 | 8.47 | 902 MiB |
| 8K | q8_0 | 6.70 | 809 MiB |
| 16K | **f16** | **8.36** | **1066 MiB** | ← sweet spot |
| 16K | q8_0 | 6.78 | 935 MiB |
| 32K | f16 | ~6.1 | ~2 GiB | OOM risk |
| 64K | — | OOM | — | kills the machine |

**Practical capacity at 16K**: ~12,000 words / 10–15 source files / full function-level code context.

## Gemma 4 QAT + MTP Benchmarks (SpaceMIT IME2/TCM, A100 cores)

**Build:** `-DGGML_CPU_RISCV64_SPACEMIT=ON -DGGML_RV_ZBA=ON -DGGML_RV_ZFH=ON -DGGML_RV_ZVFH=ON`

**Runtime evidence:**
```
CPU_RISCV64_SPACEMIT: tcm is available, blk_size: 393216
CPU_RISCV64_SPACEMIT: perfer_core_arch_id: a064, use_ime2: 1, mem_backend: HPAGE
```

**Server benchmark — complex agent prompt, no token limit, `finish_reason=stop`:**

| Model | Prefill tok/s | Gen tok/s | MTP acc | Coherence |
|---|---:|---:|---:|---|
| Gemma 4 E2B QAT UD-Q4_K_XL | 93.14 | 12.93 | 0.306 | pass |
| Gemma 4 E4B QAT UD-Q4_K_XL | 55.37 | 8.52 | 0.336 | pass |
| Gemma 4 12B QAT UD-Q4_K_XL | 20.72 | 4.32 | 0.429 | pass |

**Server benchmark — agent prompt (synthetic, for reproducibility):**

> The benchmark uses a **fixed synthetic agent prompt** designed to elicit structured,
> multi-section output from a coding assistant. It is not a real user query.
> It was chosen to produce sufficiently long, coherent output (finish_reason=stop)
> to measure generation throughput at realistic token counts (~800–900 tokens).

```
system: "You are a senior local coding agent. Return only the final answer.
         Do not include hidden reasoning, scratchpad, tool calls, function-call
         JSON, or channel tags. Be concrete and internally consistent."

user:   "We need to clean up a provider abstraction layer in a mixed
         TypeScript/C++ AI runtime. Produce a concise but complete
         implementation plan that: identifies likely dead provider code,
         separates provider registry from transport clients, preserves
         backwards compatibility for existing model IDs, adds tests and
         telemetry, handles rollback, and lists the top risks.
         Use numbered sections and concrete validation steps."

parameters: temperature=0.2, no max_tokens limit
```

> **Coherence grading**: pass = finish_reason=stop + no leaked thought/channel tags
> + ≥7/10 on-topic terms + structured numbered sections.

**llama-bench sweep — best settings per model (TCM enabled, `tcm-cleanup` before each run):**

| Model | Best pp tok/s | Settings | Best tg tok/s | Settings |
|---|---:|---|---:|---|
| Gemma 4 E2B QAT | 120.90 | t=8 ub=256 ctk=f16 | 13.36 | t=8 ub=512 ctk=f16 |
| Gemma 4 E4B QAT | 70.54 | t=8 ub=256 ctk=f16 | 8.02 | t=8 ub=256 ctk=f16 |
| Gemma 4 12B QAT | 29.06 | t=8 ub=256 ctk=f16 | 3.65 | t=8 ub=256 ctk=f16 |

**Notes:**
- t=8 (8 A100/IME2 preferred cores) consistently best; t=12/16 causes TCM contention
- f16 KV cache consistently best; q8_0 marginal or slower
- Smaller ubatch (256) wins on pp; 512 wins on tg for E2B
- Full sweep: `benchmarks/spacemit-speedup-bench-*.tsv`
- Bench script: `scripts/run-speedup-bench.sh` (calls `tcm-cleanup` before each run)


This repository is the source tree used for the Milk-V Jupiter 2 / SpacemiT K3 local LLM server experiments documented on [taoofmac.com](https://taoofmac.com/). It is a squashed `llama.cpp` fork with the SpaceMIT CPU backend, TurboQuant/KV-cache work, and a small runtime robustness patch for current Bianbu/K3 systems.

It is **not** a clean upstream branch and should be read as a working vendor fork snapshot: useful if you want to reproduce the K3 results, inspect the SpaceMIT integration points, or run the same OpenAI-compatible server on the board.

## Hardware and runtime target

The test system is a Milk-V Jupiter 2 class board based on the SpacemiT K3 SoC:

- 16 RISC-V harts total.
- User shell/cgroup-visible efficiency cores: `0-7`.
- A100/IME2 AI cores: `8-15`.
- 32 GB RAM, about 31 GiB visible to Linux.
- No swap in the test setup.
- Bianbu 4.0 / Linux 6.18.3 generic kernel.
- GCC 15 RISC-V toolchain.

A normal login shell cannot simply `taskset` onto cores `8-15`. The SpaceMIT path registers AI worker threads through `/proc/set_ai_thread` and then pins them to the A100 cores. When the backend is active, startup logs contain lines like:

```text
CPU_RISCV64_SPACEMIT: tcm is available, blk_size: 393216, blk_num: 8, is_fake_tcm: 0
CPU_RISCV64_SPACEMIT: num_cores: 16, num_perfer_cores: 8, perfer_core_arch_id: a064, exclude_main_thread: 0, use_ime1: 0, use_ime2: 1, mem_backend: HPAGE, cpu_mask: ff00, aicpu_id_offset: 8
```

The important part is `cpu_mask: ff00`, which maps to cores `8-15`.

## What this fork contains

The baseline commit is a squashed TurboQuant/SpaceMIT `llama.cpp` tree:

```text
377742a TurboQuant fork (feature/turboquant-kv-cache) with SpaceMIT IME2 support
```

The local K3 robustness patch is:

```text
7738625 spacemit: fall back when tcm sync shm is unavailable
```

That patch handles boards where `/dev/tcm_sync_mem` is absent. The original backend expected that device node for shared TCM barrier state. Current systems can log:

```text
open(/dev/tcm_sync_mem) failed, errno=2
```

The patch falls back to anonymous shared memory for the init barrier, which makes this non-fatal:

```text
CPU_RISCV64_SPACEMIT: alloc_chunk: /dev/tcm_sync_mem unavailable (errno=2), using anonymous shared memory
```

This is the only source tweak committed after publishing the fork. The rest of the work described below is configuration, benchmarking, model selection, and validation.

## Build notes

On the K3 board, the working tree lives at:

```text
/home/me/src/llama-cpp-turboquant-feature-turboquant-kv-cache
```

The production binary used in testing is:

```text
build/bin/llama-server
```

The production server wrapper is:

```text
/home/me/run-qwen-reap-server.sh
```

It performs a stale TCM block cleanup before launching the router:

```bash
#!/usr/bin/env bash
set -euo pipefail

if [ -x /home/me/bin/tcm-cleanup ]; then
  LD_LIBRARY_PATH=/usr/lib /home/me/bin/tcm-cleanup || true
fi

exec /home/me/src/llama-cpp-turboquant-feature-turboquant-kv-cache/build/bin/llama-server \
  --models-preset /home/me/models/qwen-reap-models.ini \
  --host 0.0.0.0 \
  --port 8080 \
  --models-max 1 \
  --cache-reuse 256
```

## OpenAI-compatible router setup

The production endpoint is:

```text
http://<k3-board>:8080/v1
```

The model preset file is:

```text
/home/me/models/qwen-reap-models.ini
```

The effective global settings used for most tests were:

```ini
[*]
threads = 8
threads-batch = 8
ubatch-size = 1024
ctx-size = 262144
parallel = 1
jinja = true
reasoning = off
reasoning-format = none
cache-reuse = 256
spec-type = ngram-simple
cache-ram = 8192
cache-idle-slots = true
ctx-checkpoints = 32
slot-save-path = /home/me/models/slot-cache
```

Important observations:

- `threads = 8` maps to the 8 preferred A100/IME2 workers.
- `f16/f16` KV cache is fastest and most reliable on the tested Gemma/Qwen paths.
- KV quantization (`q8_0` or `q4_*`) generally hurt performance or failed context creation on the tested Gemma 4 QAT models.
- `ngram-simple` speculative decoding is worth keeping: it is a large win on copy-heavy/repetitive prompts and has negligible impact on freeform prompts.

## Models kept or tested

The final useful model set on the board was:

| Model id / route | Model file | Why it exists |
|---|---|---|
| `qwen-reap-fast` | `Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf` | quality anchor, large context, sparse active-parameter behavior |
| `qwen-reap-longctx` | same model | long-context variant |
| `qwen-small-test` | `qwen3-0.6b-q4_k_m.gguf` | very fast toy/small utility model |
| `gemma4-e2b-qat` | `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` | best fast-but-useful local agent tier |
| `gemma4-e4b` | `gemma-4-E4B-it-Q4_K_M.gguf` | older compact fallback |

Additional test files that were downloaded during benchmarking:

| File | Notes |
|---|---|
| `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | tested and useful; not yet necessarily part of production presets |
| `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` | tested; too slow and chat-template behavior was messy |
| `gemma-4-E2B_q4_0-it.gguf` | official Google E2B text GGUF used for multimodal tests |
| `gemma-4-E2B-it-mmproj.gguf` | official Google projector, about 942 MB |
| `gemma-4-E2B-it-qat-mmproj-F16.gguf` | Unsloth projector, about 940 MB |

## Benchmark summary

All benchmark numbers below were gathered on the K3 board using the SpaceMIT/TurboQuant CPU backend unless otherwise noted. They should be treated as practical local measurements, not upstream benchmark claims.

### Practical routing table

| Model / route | Params (M) | RAM / file | Bench prefill | Bench decode | Server prefill | Server decode | Practical role |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen 3 0.6B | 596 | 373 MB | - | - | 37.5 | **43.5** | fastest; simple transforms and scaffolding |
| **Gemma 4 E2B QAT** | 4,630 | 2.5 GB | **114.4** | **12.69** | **99.6** | **12.9** | useful fast tier; local agent/tool use |
| **Gemma 4 E4B QAT** | 7,460 | 4.0 GB | 68.2 | 7.76 | 55.7 | 7.58 | middle-quality QAT tier; smaller than Qwen-REAP |
| Qwen3.6-28B-REAP-A3B | 28,240 | 17.3 GB | - | - | 28.9 | 7.15 | quality anchor; large context and harder coding |
| Gemma 4 E4B | 7,520 | 4.9 GB | 30.6 | 6.2 | 27.5 | 6.01 | older fallback, superseded by E4B QAT |
| Gemma 4 12B QAT UD-Q4_K_XL | 11,910 | 6.3 GB | 27.5 | 3.54 | 25.0 | 3.6 | not worth routing yet; template cleanup needed |

### Original model sweep

The first realistic agentic coding-turn sweep used an OpenAI-style chat prompt with a tool definition, prior tool output, and a request to generate an edit. Results were roughly:

| Model | Type / active | RAM | Prefill (t/s) | Decode (t/s) | Overall (t/s) | Turn |
|---|---|---:|---:|---:|---:|---:|
| **Qwen3.6-28B-REAP-A3B** | MoE / A3B | 17.3 GB | 29.1 | **6.5** | **11.5** | 140s |
| Gemma 4 E4B | dense / 4B-ish | 4.9 GB | 28.9 | 5.7 | 9.5 | 147s |
| Gemma 4 E2B QAT UD-Q4_K_XL | dense / 2B-ish label, 4.63B total | 2.5 GB | **99.6** | **12.9** | - | 18s/128 tok |
| Gemma 4 26B-A4B | MoE / A4B | 16.9 GB | **38.8** | 5.1 | 9.1 | 154s |
| Qwen 3.5-9B | dense / 9B | 5.6 GB | 22.5 | 4.5 | 8.2 | 195s |
| Gemma 4 12B | dense / 12B | 7.3 GB | 18.7 | 2.46 | 4.3 | 322s |
| Gemma 4 12B QAT UD-Q4_K_XL | dense / 12B | 6.3 GB | 25.0 | 3.6 | 4.2 | ~86s/300 tok |

### Speculative decoding

The useful speculative mode is **n-gram prompt lookup**, not MTP:

| Configuration | Result |
|---|---|
| Qwen3.6-28B-REAP + `ngram-simple` on copy-heavy work | decode reached **15.5 t/s** with about 81% draft acceptance |
| Qwen3.6 native MTP on CPU path | non-viable / stalled |
| Gemma 4 assistant MTP split across clusters | technically worked, but did not improve throughput |

The Gemma 4 assistant MTP experiment was useful as an affinity validation: the target/verify workers were pinned to A100 cores `8-15`, while draft/service threads remained on efficiency cores `0-7`. Performance was effectively neutral:

| Gemma 4 E4B run | Thread placement | Prefill t/s | Decode t/s |
|---|---|---:|---:|
| No drafter | target on A100 `8-15` | 26.36 | 5.99 |
| Assistant MTP, 4 draft threads | drafter on X100 `0-7`, target on A100 `8-15` | 26.35 | 5.99 |
| Assistant MTP, 8 draft threads | drafter on X100 `0-7`, target on A100 `8-15` | 26.30 | 5.97 |

## Gemma 4 QAT results

The most useful discovery was that the Gemma 4 QAT GGUFs are much better suited to this board than the older non-QAT Gemma 4 files.

### Gemma 4 E2B QAT

Model:

```text
gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf
```

Results:

```text
llama-bench: pp512 114.44 t/s, tg256 12.69 t/s
server:      prefill 99.6 t/s, decode 12.9 t/s
```

This is the best “useful but still fast” local text/agent model found so far.

### Gemma 4 E4B QAT

Model:

```text
gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf
```

Results:

```text
llama-bench: pp512 68.2 t/s, tg256 7.76 t/s
server:      prefill 55.7 t/s, decode 7.58 t/s
```

This is faster than the non-QAT E4B route and is broadly comparable to Qwen-REAP decode, while using much less disk/RAM.

### Gemma 4 12B QAT

Model:

```text
gemma-4-12B-it-qat-UD-Q4_K_XL.gguf
```

Results:

```text
llama-bench: pp512 27.5 t/s, tg256 3.54 t/s
server:      prefill 25.0 t/s, decode 3.6 t/s
```

It is better than the older dense 12B run, but still too slow for interactive use on the K3. It also leaked Gemma channel markers such as `<|channel>thought` into normal content during chat tests, so it needs template cleanup before it should be exposed as a production route.

## Tool-calling test

`gemma4-e2b-qat` was tested as an OpenAI-compatible tool-calling model through the production router.

Endpoint:

```text
http://<k3-board>:8080/v1/chat/completions
```

A weather prompt with an OpenAI `tools` schema produced a proper tool call:

```json
{
  "name": "get_weather",
  "arguments": "{\"city\":\"Lisbon\"}"
}
```

After feeding a fake tool result back as a `tool` role message, the model produced a normal final response:

```text
The weather in Lisbon is clear with a temperature of 22°C and a wind speed of 12 kph.
```

A calculator prompt produced:

```json
{
  "name": "calculator",
  "arguments": "{\"expression\":\"17 * 23\"}"
}
```

Observed performance:

```text
tool-call generation: ~13.5 t/s
tool-result answer:   ~13.3 t/s
prompt processing:    ~100 t/s
```

That makes E2B QAT a credible small local agent model rather than just a text toy.

## Multimodal test

Gemma 4 E2B is technically multimodal. Both the Unsloth and Google GGUF repos ship about 940 MB `mmproj` files, and this server loads them successfully:

```text
loaded multimodal model, /home/me/models/gguf-misc/gemma-4-E2B-it-mmproj.gguf
```

Tested combinations:

| Text model | Projector | Image path | Result |
|---|---|---:|---|
| Unsloth `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` | Unsloth `mmproj-F16.gguf` | `image_url` data URI | loaded and decoded, but called a pure red image “green” |
| Google `gemma-4-E2B_q4_0-it.gguf` | Google `gemma-4-E2B-it-mmproj.gguf` | `file://` via `--media-path` | loaded and decoded, but called a pure red image “brown” |
| Google pair | same | forced-choice color prompt | correctly answered `red` |
| Google pair | same | simple red-square/blue-circle image | answered “yellow and white” |

Projector processing time was the real problem:

```text
224x224 image: roughly 39-47 seconds before text generation
text decode after image processing: ~12.7 t/s
```

So the multimodal path is wired up, but it is not reliable or fast enough to call practical on this board today.

## E4B QAT optimization / profiling

The E4B QAT route was profiled with `perf` and `llama-bench`.

Best runtime settings remained:

```text
threads = 8
threads-batch = 8
cache-type-k = f16
cache-type-v = f16
batch-size = 1024
ubatch-size = 1024
```

### KV cache settings

| KV K/V | Prefill | Decode | Verdict |
|---|---:|---:|---|
| **f16/f16** | **68.2 t/s** | **7.74 t/s** | best |
| q8_0/q8_0 | 27.3 t/s | 7.16 t/s | worse |
| mixed q8/f16 | failed or no improvement | - | reject |
| q4 variants | failed context creation | - | reject |

Do not quantize KV for this path. It hurts prefill badly and slightly hurts decode.

### Batch / ubatch

At 8 threads, decode was essentially flat:

| Batch / ubatch | Bench prefill | Bench decode | Server decode |
|---|---:|---:|---:|
| 512 / 256 | 71.8 t/s | 7.75 t/s | 7.59 t/s |
| 512 / 512 | 68.4 t/s | 7.77 t/s | - |
| 1024 / 512 | 68.4 t/s | 7.76 t/s | - |
| **1024 / 1024** | 68.3 t/s | 7.75 t/s | **7.60 t/s** |
| 2048 / 1024 | 68.6 t/s | 7.77 t/s | - |
| 2048 / 2048 | 68.5 t/s | 7.77 t/s | - |

The slightly higher bench prefill from `512/256` did not translate into a meaningful server improvement, so the global `1024/1024` preset was left alone.

### OpenMP wait/spin settings

OpenMP sleeping is harmful for per-token decode:

| Setting | Decode |
|---|---:|
| default | ~7.75 t/s |
| `OMP_WAIT_POLICY=ACTIVE` | ~7.75 t/s |
| `OMP_WAIT_POLICY=PASSIVE` | ~5.77 t/s |
| `GOMP_SPINCOUNT=0` | ~5.78 t/s |

The backend wants actively spinning workers. Do not set passive OpenMP wait policies for this server.

### Poll / mmap

No meaningful effect:

```text
--poll 0/25/50/75/100: ~7.73-7.74 t/s
--mmap 0/1: flat
```

### Perf profile

`perf stat` for E4B QAT, 8 threads, f16/f16 KV:

```text
elapsed time:               56.8 s
CPU utilization:             6.53 CPUs
instructions:             248.8 B
cycles:                   659.6 B
IPC:                         0.38
backend stalls:             72.5%
L1-dcache load misses:      44.6%
```

Top sampled symbols:

| Hotspot | Share | Interpretation |
|---|---:|---|
| `memcpy_main_loop34` | 30.1% | custom RVV byte-copy loop inside `libggml-cpu.so` |
| `INNER_BLK_LOOP756` | 13.7% | low-level packed block loop |
| `_KsubBLK_LPST979` | 12.3% | low-level kernel block loop |
| `libgomp` worker/sync symbols | ~14.8% combined | OpenMP scheduling/barrier overhead |
| `tensor_traits<block_q4_0,256,32>::forward_mul_mat` | 6.8% | SpaceMIT repacked Q4 matmul dispatch |
| `ggml_compute_forward_glu` | 3.6% | non-matmul activation work |
| `repack_q4_0_to_q4_0_256_32_bl_ref` | 3.4% | load/setup repacking overhead |
| `ggml_vec_dot_f16` / `tinyBLAS_RVV` | ~4.1% | remaining generic/RVV compute |

`perf annotate` showed `memcpy_main_loop34` is not libc `memcpy`, but a custom RVV copy loop in the backend:

```asm
vle8.v  v0,(a1)
vle8.v  v8,(t2)
vse8.v  v0,(a0)
vse8.v  v8,(t3)
```

The remaining code-level opportunities are therefore low-level backend work:

1. reduce or fuse packed-buffer copies around the SpaceMIT Q4 path;
2. replace per-op OpenMP overhead with persistent workers/barriers;
3. vectorize/fuse the GLU/SwiGLU/tanh path;
4. keep more intermediates in SpaceMIT-compatible buffer types.

There were no obvious application-level or server-level optimizations left after the runtime sweeps.


## Assembly experiment follow-up

After the initial profiling above, the remaining work moved from C++ and server settings into the SpaceMIT/RVV assembly paths. The useful findings are mostly negative, but they narrow the real optimization surface considerably.

### Guarded backend profiling

An experimental branch adds env-gated counters for SpaceMIT backend staging:

```text
branch: exp/backend-profile
commit: 07bb371 spacemit: add experimental backend profile counters
flag:   SPACEMIT_PROFILE=1
```

Short `llama-bench` runs showed just how copy-heavy the current TCM path is:

| Model | TCM-A calls | TCM-B calls | TCM-A bytes | TCM-B bytes |
|---|---:|---:|---:|---:|
| Gemma 4 E4B QAT (`pp128/tg64`) | 510 | 22,538 | 251 MB | 173 GB |
| Qwen REAP (`pp128/tg64`) | 4,514 | 16,323 | 207 MB | 98 GB |

Hot `llama-server` profiles, with models already loaded, confirmed the same thing:

| Model | Main hot spots |
|---|---|
| Gemma 4 E4B QAT | `memcpy_main_loop34` ~38%, `INNER_BLK_LOOP756` ~20%, libgomp sync ~20% combined |
| Qwen REAP | `memcpy_main_loop34` ~25%, `_K_LPST652` ~19%, `gemm_kernel_i8i8_m1` ~11%, plus gated-delta/SSM/f32 dot work |

### Q4_K 32×256 / zero-point path

The source has a fast 32×256 IME2 path for `Q4_0`:

```cpp
static const tensor_traits<block_q4_0, 256, 32> q4_0_32x256_q8_0;
```

but `Q4_K` currently uses only 32×32:

```cpp
static const tensor_traits<block_q4_K, 32, 32> q4_k_32x32_q8_0;
```

I prototyped a `block_q4_K,256,32` trait/repack path. The first version had to fall back to the C++ reference HP zero-point path and was unusably slow:

```text
Gemma 4 E4B Q4_K_M tiny test: pp16 0.65 t/s, tg4 0.34 t/s
```

A second version used the existing `m4` zero-point assembly plus a hybrid `m1_zp` path. The `m1_zp` hybrid was correct against the new self-test harness, but the model path still did not win:

| Model | Result with Q4_K 32×256 prototype |
|---|---:|
| Gemma 4 E4B Q4_K_M | `pp512 29.77 t/s`, `tg128 6.13 t/s` |
| Qwen REAP Q4_K_M | `pp512 28.37 t/s`, `tg128 7.52 t/s` |
| Gemma 4 E4B QAT Q4_0 regression | `pp512 67.68 t/s`, `tg128 7.71 t/s` |

Conclusion: simply forcing Q4_K through the 32×256 HP layout is not enough. The existing 32×32 Q4_K path is already competitive, and improving Q4_K now requires optimizing the existing zero-point HP assembly itself, not just adding a trait.

### IME2 HP kernel self-test harness

A guarded self-test target was added on a separate branch:

```text
branch: exp/ime2-hp-zp-harness
commit: 8ff8877 spacemit: add IME2 HP kernel self-test harness
target: spacemit-ime2-kernel-selftest
```

The harness registers an A100 thread, pins to core 8, and compares the hand-written HP kernels against the local reference implementation:

```text
m1:    max_abs=0 max_rel=0 bad=0/32
m4:    max_abs=0 max_rel=0 bad=0/128
m1_zp: max_abs=0 max_rel=0 bad=0/32   # hybrid prototype only
```

This is the required safety net for future inline-assembly edits to `gemm_kernel_i8i4_hp_m1` and `gemm_kernel_i8i4_hp_m4`.

### TCM copy/staging experiments

The largest hot symbol is a custom RVV copy loop, not libc `memcpy`:

```asm
vle8.v  v0,(a1)
vle8.v  v8,(t2)
vse8.v  v0,(a0)
vse8.v  v8,(t3)
```

Simple changes did not help:

| Experiment | Result |
|---|---|
| Disable B-panel TCM staging (`SPACEMIT_DISABLE_TCM_B=1`) | worse: E4B QAT `7.72→6.77 t/s`, Qwen `7.75→7.27 t/s` |
| Keep quantized A outside TCM, still stage B | worse: E4B QAT `7.74→7.41 t/s`, Qwen `7.78→7.60 t/s` |
| A100/VLEN=1024 copy loop 4096B unroll | worse/flat: E4B QAT dropped to `7.60 t/s`, Qwen flat |
| `prefetch.r` in the current 2048B A100 copy loop | flat in repeated runs: E4B QAT `7.77 t/s`, Qwen `7.78 t/s` |

Conclusion: the copy is expensive, but it is buying enough TCM locality to be worth it. The next useful version would need to fuse or pipeline copy+kernel feed; standalone copy-loop tweaks are exhausted.

### OpenMP and activation experiments

A no-OpenMP build slightly improved `llama-bench` but regressed the real server path:

| Model | OpenMP server | no-OpenMP server |
|---|---:|---:|
| Gemma 4 E4B QAT | `7.53 t/s` | `7.16 t/s` |
| Qwen REAP | `6.99 t/s` | `6.74 t/s` |

So removing OpenMP is not a production win. A persistent worker/barrier design may still be interesting, but it is an architectural rewrite rather than a CMake option.

The F32 SiLU/SwiGLU path already has RVV intrinsics. A prototype F16 RVV SwiGLU widening/narrowing path compiled but did not move model benchmarks:

```text
Gemma 4 E4B QAT tg256: 7.72 t/s
Qwen REAP tg256:       7.81 t/s
```

### Current backend conclusion

The remaining plausible wins are very narrow and assembly-heavy:

1. optimize the existing Q4_K/ZP HP assembly, especially the `m4` path;
2. build a fused/double-buffered TCM copy+kernel-feed path instead of a better standalone copy;
3. design persistent A100 workers/barriers if OpenMP synchronization becomes the main target.

All simple C++/configuration/RVV-intrinsic changes tested so far were neutral or slower, and none were merged into production.

## What kind of AI workload fits the K3?

The K3 is not a “run a big chatbot fast” board. It is a quantized edge-inference board with a surprisingly capable CPU-side matrix engine.

The hardware rewards:

- small dense LLMs, roughly 1B-5B;
- QAT / low-bit GGUF models;
- sparse MoE models with low active parameter counts;
- embedding, reranking, classification and small-agent workloads;
- ONNX graphs dominated by `Conv`, `Gemm`, `MatMul`, activations and normalization;
- batch prefill or other workloads that amortize weight reads.

It struggles with:

- large dense 9B-12B+ autoregressive chat;
- multimodal projector paths;
- MTP/speculative draft models on CPU;
- anything that expects a public standalone NPU SDK.

The practical interpretation is simple: if the model can be repacked into the SpaceMIT low-bit layout and keep its active working set small enough not to drown in LPDDR traffic, the A100/IME2/TCM path pays off. If it has to stream 10 GB of dense weights one token at a time, the marketing TOPS number stops mattering quickly.

## Useful commands

Start the production router:

```bash
/home/me/run-qwen-reap-server.sh
```

Check health:

```bash
curl http://127.0.0.1:8080/health
```

List models:

```bash
curl http://127.0.0.1:8080/v1/models | jq .
```

Run a single E4B QAT benchmark:

```bash
/home/me/src/llama-cpp-turboquant-feature-turboquant-kv-cache/build/bin/llama-bench \
  -m /home/me/models/gguf-misc/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  -p 512 -n 256 -t 8 -b 1024 -ub 1024 \
  -ctk f16 -ctv f16 -r 1 -o md
```

Run a tool-calling request against E2B QAT:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H Content-Type: application/json \
  --data-binary @tool-request.json
```

## Relationship to upstream llama.cpp

This repository is a practical K3/TurboQuant/SpaceMIT server snapshot, not a replacement for upstream [`llama.cpp`](https://github.com/ggml-org/llama.cpp). For general usage, model conversion, upstream APIs and cross-platform documentation, use upstream.

This fork is useful for:

- reproducing the Milk-V Jupiter 2 / SpacemiT K3 experiments;
- inspecting the SpaceMIT/TurboQuant backend wiring;
- running the same local OpenAI-compatible server on a K3 board;
- understanding which small/QAT models actually make sense on this hardware.
