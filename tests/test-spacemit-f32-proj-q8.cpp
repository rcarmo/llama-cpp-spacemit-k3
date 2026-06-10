#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-cpu/spacemit/ime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void fill_deterministic(std::vector<float> & v, uint32_t seed, float scale) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (float & x : v) {
        x = dist(rng);
    }
}

static bool run_case(const char * name, int64_t rows, int64_t tokens, int64_t k, int n_threads) {
    if (k % 32 != 0 || rows % 32 != 0) {
        std::fprintf(stderr, "invalid case %s rows=%lld tokens=%lld k=%lld\n", name, (long long) rows,
                     (long long) tokens, (long long) k);
        return false;
    }

    std::vector<float> weights((size_t) rows * (size_t) k);
    std::vector<float> activations((size_t) tokens * (size_t) k);
    fill_deterministic(weights, 0xf3200000u + (uint32_t) rows + (uint32_t) k, 0.50f);
    fill_deterministic(activations, 0xf3300000u + (uint32_t) tokens + (uint32_t) k, 0.80f);

    std::vector<float> ref((size_t) rows * (size_t) tokens, 0.0f);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t t = 0; t < tokens; ++t) {
            double acc = 0.0;
            for (int64_t kk = 0; kk < k; ++kk) {
                acc += (double) weights[(size_t) r * (size_t) k + (size_t) kk] *
                       (double) activations[(size_t) t * (size_t) k + (size_t) kk];
            }
            ref[(size_t) t * (size_t) rows + (size_t) r] = (float) acc;
        }
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "ggml_backend_cpu_init failed\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    const size_t graph_nodes = 16;
    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(graph_nodes, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
    ggml_set_name(a, name);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, tokens);
    ggml_set_name(b, "activation");
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_set_name(out, "out");

    ggml_backend_buffer_type_t spacemit_buft = ggml_backend_cpu_riscv64_spacemit_buffer_type();
    const size_t               a_alloc_size  = ggml_backend_buft_get_alloc_size(spacemit_buft, a);
    ggml_backend_buffer_t      buf           = ggml_backend_buft_alloc_buffer(spacemit_buft, a_alloc_size);
    if (!buf) {
        std::fprintf(stderr, "ggml_backend_buft_alloc_buffer failed\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    if (ggml_backend_tensor_alloc(buf, a, ggml_backend_buffer_get_base(buf)) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml_backend_tensor_alloc failed for F32 projection weight\n");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<uint8_t> b_storage(ggml_nbytes(b));
    std::vector<uint8_t> out_storage(ggml_nbytes(out));
    b->data   = b_storage.data();
    out->data = out_storage.data();

    ggml_backend_tensor_set(a, weights.data(), 0, weights.size() * sizeof(float));
    std::memcpy(b->data, activations.data(), activations.size() * sizeof(float));
    std::memset(out->data, 0, ggml_nbytes(out));

    const bool supports_op = ggml_backend_supports_op(backend, out);
    if (!supports_op || a->extra == nullptr) {
        std::fprintf(stderr,
                     "case %s rows=%lld tokens=%lld k=%lld did not select SpaceMIT F32->Q8 path: supports=%d a_extra=%p a_alloc=%zu a_plain=%zu\n",
                     name, (long long) rows, (long long) tokens, (long long) k, supports_op ? 1 : 0, a->extra,
                     a_alloc_size, ggml_nbytes(a));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_nodes, false);
    ggml_build_forward_expand(gf, out);

    ggml_status status = ggml_backend_graph_compute(backend, gf);
    bool ok = status == GGML_STATUS_SUCCESS;
    if (!ok) {
        std::fprintf(stderr, "ggml_backend_graph_compute failed: %s\n", ggml_status_to_string(status));
    }

    const float * got = (const float *) out->data;
    double max_abs    = 0.0;
    double max_rel    = 0.0;
    double mse        = 0.0;
    double ref_energy = 0.0;
    double got_energy = 0.0;
    size_t bad = 0;
    const size_t nout = (size_t) rows * (size_t) tokens;
    for (size_t i = 0; i < nout; ++i) {
        const double diff = (double) got[i] - (double) ref[i];
        const double ad = std::fabs(diff);
        const double rel = ad / std::max(1e-6, std::fabs((double) ref[i]));
        max_abs = std::max(max_abs, ad);
        max_rel = std::max(max_rel, rel);
        mse += diff * diff;
        ref_energy += (double) ref[i] * (double) ref[i];
        got_energy += (double) got[i] * (double) got[i];
        if (!std::isfinite(got[i]) || ad > 0.25 + 0.08 * std::fabs((double) ref[i])) {
            if (bad < 12) {
                std::fprintf(stderr, "bad[%zu] got=% .8f ref=% .8f diff=% .8f rel=% .8f\n", i, got[i], ref[i],
                             (float) diff, (float) rel);
            }
            ++bad;
        }
    }
    mse /= std::max<size_t>(1, nout);
    ref_energy /= std::max<size_t>(1, nout);
    got_energy /= std::max<size_t>(1, nout);
    const double ref_rms = std::sqrt(ref_energy);
    const double got_rms = std::sqrt(got_energy);
    const double rmse    = std::sqrt(mse);
    const double nmse    = mse / std::max(1e-12, ref_energy);

    std::printf("case name=%s rows=%lld tokens=%lld k=%lld threads=%d max_abs=%.9g max_rel=%.9g rmse=%.9g ref_rms=%.9g got_rms=%.9g nmse=%.9g bad=%zu/%zu\n",
                name, (long long) rows, (long long) tokens, (long long) k, n_threads, max_abs, max_rel, rmse,
                ref_rms, got_rms, nmse, bad, nout);

    ok = ok && bad == 0 && nmse < 2.0e-3 && got_rms > 0.95 * ref_rms && got_rms < 1.05 * ref_rms;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
}

int main(int argc, char ** argv) {
    ggml_time_init();
    setenv("SPACEMIT_EXPERIMENTAL_F32_PROJ_Q8", "1", 1);

    int n_threads = 8;
    if (argc > 1) {
        n_threads = std::atoi(argv[1]);
        if (n_threads <= 0) {
            n_threads = 1;
        }
    }

    bool ok = true;
    ok = run_case("blk.0.inp_gate.weight", 256, 1, 2560, n_threads) && ok;
    ok = run_case("blk.0.proj.weight", 2560, 1, 256, n_threads) && ok;
    ok = run_case("blk.0.inp_gate.weight", 256, 4, 2560, n_threads) && ok;
    ok = run_case("blk.0.proj.weight", 2560, 4, 256, n_threads) && ok;
    return ok ? 0 : 1;
}
