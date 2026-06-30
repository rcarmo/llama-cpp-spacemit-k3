#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "ime.h"

#include "binary-ops.h"
#include "common.h"
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ime_env.h"
#include "ime_kernels.h"
#include "ops.h"
#include "quants.h"
#include "repack.h"
#include "rvv_kernels.h"
#include "spine_mem_pool.h"
#include "traits.h"
#include "vec.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>  // for GGML_ASSERT
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
// clang-format off
#if defined(__riscv)

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#else
#include <riscv_vector.h>
#endif

#if !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#endif

#if !defined(__riscv_zba)
#error "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#endif

#if defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#else
#error "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#endif

#else

#error "riscv not enabled in this build"

#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

extern "C" {

static bool ggml_riscv64_spacemit_thread_affinity_enabled() {
    const char * v = std::getenv("GGML_RISCV64_SPACEMIT_THREAD_AFFINITY");
    return !(v && (std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0 || std::strcmp(v, "false") == 0));
}

extern void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value);
extern int  ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value);
}

namespace ggml::cpu::riscv64_spacemit {

struct TLSContext {
    int       cpu_id{ -1 };
    cpu_set_t cpuset;
    void *    tcm_buffer{ nullptr };
    size_t    tcm_buffer_size{ 0 };
};

thread_local TLSContext tls_context;


template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> constexpr size_t get_repacked_block_type_size() {
    if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        return sizeof(block_q8_0);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
        return sizeof(block_q4_0) * INTER_SIZE / QK4_0;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K>) {
        return (sizeof(block_q4_0) + sizeof(uint8_t)) * INTER_SIZE / QK4_1;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        return sizeof(spacemit_kernels::nrow_block_q2_k<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        return sizeof(spacemit_kernels::nrow_block_q3_k<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
        return sizeof(spacemit_kernels::nrow_block_mxfp4<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K>) {
        return sizeof(spacemit_kernels::nrow_block_q5_1<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_0>) {
        return sizeof(spacemit_kernels::nrow_block_q5_0<1>);
    } else {
        assert(false);
        return 0;
    }
}

template <typename BLOC_TYPE> constexpr bool block_type_has_zp() {
    if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0> ||
                  std::is_same_v<BLOC_TYPE, block_q3_K> || std::is_same_v<BLOC_TYPE, block_q4_0> ||
                  std::is_same_v<BLOC_TYPE, block_mxfp4> || std::is_same_v<BLOC_TYPE, block_q5_0>) {
        return false;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K> ||
                         std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q5_1> ||
                         std::is_same_v<BLOC_TYPE, block_q5_K>) {
        return true;
    } else {
        assert(false);
        return false;
    }
}

class tensor_traits_base : public ggml::cpu::tensor_traits {
  public:
    virtual int repack(ggml_tensor * t, const void * data, size_t data_size) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> class tensor_traits : public tensor_traits_base {
    bool work_size(int n_threads, const ggml_tensor * op, size_t & size) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } else if constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, true);
                    } else if constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, true, true);
                    } else {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, sizeof(int64_t));

                    return true;
                }
            case GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } else if constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, true);
                    } else if constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, true, true);
                    } else {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, sizeof(int64_t));

                    const int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    const int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    const size_t sizeof_mmid_row_mapping = sizeof(int64_t);
                    size += sizeof_mmid_row_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * sizeof(int64_t);

                    size = GGML_PAD(size, sizeof(int64_t));

                    return true;
                }
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                switch (op->src[0]->type) {
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(params, op);
                        return true;
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_IQ4_NL:
                        if constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                            forward_mul_mat(params, op);
                            return true;
                        }
                        return false;
                    default:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        return false;
                }
                break;
            case GGML_OP_MUL_MAT_ID:
                switch (op->src[0]->type) {
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(params, op);
                        return true;
                    default:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        return false;
                }
                break;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    void forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        int ith = params->ith;
        int nth = params->nth;

        [[maybe_unused]] const enum ggml_type type = src0->type;

        void *        w_data  = (void *) src0->data;
        const float * feature = (const float *) src1->data;
        float *       output  = (float *) dst->data;

        const int64_t gemm_m = ne11 * ne12 * ne13;
        const int64_t gemm_k = ne10;
        const int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_row_def       quantize_a_row_i8;
        spacemit_kernels::quantize_a_row_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        bool                                       set_kernel_impl = false;

        int64_t block_stride_a = spacemit_kernels::q8_blk_size(a_blk_len);

#if defined(RISCV64_SPACEMIT_IME2)
        if (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);

            if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                if constexpr (INTER_SIZE == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    block_stride_a     = spacemit_kernels::q8_hp_blk_size(a_blk_len, true, true);
                    set_kernel_impl    = true;
                } else {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);
                    set_kernel_impl    = true;
                }
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                set_kernel_impl = true;
            }
        }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
        if (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8  = spacemit_kernels::ime1::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = true;
            }
        }
#endif
        if (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        const int64_t a_k_blks = spacemit_kernels::div_round_up(gemm_k, a_blk_len);
        const int64_t b_k_blks = spacemit_kernels::div_round_up(gemm_k, b_blk_len);

        const int64_t row_stride_a        = a_k_blks * block_stride_a;
        const int64_t gemm_workspace_size = GGML_PAD(gemm_m * row_stride_a, alignof(int64_t));

        if (ith == 0 && params->wsize < gemm_workspace_size) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        void *        tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        const int64_t tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        auto * quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t row_align = 4;
        const int64_t     row_blks  = spacemit_kernels::div_round_up(gemm_m, row_align);

        const int64_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        const int64_t per_mb_rows_wsize = row_align * row_stride_a;
        const int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        const int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_count);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        if (gemm_m == 1) {
            int task_per_thread = spacemit_kernels::div_round_up(a_k_blks, nth);
            int a_blk_start     = ith * task_per_thread;
            int a_blk_end       = std::min(a_blk_start + task_per_thread, (int) a_k_blks);
            if (a_blk_start < a_blk_end) {
                quantize_a_row_i8(a_blk_len, feature + a_blk_start * a_blk_len, (a_blk_end - a_blk_start) * a_blk_len,
                                  quant_a_buffer + a_blk_start * block_stride_a);
            }
        } else {
            int task_per_thread = spacemit_kernels::div_round_up(row_blks, nth);
            int m_row_blk_start = ith * task_per_thread;
            int m_row_blk_end   = std::min(m_row_blk_start + task_per_thread, (int) row_blks);
            for (int m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
                int m_idx             = m_row_blk * row_align;
                int rows_tobe_handled = (gemm_m - m_idx) > row_align ? row_align : (gemm_m - m_idx);

                if (rows_tobe_handled == row_align && quantize_a_4row_i8 != nullptr) {
                    const float * a_row_ptr       = feature + m_idx * gemm_k;
                    auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                } else {
                    while (rows_tobe_handled) {
                        const float * a_row_ptr       = feature + m_idx * gemm_k;
                        auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                        quantize_a_row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                        rows_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        const int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        const int64_t gemm_m_blocked    = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
        const int64_t max_gemm_n_stride = spacemit_kernels::div_round_up(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        if (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_round_up(max_gemm_n_stride, NB_COLS) * NB_COLS);
        }

        if (gemm_n_stride == gemm_n && tcm_buffer != nullptr && per_mb_rows_wsize <= tcm_buffer_size) {
            for (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth) {
                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data);
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t m_row_real = std::min(gemm_m - m_start, row_align);

                spacemit_kernels::rvv::memcpy1d(tcm_buffer, quant_a_buffer + m_start * row_stride_a,
                                                m_row_real * row_stride_a);

                int64_t n_blk_real = 0;
                for (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_COLS);

                    uint8_t * a_row_ptr = (uint8_t *) tcm_buffer;
                    float *   c_blk     = output + m_start * gemm_n + ni;

                    int32_t rows_remaining = m_row_real;

                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row_ptr, b_col, b_col_zp, c_blk, rows_remaining,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_ptr += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        } else if (tcm_buffer != nullptr && per_nb_cols_wsize <= tcm_buffer_size) {
            uint8_t * a_row = quant_a_buffer;
            uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
            if ((gemm_workspace_size + per_nb_cols_wsize) <= tcm_buffer_size) {
                a_row = (uint8_t *) tcm_buffer;
                b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + gemm_workspace_size;
            }
            uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

            int64_t ni      = ith * NB_COLS;
            int64_t nb_real = std::min(gemm_n - ni, NB_COLS);

            if (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
                if (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
            }

            spine_barrier_wait(cur_barrier);

            if (ith % 2 != 0 && nb_real > 0) {
                if (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
            }

            for (; ni < gemm_n; ni += NB_COLS * nth) {
                int64_t rows_remaining = gemm_m;
                float * c_blk          = output + ni;
                auto *  a_row_cur      = a_row;

                if (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                while (rows_remaining > 0) {
                    auto rows_handled = gemm_kernel(b_blk_len, a_row_cur, b_col, b_col_zp, c_blk, rows_remaining,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += rows_handled * gemm_n;
                    a_row_cur += rows_handled * row_stride_a;

                    rows_remaining -= rows_handled;
                }

                if (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                const int64_t next_ni = ni + NB_COLS * nth;
                if (next_ni < gemm_n) {
                    nb_real = std::min(gemm_n - next_ni, NB_COLS);
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + next_ni * row_stride_b,
                                                    nb_real * row_stride_b);
                }
            }
        } else {
            const int64_t task_count_m = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
            const int64_t task_count_n = spacemit_kernels::div_round_up(gemm_n, gemm_n_stride);

            int64_t task_count      = task_count_m * task_count_n;
            int64_t task_per_thread = (task_count + nth - 1) / nth;
            int64_t start           = ith * task_per_thread;
            int64_t end             = std::min((ith + 1) * task_per_thread, task_count);
            for (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                const auto tid_n = compute_idx / task_count_m;
                const auto tid_m = compute_idx % task_count_m;

                const int64_t m_start = tid_m * gemm_m_stride;
                const int64_t m_count = std::min(gemm_m - m_start, (int64_t) gemm_m_stride);

                const int64_t n_start = tid_n * gemm_n_stride;
                const int64_t n_count = std::min(gemm_n - n_start, (int64_t) gemm_n_stride);

                const int64_t n_blk = m_count == 1 ? n_count : NB_COLS;

                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data) + n_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t n_blk_real = 0;
                for (int64_t ni = 0; ni < n_count; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(n_count - ni, n_blk);

                    uint8_t * a_row = quant_a_buffer + m_start * row_stride_a;

                    float * c_blk = output + m_start * gemm_n + n_start + ni;

                    int64_t rows_remaining = m_count;

                    uint8_t * b_col_cur    = b_col;
                    uint8_t * b_col_zp_cur = b_col_zp;

                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row, b_col_cur, b_col_zp_cur, c_blk,
                                                        rows_remaining, n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        }
    }

    void forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        int ith = params->ith;
        int nth = params->nth;

        // row groups
        const int n_ids = ids->ne[0];  // n_expert_used
        const int n_as  = ne02;        // n_expert

        struct mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_row_def           quantize_a_row_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        bool                                           set_kernel_impl = false;
        size_t                                         block_stride_a  = spacemit_kernels::q8_blk_size(QK4_0);

#if defined(RISCV64_SPACEMIT_IME2)
        if (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8;
            block_stride_a    = spacemit_kernels::q8_blk_size(QK4_0, true);

            if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                if constexpr (INTER_SIZE == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    block_stride_a    = spacemit_kernels::q8_hp_blk_size(a_blk_len, true, true);
                    set_kernel_impl   = true;
                } else {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);
                    set_kernel_impl    = true;
                }
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl   = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl   = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                set_kernel_impl    = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                set_kernel_impl    = true;
            }
        }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
        if (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8 = spacemit_kernels::ime1::quantize_a_row_i8;

            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = true;
            }
        }
#endif
        if (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        const size_t a_k_blks = spacemit_kernels::div_round_up(ne10, a_blk_len);
        const size_t b_k_blks = spacemit_kernels::div_round_up(ne10, b_blk_len);

        const size_t nbw1                = a_k_blks * block_stride_a;
        const size_t nbw2                = ne11 * nbw1;
        const size_t nbw3                = nbw2 * ne12;
        const size_t gemm_workspace_size = GGML_PAD(nbw3, alignof(int64_t));

        const uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        auto *          quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        if (ne11 == 1) {
            for (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_row_i8(a_blk_len, (float *) ((char *) src1->data + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_buffer + i12 * nbw2 + ak_blk_id * block_stride_a);
            }
        } else {
            for (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_row_i8(a_blk_len, (float *) ((char *) src1->data + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_buffer + i12 * nbw2 + i11 * nbw1);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) *ne12 + (i1)]

        int64_t *          matrix_row_counts       = (int64_t *) (ws_ptr + gemm_workspace_size);
        int32_t *          valid_ep_count          = (int32_t *) (matrix_row_counts + n_as);
        int32_t *          valid_act_count         = (int32_t *) (valid_ep_count + 1);
        int64_t *          valid_matrix_row_counts = (int64_t *) (valid_act_count + 1);
        mmid_row_mapping * matrix_rows             = (mmid_row_mapping *) (valid_matrix_row_counts + n_as);

        if (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * sizeof(int64_t));

            // group rows by src0 matrix
            for (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                for (int32_t id = 0; id < n_ids; ++id) {
                    const int32_t i02 =
                        *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }

            int32_t valid_ep_count_t  = 0;
            int32_t valid_act_count_t = 0;
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                valid_matrix_row_counts[valid_ep_count_t] = cur_a;
                valid_act_count_t += cne1;
                valid_ep_count_t += 1;
            }
            valid_ep_count[0]  = valid_ep_count_t;
            valid_act_count[0] = valid_act_count_t;
        }

        const int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_count);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        const size_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        const size_t expert_b_stride   = ne01 * row_stride_b;
        const size_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        std::array<const uint8_t *, 2> src_workspaces;
        std::array<float *, 2>         dst_workspaces;

        auto *     tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        const auto tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        const auto valid_ep_count_t  = valid_ep_count[0];
        const auto valid_act_count_t = valid_act_count[0];

        int nth_es = 1;
        int nth_n  = nth;

        int ith_es = ith % nth_es;
        int ith_n  = (ith / nth_es) % nth_n;

        if (valid_ep_count_t % nth == 0 && tcm_buffer != nullptr && valid_ep_count_t == n_as &&
            valid_act_count_t == n_as && per_nb_cols_wsize <= tcm_buffer_size) {
            for (int64_t valid_id = ith; valid_id < valid_ep_count_t; valid_id += nth) {
                const int64_t cur_a = valid_matrix_row_counts[valid_id];

                auto * src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride;

                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, 0);
                const int        id          = row_mapping.i1;
                const int64_t    i11         = id % ne11;
                const int64_t    i12         = row_mapping.i2;
                const int64_t    i1          = id;
                const int64_t    i2          = i12;

                auto *  src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                float * c_blk    = (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2));

                uint8_t * a_row = src1_col;
                uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
                if ((nbw1 + per_nb_cols_wsize) <= tcm_buffer_size) {
                    a_row = (uint8_t *) tcm_buffer;
                    b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + nbw1;
                }
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                if (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);

                    if (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                if (ith % 2 != 0) {
                    if (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_COLS);
                for (int64_t ni = 0; ni < ne01; ni += NB_COLS) {
                    if (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_row, b_col, b_col_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    if (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    const int64_t next_ni = ni + NB_COLS;
                    if (next_ni < ne01) {
                        nb_real = std::min(ne01 - next_ni, NB_COLS);
                        spacemit_kernels::rvv::memcpy1d(
                            b_col, reinterpret_cast<uint8_t *>(src0_cur) + next_ni * row_stride_b, per_nb_cols_wsize);
                    }
                }
            }
        } else {
            for (int64_t valid_id = ith_es; valid_id < valid_ep_count_t; valid_id += nth_es) {
                const int64_t cur_a = valid_matrix_row_counts[valid_id];
                const int64_t cne1  = matrix_row_counts[cur_a];

                int64_t src1_cur_start = 0;
                int64_t src1_cur_end   = cne1;

                int64_t src0_cur_start = (ith_n * ne01) / nth_n;
                int64_t src0_cur_end   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                if (src1_cur_start >= src1_cur_end || src0_cur_start >= src0_cur_end) {
                    continue;
                }

                src0_cur_start =
                    (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                src0_cur_end =
                    (src0_cur_end % NB_COLS) ? src0_cur_end + NB_COLS - (src0_cur_end % NB_COLS) : src0_cur_end;

                auto *    src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride + src0_cur_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;

                size_t extra_tcm_buffer_size = tcm_buffer_size;
                void * extra_tcm_buffer      = tcm_buffer;
                if (tcm_buffer != nullptr && (src1_cur_end - src1_cur_start) >= 4 &&
                    (src0_cur_end - src0_cur_start) * row_stride_b <= tcm_buffer_size) {
                    spacemit_kernels::rvv::memcpy1d(tcm_buffer, src0_cur,
                                                    (src0_cur_end - src0_cur_start) * row_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_buffer);
                    b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;
                    extra_tcm_buffer_size -= (src0_cur_end - src0_cur_start) * row_stride_b;
                    extra_tcm_buffer = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(tcm_buffer) +
                                                                (src0_cur_end - src0_cur_start) * row_stride_b);
                }

                int ir1 = src1_cur_start;

                if (extra_tcm_buffer_size >= nbw1 && extra_tcm_buffer != nullptr) {
                    int64_t quant_a_tile_size = extra_tcm_buffer_size / nbw1;
                    do {
                        quant_a_tile_size = MIN(quant_a_tile_size, src1_cur_end - ir1);

                        uint8_t * quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);

                        int iir1 = ir1;
                        for (; iir1 < (ir1 + quant_a_tile_size); ++iir1) {
                            mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1);

                            const int id = row_mapping.i1;  // selected expert index

                            const int64_t i11 = id % ne11;
                            const int64_t i12 = row_mapping.i2;  // row index in src1

                            auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_buffer, src1_col, nbw1);
                            quant_a_tile_buffer = quant_a_tile_buffer + nbw1;
                        }

                        quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);
                        iir1                = ir1;

                        if (moe_gemm_kernel_m2 != nullptr) {
                            for (; iir1 < (ir1 + quant_a_tile_size - 1); iir1 += 2, quant_a_tile_buffer += 2 * nbw1) {
                                mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);
                                mmid_row_mapping row_mapping_1 = MMID_MATRIX_ROW(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_buffer;
                                src_workspaces[1] = quant_a_tile_buffer + nbw1;

                                dst_workspaces[0] =
                                    (float *) ((char *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start;
                                dst_workspaces[1] = (float *) ((char *) dst->data +
                                                               ((row_mapping_1.i1) * nb1 + (row_mapping_1.i2) * nb2)) +
                                                    src0_cur_start;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                                   dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks,
                                                   ne01);
                            }
                        }

                        for (; iir1 < (ir1 + quant_a_tile_size); iir1++, quant_a_tile_buffer += nbw1) {
                            mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_buffer, src0_cur, b_col_zp,
                                (float *) ((char *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start,
                                1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_size;
                    } while (ir1 < src1_cur_end);
                } else {
                    if (moe_gemm_kernel_m2 != nullptr) {
                        for (; ir1 < src1_cur_end - 1; ir1 += 2) {
                            for (int iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1 + iir1);

                                const int id = row_mapping.i1;  // selected expert index

                                const int64_t i11 = id % ne11;
                                const int64_t i12 = row_mapping.i2;  // row index in src1

                                const int64_t i1 = id;               // selected expert index
                                const int64_t i2 = i12;              // row

                                src_workspaces[iir1] = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                               dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }
                    }

                    for (; ir1 < src1_cur_end; ir1++) {
                        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        const int id = row_mapping.i1;  // selected expert index

                        const int64_t i11 = id % ne11;
                        const int64_t i12 = row_mapping.i2;  // row index in src1

                        const int64_t i1 = id;               // selected expert index
                        const int64_t i2 = i12;              // row

                        auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_col, src0_cur, b_col_zp,
                                    (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, 1,
                                    src0_cur_end - src0_cur_start, b_k_blks, ne01);
                    }
                }
            }
        }
#undef MMID_MATRIX_ROW
    }

    int repack(ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (int) NB_COLS, (int) INTER_SIZE);
        return ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};


class tensor_traits_iq_compact : public tensor_traits_base {
    ggml_type type_;

    struct mmid_row_mapping {
        int32_t i1;
        int32_t i2;
    };

    static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {
        void * ptr = *p;
        ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
        *p  = (void *) ((char *) ptr + size);
        return ptr;
    }

  public:
    explicit constexpr tensor_traits_iq_compact(ggml_type type) : type_(type) {}

    bool work_size(int n_threads, const ggml_tensor * op, size_t & size) override {
        if (op->op != GGML_OP_MUL_MAT_ID || op->src[0]->type != type_) {
            return false;
        }
        const char * compact_compute = std::getenv("GGML_RISCV64_SPACEMIT_IQ_COMPACT_COMPUTE");
        if (compact_compute && (std::strcmp(compact_compute, "0") == 0 || std::strcmp(compact_compute, "off") == 0 ||
                                std::strcmp(compact_compute, "false") == 0)) {
            return false;
        }
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const ggml_tensor * ids  = op->src[2];
        size = 0;
        // Quantized src1 rows for direct compact-IQ x Q8_K dots, plus one
        // fallback dequantized src0 row per compute thread.
        size += ggml_row_size(GGML_TYPE_Q8_K, src0->ne[0]) * (size_t) ggml_nelements(src1) / (size_t) src0->ne[0];
        size = GGML_PAD(size, sizeof(int64_t));
        size += (size_t) n_threads * (size_t) src0->ne[0] * sizeof(float);
        size = GGML_PAD(size, sizeof(int64_t));
        const int64_t n_as     = src0->ne[2];
        const int64_t max_rows = ids->ne[0] * ids->ne[1];
        size += n_as * sizeof(int64_t);
        size = GGML_PAD(size, sizeof(int64_t));
        size += n_as * max_rows * sizeof(mmid_row_mapping);
        size = GGML_PAD(size, CACHE_LINE_SIZE);
        // Extra space for the generic CPU MUL_MAT_ID fallback diagnostic path.
        size += CACHE_LINE_SIZE * n_as;
        size = GGML_PAD(size, sizeof(int64_t));
        return true;
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT_ID || op->src[0]->type != type_) {
            return false;
        }
        const char * compact_compute = std::getenv("GGML_RISCV64_SPACEMIT_IQ_COMPACT_COMPUTE");
        if (compact_compute && (std::strcmp(compact_compute, "0") == 0 || std::strcmp(compact_compute, "off") == 0 ||
                                std::strcmp(compact_compute, "false") == 0)) {
            return false;
        }
        forward_mul_mat_id(params, op);
        return true;
    }

    static float dot_f32(const float * a, const float * b, int64_t n) {
#if defined(__riscv_v_intrinsic)
        float sum = 0.0f;
        for (int64_t i = 0; i < n;) {
            const size_t vl = __riscv_vsetvl_e32m8((size_t) (n - i));
            vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
            vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
            vfloat32m8_t vp = __riscv_vfmul_vv_f32m8(va, vb, vl);
            vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, vl);
            vfloat32m1_t red  = __riscv_vfredusum_vs_f32m8_f32m1(vp, zero, vl);
            sum += __riscv_vfmv_f_s_f32m1_f32(red);
            i += (int64_t) vl;
        }
        return sum;
#else
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
#endif
    }

    static float dot_i8_f32_scaled(const int8_t * q, const float * y, int n, float scale) {
#if defined(__riscv_v_intrinsic)
        float sum = 0.0f;
        for (int i = 0; i < n;) {
            const size_t vl = __riscv_vsetvl_e8m1((size_t) (n - i));
            vint8m1_t  vq8  = __riscv_vle8_v_i8m1(q + i, vl);
            vint16m2_t vq16 = __riscv_vwadd_vx_i16m2(vq8, 0, vl);
            vint32m4_t vq32 = __riscv_vwadd_vx_i32m4(vq16, 0, vl);
            vfloat32m4_t vf = __riscv_vfcvt_f_x_v_f32m4(vq32, vl);
            vfloat32m4_t vy = __riscv_vle32_v_f32m4(y + i, vl);
            vfloat32m4_t vp = __riscv_vfmul_vf_f32m4(__riscv_vfmul_vv_f32m4(vf, vy, vl), scale, vl);
            vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, vl);
            vfloat32m1_t red  = __riscv_vfredusum_vs_f32m4_f32m1(vp, zero, vl);
            sum += __riscv_vfmv_f_s_f32m1_f32(red);
            i += (int) vl;
        }
        return sum;
#else
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += scale * (float) q[i] * y[i];
        }
        return sum;
#endif
    }

    static int32_t dot_i8_i8(const int8_t * a, const int8_t * b, int n) {
#if defined(__riscv_v_intrinsic)
        int32_t sum = 0;
        for (int i = 0; i < n;) {
            const size_t vl = __riscv_vsetvl_e8m1((size_t) (n - i));
            vint8m1_t va = __riscv_vle8_v_i8m1(a + i, vl);
            vint8m1_t vb = __riscv_vle8_v_i8m1(b + i, vl);
            vint16m2_t prod = __riscv_vwmul_vv_i16m2(va, vb, vl);
            vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
            vint32m1_t red  = __riscv_vwredsum_vs_i16m2_i32m1(prod, zero, vl);
            sum += __riscv_vmv_x_s_i32m1_i32(red);
            i += (int) vl;
        }
        return sum;
#else
        int32_t sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += (int32_t) a[i] * (int32_t) b[i];
        }
        return sum;
#endif
    }

    static float dot_iq2_xs_q8k(const block_iq2_xs * x, const block_q8_K * y, int64_t k) {
        GGML_ASSERT(k % QK_K == 0);
        const int64_t nb = k / QK_K;
        float sum = 0.0f;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
            for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                const float db0 = d * (0.5f + (x[i].scales[ib32] & 0x0f)) * 0.25f;
                const float db1 = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
                int8_t qv[32];
                for (int l = 0; l < 4; ++l) {
                    const uint16_t q = x[i].qs[4*ib32 + l];
                    const uint8_t * grid = (const uint8_t *) (iq2xs_grid + (q & 511));
                    const uint8_t signs = ksigns_iq2xs[q >> 9];
                    for (int j = 0; j < 8; ++j) {
                        qv[l*8 + j] = (int8_t) ((signs & kmask_iq2xs[j]) ? -(int) grid[j] : (int) grid[j]);
                    }
                }
                sum += db0 * (float) dot_i8_i8(qv +  0, y[i].qs + ib32*32 +  0, 16);
                sum += db1 * (float) dot_i8_i8(qv + 16, y[i].qs + ib32*32 + 16, 16);
            }
        }
        return sum;
    }

    static float dot_iq3_xxs_q8k(const block_iq3_xxs * x, const block_q8_K * y, int64_t k) {
        GGML_ASSERT(k % QK_K == 0);
        const int64_t nb = k / QK_K;
        float sum = 0.0f;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
            const uint8_t * qs = x[i].qs;
            const uint8_t * scales_and_signs = qs + QK_K/4;
            for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                uint32_t aux32;
                memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
                const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
                int8_t qv[32];
                for (int l = 0; l < 4; ++l) {
                    const uint8_t signs = ksigns_iq2xs[(aux32 >> (7*l)) & 127];
                    const uint8_t * grid1 = (const uint8_t *) (iq3xxs_grid + qs[2*l+0]);
                    const uint8_t * grid2 = (const uint8_t *) (iq3xxs_grid + qs[2*l+1]);
                    for (int j = 0; j < 4; ++j) {
                        qv[l*8 + j + 0] = (int8_t) ((signs & kmask_iq2xs[j+0]) ? -(int) grid1[j] : (int) grid1[j]);
                        qv[l*8 + j + 4] = (int8_t) ((signs & kmask_iq2xs[j+4]) ? -(int) grid2[j] : (int) grid2[j]);
                    }
                }
                sum += db * (float) dot_i8_i8(qv, y[i].qs + ib32*32, 32);
                qs += 8;
            }
        }
        return sum;
    }

    static float dot_iq4_xs_q8k(const block_iq4_xs * x, const block_q8_K * y, int64_t k) {
        GGML_ASSERT(k % QK_K == 0);
        const int64_t nb = k / QK_K;
        float sum = 0.0f;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
            const uint8_t * qs = x[i].qs;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                const int ls = ((x[i].scales_l[ib/2] >> (4*(ib%2))) & 0x0f) | (((x[i].scales_h >> (2*ib)) & 3) << 4);
                const float dl = d * (ls - 32);
                int8_t qv[32];
                for (int j = 0; j < 16; ++j) {
                    qv[j + 0]  = kvalues_iq4nl[qs[j] & 0x0f];
                    qv[j + 16] = kvalues_iq4nl[qs[j] >> 4];
                }
                sum += dl * (float) dot_i8_i8(qv, y[i].qs + ib*32, 32);
                qs += 16;
            }
        }
        return sum;
    }

    static bool dot_iq_direct_q8k(ggml_type type, const void * x, const block_q8_K * y, int64_t k, float * out) {
        switch (type) {
            case GGML_TYPE_IQ2_XS:
                *out = dot_iq2_xs_q8k((const block_iq2_xs *) x, y, k);
                return true;
            case GGML_TYPE_IQ3_XXS:
                *out = dot_iq3_xxs_q8k((const block_iq3_xxs *) x, y, k);
                return true;
            case GGML_TYPE_IQ4_XS:
                *out = dot_iq4_xs_q8k((const block_iq4_xs *) x, y, k);
                return true;
            default:
                return false;
        }
    }

    static float dot_iq2_xs_f32(const block_iq2_xs * x, const float * y, int64_t k) {
        GGML_ASSERT(k % QK_K == 0);
        const int64_t nb = k / QK_K;
        float sum = 0.0f;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = GGML_FP16_TO_FP32(x[i].d);
            for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                const float db0 = d * (0.5f + (x[i].scales[ib32] & 0x0f)) * 0.25f;
                const float db1 = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
                const float * yy = y + i*QK_K + ib32*32;
                int8_t qv[32];
                for (int l = 0; l < 4; ++l) {
                    const uint16_t q = x[i].qs[4*ib32 + l];
                    const uint8_t * grid = (const uint8_t *) (iq2xs_grid + (q & 511));
                    const uint8_t signs = ksigns_iq2xs[q >> 9];
                    for (int j = 0; j < 8; ++j) {
                        qv[l*8 + j] = (int8_t) ((signs & kmask_iq2xs[j]) ? -(int) grid[j] : (int) grid[j]);
                    }
                }
                sum += dot_i8_f32_scaled(qv +  0, yy +  0, 16, db0);
                sum += dot_i8_f32_scaled(qv + 16, yy + 16, 16, db1);
            }
        }
        return sum;
    }

    static float dot_iq3_xxs_f32(const block_iq3_xxs * x, const float * y, int64_t k) {
        GGML_ASSERT(k % QK_K == 0);
        const int64_t nb = k / QK_K;
        float sum = 0.0f;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = GGML_FP16_TO_FP32(x[i].d);
            const uint8_t * qs = x[i].qs;
            const uint8_t * scales_and_signs = qs + QK_K/4;
            const float * yy = y + i*QK_K;
            for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                uint32_t aux32;
                memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
                const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
                const float * yy32 = yy + ib32*32;
                int8_t qv[32];
                for (int l = 0; l < 4; ++l) {
                    const uint8_t signs = ksigns_iq2xs[(aux32 >> (7*l)) & 127];
                    const uint8_t * grid1 = (const uint8_t *) (iq3xxs_grid + qs[2*l+0]);
                    const uint8_t * grid2 = (const uint8_t *) (iq3xxs_grid + qs[2*l+1]);
                    for (int j = 0; j < 4; ++j) {
                        qv[l*8 + j + 0] = (int8_t) ((signs & kmask_iq2xs[j+0]) ? -(int) grid1[j] : (int) grid1[j]);
                        qv[l*8 + j + 4] = (int8_t) ((signs & kmask_iq2xs[j+4]) ? -(int) grid2[j] : (int) grid2[j]);
                    }
                }
                sum += dot_i8_f32_scaled(qv, yy32, 32, db);
                qs += 8;
            }
        }
        return sum;
    }

    static float dot_iq4_xs_f32(const block_iq4_xs * x, const float * y, int64_t k) {
        GGML_ASSERT(k % QK_K == 0);
        const int64_t nb = k / QK_K;
        float sum = 0.0f;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = GGML_FP16_TO_FP32(x[i].d);
            const uint8_t * qs = x[i].qs;
            const float * yy = y + i*QK_K;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                const int ls = ((x[i].scales_l[ib/2] >> (4*(ib%2))) & 0x0f) | (((x[i].scales_h >> (2*ib)) & 3) << 4);
                const float dl = d * (ls - 32);
                const float * yy32 = yy + ib*32;
                int8_t qv[32];
                for (int j = 0; j < 16; ++j) {
                    qv[j + 0]  = kvalues_iq4nl[qs[j] & 0x0f];
                    qv[j + 16] = kvalues_iq4nl[qs[j] >> 4];
                }
                sum += dot_i8_f32_scaled(qv, yy32, 32, dl);
                qs += 16;
            }
        }
        return sum;
    }

    static bool dot_iq_direct_f32(ggml_type type, const void * x, const float * y, int64_t k, float * out) {
        switch (type) {
            case GGML_TYPE_IQ2_XS:
                *out = dot_iq2_xs_f32((const block_iq2_xs *) x, y, k);
                return true;
            case GGML_TYPE_IQ3_XXS:
                *out = dot_iq3_xxs_f32((const block_iq3_xxs *) x, y, k);
                return true;
            case GGML_TYPE_IQ4_XS:
                *out = dot_iq4_xs_f32((const block_iq4_xs *) x, y, k);
                return true;
            default:
                return false;
        }
    }

    void forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) const {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;
        GGML_TENSOR_BINARY_OP_LOCALS

        const int ith = params->ith;
        const int nth = params->nth;

        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(nb00 == ggml_type_size(src0->type));
        GGML_ASSERT(nb10 == ggml_type_size(src1->type));
        GGML_ASSERT(nb0 == sizeof(float));

        const ggml_type_traits * type_traits = ggml_get_type_traits(src0->type);
        const ggml_to_float_t to_float = type_traits->to_float;
        GGML_ASSERT(to_float != nullptr);

        void * wdata_cur = params->wdata;
        const size_t q8_row_size = ggml_row_size(GGML_TYPE_Q8_K, ne10);
        char * qsrc1 = (char *) incr_ptr_aligned(&wdata_cur, q8_row_size * (size_t) ggml_nelements(src1) / (size_t) ne10, sizeof(int64_t));
        float * deq_rows = (float *) incr_ptr_aligned(&wdata_cur, (size_t) nth * (size_t) ne00 * sizeof(float), sizeof(int64_t));
        float * deq_row  = deq_rows + (size_t) ith * (size_t) ne00;

        const int64_t n_ids    = ids->ne[0];
        const int64_t n_as     = ne02;
        const int64_t max_rows = ids->ne[0] * ids->ne[1];
        int64_t * matrix_row_counts = (int64_t *) incr_ptr_aligned(&wdata_cur, n_as * sizeof(int64_t), sizeof(int64_t));
        mmid_row_mapping * matrix_rows = (mmid_row_mapping *) incr_ptr_aligned(&wdata_cur, n_as * max_rows * sizeof(mmid_row_mapping), sizeof(int64_t));
        GGML_ASSERT(params->wsize >= (size_t) ((char *) wdata_cur - (char *) params->wdata));

        const int64_t src1_rows = ne11 * ne12 * ne13;
        for (int64_t row = ith; row < src1_rows; row += nth) {
            const int64_t i11q = row % ne11;
            const int64_t i12q = (row / ne11) % ne12;
            const int64_t i13q = row / (ne11 * ne12);
            const float * src1_row = (const float *) ((const char *) src1->data + i13q * nb13 + i12q * nb12 + i11q * nb11);
            quantize_row_q8_K_generic(src1_row, qsrc1 + (size_t) row * q8_row_size, ne10);
        }
        ggml_barrier(params->threadpool);

        if (ith == 0) {
            memset(matrix_row_counts, 0, n_as * sizeof(int64_t));
            for (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                for (int32_t id = 0; id < n_ids; ++id) {
                    const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);
                    GGML_ASSERT(i02 >= 0 && i02 < n_as);
                    matrix_rows[i02 * max_rows + matrix_row_counts[i02]] = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }
        }
        ggml_barrier(params->threadpool);

        const bool src1_cont = ggml_is_contiguous(src1);
        for (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
            const int64_t cne1 = matrix_row_counts[cur_a];
            if (cne1 == 0) {
                continue;
            }
            const char * src0_cur = (const char *) src0->data + cur_a * nb02;
            const int64_t total   = cne1 * ne01;
            for (int64_t task = ith; task < total; task += nth) {
                const int64_t ir1 = task / ne01;
                const int64_t ir0 = task % ne01;
                const mmid_row_mapping row_mapping = matrix_rows[cur_a * max_rows + ir1];
                const int64_t i11 = row_mapping.i1 % ne11;
                const int64_t i12 = row_mapping.i2;
                const int64_t i1  = row_mapping.i1;
                const int64_t i2  = i12;

                const char * src1_col_bytes = (const char *) src1->data + (src1_cont ? (i11 + i12 * ne11) * ne10 * nb10 : (i11 * nb11 + i12 * nb12));
                const float * src1_col = (const float *) src1_col_bytes;
                const char * src0_row = src0_cur + ir0 * nb01;

                const int64_t src1_row_idx = i11 + i12 * ne11;
                const block_q8_K * src1_col_q8 = (const block_q8_K *) (qsrc1 + (size_t) src1_row_idx * q8_row_size);
                float v;
                if (!dot_iq_direct_q8k(src0->type, src0_row, src1_col_q8, ne00, &v)) {
                    if (!dot_iq_direct_f32(src0->type, src0_row, src1_col, ne00, &v)) {
                        to_float(src0_row, deq_row, ne00);
                        v = dot_f32(deq_row, src1_col, ne00);
                    }
                }
                float * dst_col = (float *) ((char *) dst->data + i1 * nb1 + i2 * nb2);
                dst_col[ir0] = v;
            }
        }
    }

    int repack(ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: copy compact tensor %s with %s\n", __func__, t->name, ggml_type_name(t->type));
        GGML_ASSERT(t->type == type_);
        GGML_ASSERT(data_size == ggml_nbytes(t));
        memcpy(t->data, data, data_size);
        return 0;
    }
};

class tensor_traits_bf16_proj_q8_0 : public tensor_traits<block_q8_0, 32, 32> {
    int repack(ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with bf16_q8_0_32x32\n", __func__, t->name);
        return ggml::cpu::riscv64_spacemit::repack_bf16_to_q8_0_32x32(t, data, data_size);
    }
};

class tensor_traits_f32_proj_q8_0 : public tensor_traits<block_q8_0, 32, 32> {
    int repack(ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with f32_q8_0_32x32\n", __func__, t->name);
        return ggml::cpu::riscv64_spacemit::repack_f32_to_q8_0_32x32(t, data, data_size);
    }
};

class tensor_traits_iq4_nl_proj_q8_0 : public tensor_traits<block_q8_0, 32, 32> {
    int repack(ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with iq4_nl_q8_0_32x32\n", __func__, t->name);
        return ggml::cpu::riscv64_spacemit::repack_iq4_nl_to_q8_0_32x32(t, data, data_size);
    }
};
class tensor_traits_iq2_xs_proj_q8_0 : public tensor_traits<block_q8_0, 32, 32> {
    int repack(ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with iq2_xs_q8_0_32x32\n", __func__, t->name);
        return ggml::cpu::riscv64_spacemit::repack_iq2_xs_to_q8_0_32x32(t, data, data_size);
    }
};

class tensor_traits_iq3_xxs_proj_q8_0 : public tensor_traits<block_q8_0, 32, 32> {
    int repack(ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with iq3_xxs_q8_0_32x32\n", __func__, t->name);
        return ggml::cpu::riscv64_spacemit::repack_iq3_xxs_to_q8_0_32x32(t, data, data_size);
    }
};

class tensor_traits_iq4_xs_proj_q8_0 : public tensor_traits<block_q8_0, 32, 32> {
    int repack(ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with iq4_xs_q8_0_32x32\n", __func__, t->name);
        return ggml::cpu::riscv64_spacemit::repack_iq4_xs_to_q8_0_32x32(t, data, data_size);
    }
};


class tensor_traits_passthrough : public tensor_traits_base {
  public:
    bool work_size(int, const ggml_tensor *, size_t &) override { return false; }
    bool compute_forward(ggml_compute_params *, ggml_tensor *) override { return false; }
    int repack(ggml_tensor *, const void *, size_t) override { return -1; }
};


class tensor_traits_common : public tensor_traits_base {
    bool work_size(int n_threads, const ggml_tensor * op, size_t & size) override {
        switch (op->op) {
            case GGML_OP_FLASH_ATTN_EXT:
                {
                    const int     n_tasks = n_threads;
                    const int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    const int64_t DK      = op->src[1]->ne[0];
                    const int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = sizeof(float) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = sizeof(float) * (neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV));

                    size = MAX(prefill, decode);
                }
                return true;
            default:
                break;
        }
        return false;
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_NORM:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(params, op);
                        return true;
                    default:
                        GGML_ABORT("fatal error");
                }
            case GGML_OP_RMS_NORM:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        return true;
                    default:
                        GGML_ABORT("fatal error");
                }
            case GGML_OP_ADD:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_ADD, float>(params, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_ADD, _Float16>(params, op);
                        return true;
                    default:
                        ggml_compute_forward_add(params, op);
                        return true;
                }
            case GGML_OP_SUB:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_SUB, float>(params, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_SUB, _Float16>(params, op);
                        return true;
                    default:
                        ggml_compute_forward_sub(params, op);
                        return true;
                }
            case GGML_OP_MUL:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_MUL, float>(params, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_MUL, _Float16>(params, op);
                        return true;
                    default:
                        ggml_compute_forward_mul(params, op);
                        return true;
                }
            case GGML_OP_DIV:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_DIV, float>(params, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_DIV, _Float16>(params, op);
                        return true;
                    default:
                        ggml_compute_forward_div(params, op);
                        return true;
                }
            case GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                return true;
            case GGML_OP_CONT:
                {
                    const ggml_tensor * src0 = op->src[0];
                    if (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_permute(params, op);
                    } else {
                        ggml_compute_forward_cont(params, op);
                    }
                    return true;
                }
            case GGML_OP_CPY:
                {
                    const ggml_tensor * src0 = op->src[0];
                    if (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_permute(params, op);
                    } else {
                        ggml_compute_forward_cpy(params, op);
                    }
                    return true;
                }
            case GGML_OP_REPEAT:
                {
                    const bool rows_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    const bool broadcast_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    if (rows_equal && broadcast_or_equal) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    if (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                return true;
            case GGML_OP_SUM_ROWS:
                {
                    if (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_sum_rows<float>(params, op);
                    } else {
                        ggml_compute_forward_sum_rows(params, op);
                    }
                }
                return true;
            case GGML_OP_GET_ROWS:
                {
                    if (op->src[0]->type == op->type) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_rows<int32_t>(params, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_rows<int16_t>(params, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    ggml_compute_forward_get_rows(params, op);
                }
                return true;
            case GGML_OP_CONCAT:
                {
                    const int32_t dim = ggml_get_op_params_i32(op, 0);
                    if (dim == 0 && op->type == op->src[0]->type) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                return true;
            // TODO For GGML_OP_GATED_DELTA_NET
            // case GGML_OP_GATED_DELTA_NET:
            //     return true;
            default:
                break;
        }
        return false;
    }

    void forward_flash_attn_ext_f16(const ggml_compute_params * params, ggml_tensor * dst) {
        const ggml_tensor * q = dst->src[0];
        const ggml_tensor * k = dst->src[1];
        const ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        const int64_t DK = nek0;
        const int64_t DV = nev0;

        const bool supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        const bool supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        const bool supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        const bool supported_vlen  = (__riscv_vlenb() == 128);

        if (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            ggml_compute_forward_flash_attn_ext(params, dst);
            return;
        }

        // total rows in q
        const int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        const int ith = params->ith;
        const int nth = params->nth;

        static constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        const bool               use_tiled = !params->use_ref && (neq1 >= Q_TILE_SZ);

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        int64_t nchunk = nth;

        if (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The number of elements in each chunk
        const int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        int current_chunk = ith;

        while (current_chunk < nchunk) {
            const int64_t ir0 = dr * current_chunk;
            const int64_t ir1 = MIN(ir0 + dr, nr);

            if (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            } else {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    int repack(ggml_tensor * t, const void * data, size_t data_size) override {
        memcpy(t->data, data, data_size);
        return 0;
    }
};

// Impl By IME1
static const tensor_traits<block_q4_0, 32, 16>  q4_0_16x32_q8_0;
static const tensor_traits<block_q4_1, 32, 16>  q4_1_16x32_q8_0;
static const tensor_traits<block_q4_K, 32, 16>  q4_k_16x32_q8_0;
// Impl By IME2
static const tensor_traits<block_q2_K, 256, 32> q2_k_32x256_q8_0;
static const tensor_traits<block_q3_K, 256, 32> q3_k_32x256_q8_0;
static const tensor_traits<block_q4_0, 32, 32>  q4_0_32x32_q8_0;
static const tensor_traits<block_q4_1, 32, 32>  q4_1_32x32_q8_0;
static const tensor_traits<block_q4_0, 256, 32> q4_0_32x256_q8_0;
static const tensor_traits<block_q4_1, 256, 32> q4_1_32x256_q8_0;
static const tensor_traits<block_q4_K, 32, 32>  q4_k_32x32_q8_0;
static const tensor_traits<block_q4_K, 256, 32> q4_k_32x256_q8_0;
static const tensor_traits<block_q6_K, 32, 32>  q6_k_32x32_q8_0;
static const tensor_traits<block_q8_0, 32, 32>  q8_0_32x32_q8_0;
static const tensor_traits_bf16_proj_q8_0       bf16_proj_q8_0_32x32;
static const tensor_traits_f32_proj_q8_0        f32_proj_q8_0_32x32;
static const tensor_traits_iq4_nl_proj_q8_0    iq4_nl_proj_q8_0_32x32;
static const tensor_traits_iq2_xs_proj_q8_0    iq2_xs_proj_q8_0_32x32;
static const tensor_traits_iq3_xxs_proj_q8_0   iq3_xxs_proj_q8_0_32x32;
static const tensor_traits_iq4_xs_proj_q8_0    iq4_xs_proj_q8_0_32x32;
static tensor_traits_passthrough          passthrough;
static const tensor_traits_iq_compact       iq2_xs_compact(GGML_TYPE_IQ2_XS);
static const tensor_traits_iq_compact       iq3_xxs_compact(GGML_TYPE_IQ3_XXS);
static const tensor_traits_iq_compact       iq4_xs_compact(GGML_TYPE_IQ4_XS);
static const tensor_traits_iq_compact       iq4_nl_compact(GGML_TYPE_IQ4_NL);
static const tensor_traits<block_mxfp4, 32, 32> mxfp4_32x32_q8_0;
static const tensor_traits<block_q5_K, 32, 32>  q5_k_32x32_q8_0;
static const tensor_traits<block_q5_1, 32, 32>  q5_1_32x32_q8_0;
static const tensor_traits<block_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
static const tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

static bool ggml_riscv64_spacemit_name_ends_with(const char * name, const char * suffix) {
    const size_t name_len   = std::strlen(name);
    const size_t suffix_len = std::strlen(suffix);
    return name_len >= suffix_len && std::strcmp(name + name_len - suffix_len, suffix) == 0;
}

static bool ggml_riscv64_spacemit_is_bf16_proj_q8_candidate(const ggml_tensor * cur) {
#if defined(RISCV64_SPACEMIT_IME2)
    return ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 && cur->type == GGML_TYPE_BF16 &&
           ggml_n_dims(cur) == 2 && cur->ne[0] % QK8_0 == 0 && cur->ne[1] % 32 == 0 &&
           std::strcmp(cur->name, "per_layer_model_proj.weight") == 0;
#else
    GGML_UNUSED(cur);
    return false;
#endif
}

static bool ggml_riscv64_spacemit_is_f32_proj_q8_candidate(const ggml_tensor * cur) {
#if defined(RISCV64_SPACEMIT_IME2)
    return ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 && cur->type == GGML_TYPE_F32 &&
           ggml_n_dims(cur) == 2 && cur->ne[0] % QK8_0 == 0 && cur->ne[1] % 32 == 0 &&
           std::strncmp(cur->name, "blk.", 4) == 0 &&
           (ggml_riscv64_spacemit_name_ends_with(cur->name, ".inp_gate.weight") ||
            ggml_riscv64_spacemit_name_ends_with(cur->name, ".proj.weight"));
#else
    GGML_UNUSED(cur);
    return false;
#endif
}

static bool ggml_riscv64_spacemit_iq_repack_enabled(ggml_type type) {
    const char * mode = std::getenv("GGML_RISCV64_SPACEMIT_IQ_REPACK");
    if (mode == nullptr || mode[0] == '\0') {
        // Default to the small 4-bit IQ formats. Repacking IQ2/IQ3 expert tensors
        // expands Qwen3.6 UD Q2_K_XL by ~22 GiB and does not fit on 32 GiB boards.
        return type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ4_XS;
    }
    if (std::strcmp(mode, "0") == 0 || std::strcmp(mode, "off") == 0 || std::strcmp(mode, "false") == 0) {
        return false;
    }
    if (std::strcmp(mode, "1") == 0 || std::strcmp(mode, "all") == 0 || std::strcmp(mode, "true") == 0) {
        return true;
    }
    if (std::strcmp(mode, "iq4") == 0 || std::strcmp(mode, "small") == 0) {
        return type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ4_XS;
    }
    if (std::strcmp(mode, "iq2") == 0) {
        return type == GGML_TYPE_IQ2_XS;
    }
    if (std::strcmp(mode, "iq3") == 0) {
        return type == GGML_TYPE_IQ3_XXS;
    }
    return false;
}

static bool ggml_riscv64_spacemit_is_iq4_nl_proj_q8_candidate(const ggml_tensor * cur) {
#if defined(RISCV64_SPACEMIT_IME2)
    return ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
           ggml_riscv64_spacemit_iq_repack_enabled(GGML_TYPE_IQ4_NL) && cur->type == GGML_TYPE_IQ4_NL && ggml_n_dims(cur) == 2 &&
           cur->ne[0] % QK8_0 == 0 && cur->ne[1] % 32 == 0;
#else
    GGML_UNUSED(cur);
    return false;
#endif
}
static bool ggml_riscv64_spacemit_is_iq_k_proj_q8_candidate(const ggml_tensor * cur, ggml_type type) {
#if defined(RISCV64_SPACEMIT_IME2)
    return ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 && ggml_riscv64_spacemit_iq_repack_enabled(type) &&
           cur->type == type && ggml_n_dims(cur) == 2 && cur->ne[0] % QK_K == 0 && cur->ne[1] % 32 == 0;
#else
    GGML_UNUSED(cur);
    GGML_UNUSED(type);
    return false;
#endif
}




static bool ggml_riscv64_spacemit_claim_iq_compact() {
    const char * claim = std::getenv("GGML_RISCV64_SPACEMIT_CLAIM_IQ_COMPACT");
    return !(claim && (std::strcmp(claim, "0") == 0 || std::strcmp(claim, "off") == 0 ||
                       std::strcmp(claim, "false") == 0));
}

static bool ggml_riscv64_spacemit_is_iq_compact_moe_tensor(const ggml_tensor * cur) {
    if (ggml_n_dims(cur) != 3) {
        return false;
    }
    switch (cur->type) {
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return true;
        default:
            return false;
    }
}


static bool ggml_riscv64_spacemit_type_disabled_for_repack(ggml_type type) {
    const char * disabled = std::getenv("GGML_RISCV64_SPACEMIT_DISABLE_REPACK_TYPES");
    if (disabled == nullptr || disabled[0] == '\0') {
        return false;
    }
    const char * name = ggml_type_name(type);
    const size_t name_len = std::strlen(name);
    const char * p = disabled;
    while (*p != '\0') {
        while (*p == ',' || *p == ':' || *p == ';' || *p == ' ') {
            ++p;
        }
        const char * start = p;
        while (*p != '\0' && *p != ',' && *p != ':' && *p != ';' && *p != ' ') {
            ++p;
        }
        if ((size_t) (p - start) == name_len && std::strncmp(start, name, name_len) == 0) {
            return true;
        }
    }
    return false;
}

static const ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(const ggml_tensor * cur) {
    const char * repack = std::getenv("GGML_RISCV64_SPACEMIT_REPACK");
    if (repack && (std::strcmp(repack, "0") == 0 || std::strcmp(repack, "off") == 0 ||
                   std::strcmp(repack, "false") == 0)) {
        return nullptr;
    }
    if (ggml_riscv64_spacemit_type_disabled_for_repack(cur->type)) {
        return nullptr;
    }
    if (ggml_riscv64_spacemit_is_bf16_proj_q8_candidate(cur)) {
        return &ggml::cpu::riscv64_spacemit::bf16_proj_q8_0_32x32;
    }
    if (ggml_riscv64_spacemit_is_f32_proj_q8_candidate(cur)) {
        return &ggml::cpu::riscv64_spacemit::f32_proj_q8_0_32x32;
    }
    if (ggml_riscv64_spacemit_is_iq4_nl_proj_q8_candidate(cur)) {
        return &ggml::cpu::riscv64_spacemit::iq4_nl_proj_q8_0_32x32;
    }
    if (ggml_riscv64_spacemit_is_iq_k_proj_q8_candidate(cur, GGML_TYPE_IQ2_XS)) {
        return &ggml::cpu::riscv64_spacemit::iq2_xs_proj_q8_0_32x32;
    }
    if (ggml_riscv64_spacemit_is_iq_k_proj_q8_candidate(cur, GGML_TYPE_IQ3_XXS)) {
        return &ggml::cpu::riscv64_spacemit::iq3_xxs_proj_q8_0_32x32;
    }
    if (ggml_riscv64_spacemit_is_iq_k_proj_q8_candidate(cur, GGML_TYPE_IQ4_XS)) {
        return &ggml::cpu::riscv64_spacemit::iq4_xs_proj_q8_0_32x32;
    }

    switch (cur->type) {
        case GGML_TYPE_Q2_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (!ggml_riscv64_spacemit_type_disabled_for_repack(cur->type) && cur->ne[1] % 32 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q3_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (!ggml_riscv64_spacemit_type_disabled_for_repack(cur->type) && cur->ne[1] % 32 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_1:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_k_32x256_q8_0;
                }

                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q6_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q8_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_MXFP4:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#endif
            }
            break;
        case GGML_TYPE_Q5_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q5_1:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q5_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#endif
            }
            break;
        default:
            break;
    }

    return nullptr;
}

static enum ggml_status ggml_backend_riscv64_spacemit_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (void *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));


    GGML_UNUSED(buffer);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_riscv64_spacemit_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    void * base = buffer->context;
    if (base == nullptr) {
        return;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(base);
}

static void * ggml_backend_riscv64_spacemit_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    void * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    return base;
}

static void ggml_backend_riscv64_spacemit_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               size_t                offset,
                                                               size_t                size) {
    GGML_ASSERT(tensor);
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_riscv64_spacemit_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);

    void * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    memset(base, value, buffer->size);
}

static void ggml_backend_riscv64_spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                            ggml_tensor *         tensor,
                                                            const void *          data,
                                                            size_t                offset,
                                                            size_t                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_base *) tensor->extra;
    if (tensor_traits) {
        auto OK = tensor_traits->repack(tensor, data, size);
        GGML_ASSERT(OK == 0);
    } else {
        memcpy(tensor->data, data, size);
    }

    GGML_UNUSED(buffer);
}

static const ggml_backend_buffer_i ggml_backend_riscv64_spacemit_buffer_i = {
    /* .free_buffer     = */ ggml_backend_riscv64_spacemit_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_riscv64_spacemit_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_riscv64_spacemit_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_riscv64_spacemit_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_riscv64_spacemit_buffer_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_riscv64_spacemit_buffer_clear,
    /* .reset           = */ nullptr,
};

static const char * ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_RISCV64_SPACEMIT";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                                        size_t size) {
    void * base = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    if (base == nullptr) {
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_riscv64_spacemit_buffer_i, base, size);
}

static size_t ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
    }

    GGML_UNUSED(buft);

    const auto plain_nbytes = [&]() {
        size_t total = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        return total;
    };

    if (ggml_riscv64_spacemit_is_bf16_proj_q8_candidate(tensor) ||
        ggml_riscv64_spacemit_is_f32_proj_q8_candidate(tensor) ||
        ggml_riscv64_spacemit_is_iq4_nl_proj_q8_candidate(tensor) ||
        ggml_riscv64_spacemit_is_iq_k_proj_q8_candidate(tensor, GGML_TYPE_IQ2_XS) ||
        ggml_riscv64_spacemit_is_iq_k_proj_q8_candidate(tensor, GGML_TYPE_IQ3_XXS) ||
        ggml_riscv64_spacemit_is_iq_k_proj_q8_candidate(tensor, GGML_TYPE_IQ4_XS)) {
        const int64_t nrow        = ggml_nrows(tensor);
        const int64_t padded_nrow = GGML_PAD(nrow, 32);
        const int64_t nblocks     = tensor->ne[0] / QK8_0;
        return (size_t) padded_nrow * nblocks * sizeof(block_q8_0);
    }

    const size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        return plain_nbytes();
    }

    const size_t row_nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;

    const auto add_strided_nbytes = [&](size_t total, size_t src_block_size, size_t dst_block_size) {
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * (tensor->nb[i] / src_block_size) * dst_block_size;
        }
        return total;
    };

    const auto remap_block_nbytes = [&](size_t src_block_size, size_t dst_block_size, int64_t padded_rows = 0) {
        GGML_ASSERT(row_nbytes % src_block_size == 0);

        size_t total =
            add_strided_nbytes((row_nbytes / src_block_size) * dst_block_size, src_block_size, dst_block_size);

        if (padded_rows > 0 && tensor->ne[1] % padded_rows != 0) {
            total += (padded_rows - tensor->ne[1] % padded_rows) * (tensor->nb[1] / src_block_size) * dst_block_size;
        }

        return total;
    };

    size_t nbytes = row_nbytes;
    switch (tensor->type) {
        case GGML_TYPE_Q4_K:
            nbytes = remap_block_nbytes(sizeof(block_q4_K), sizeof(block_q4_1) * 8);
            break;
        case GGML_TYPE_Q6_K:
            nbytes = remap_block_nbytes(sizeof(block_q6_K), sizeof(block_q8_0) * 8, 32);
            break;
        case GGML_TYPE_Q8_0:
            nbytes = remap_block_nbytes(sizeof(block_q8_0), sizeof(block_q8_0), 32);
            break;
        case GGML_TYPE_Q2_K:
            nbytes = remap_block_nbytes(sizeof(block_q2_K), sizeof(spacemit_kernels::nrow_block_q2_k<1>));
            break;
        case GGML_TYPE_Q3_K:
            nbytes = remap_block_nbytes(sizeof(block_q3_K), sizeof(spacemit_kernels::nrow_block_q3_k<1>));
            break;
        case GGML_TYPE_MXFP4:
            nbytes = remap_block_nbytes(sizeof(block_mxfp4), sizeof(spacemit_kernels::nrow_block_mxfp4<1>));
            break;
        case GGML_TYPE_Q5_K:
            nbytes = remap_block_nbytes(sizeof(block_q5_K), sizeof(spacemit_kernels::nrow_block_q5_1<1>) * 8);
            break;
        case GGML_TYPE_Q5_1:
            nbytes = remap_block_nbytes(sizeof(block_q5_1), sizeof(spacemit_kernels::nrow_block_q5_1<1>));
            break;
        case GGML_TYPE_Q5_0:
            nbytes = remap_block_nbytes(sizeof(block_q5_0), sizeof(spacemit_kernels::nrow_block_q5_0<1>));
            break;
        default:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            break;
    }

    return nbytes;
}

namespace ggml::cpu::riscv64_spacemit {

class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_type_disabled_for_repack(op->src[0]->type)) {
                    return true;
                }
                if (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        return false;
                    }
                    if (op->src[1]->type == GGML_TYPE_F32) {
                        return true;
                    }
                }
                break;
            case GGML_OP_MUL_MAT_ID:
                if (ggml_riscv64_spacemit_claim_iq_compact() && ggml_riscv64_spacemit_is_iq_compact_moe_tensor(op->src[0])) {
                    if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        return false;
                    }
                    return op->src[1]->type == GGML_TYPE_F32;
                }
                if (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_type_disabled_for_repack(op->src[0]->type)) {
                    return true;
                }
                if (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        return false;
                    }
                    if (op->src[1]->type == GGML_TYPE_F32) {
                        return true;
                    }
                }
                break;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
                if (op->op == GGML_OP_MUL_MAT_ID && ggml_riscv64_spacemit_claim_iq_compact() &&
                    ggml_riscv64_spacemit_is_iq_compact_moe_tensor(op->src[0])) {
                    switch (op->src[0]->type) {
                        case GGML_TYPE_IQ2_XS:
                            return (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::iq2_xs_compact);
                        case GGML_TYPE_IQ3_XXS:
                            return (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::iq3_xxs_compact);
                        case GGML_TYPE_IQ4_XS:
                            return (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::iq4_xs_compact);
                        case GGML_TYPE_IQ4_NL:
                            return (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::iq4_nl_compact);
                        default:
                            break;
                    }
                }
                if (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
                    if (ggml_riscv64_spacemit_type_disabled_for_repack(op->src[0]->type)) {
                        return (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::passthrough);
                    }
                    return (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV:
            case GGML_OP_FLASH_ATTN_EXT:
            case GGML_OP_CONT:
            case GGML_OP_CPY:
            case GGML_OP_REPEAT:
            case GGML_OP_SUM_ROWS:
            case GGML_OP_GET_ROWS:
            case GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                {
                    const char * common_ops = std::getenv("GGML_RISCV64_SPACEMIT_COMMON_OPS");
                    if (common_ops && (std::strcmp(common_ops, "0") == 0 || std::strcmp(common_ops, "off") == 0 ||
                                       std::strcmp(common_ops, "false") == 0)) {
                        return nullptr;
                    }
                    return (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
                }
            default:
                // GGML_ABORT("fatal error");
                break;
        }

        return nullptr;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(void) {
    static ggml_backend_buffer_type ggml_backend_cpu_buffer_type_riscv64_spacemit = {
  /* .iface    = */
        {
         /* .get_name         = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name,
         /* .alloc_buffer     = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer,
         /* .get_alignment    = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment,
         /* .get_max_size     = */ nullptr,
         /* .get_alloc_size   = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .is_host          = */ nullptr,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        new ggml::cpu::riscv64_spacemit::extra_buffer_type(),
    };

    return &ggml_backend_cpu_buffer_type_riscv64_spacemit;
}

extern "C" {
static int bind_ai_thread() {
    int  fd, bytes;
    char str[32];

    fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        return -1;
    }

    snprintf(str, 16, "%d", 0);
    bytes = write(fd, str, strlen(str));
    if (bytes < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        close(fd);
        return -1;
    }

    close(fd);
    return 0;
}

void ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(int thread_n) {
    if (!ggml_riscv64_spacemit_thread_affinity_enabled()) {
        return;
    }
    int cpu_id = sched_getcpu();
    if (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_mask)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    if (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        const auto & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        if (thread_n < 0 || static_cast<size_t>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        auto perfer_cpu_id = perfer_core_ids[static_cast<size_t>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        int s =
            pthread_setaffinity_np(main_thread, sizeof(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        if (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        int ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_offset;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_size;
    }

    if (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        void * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        if (rt == nullptr) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

void ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(int thread_n) {
    if (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        auto rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        if (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
