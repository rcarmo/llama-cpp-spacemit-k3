#pragma once

#include "ggml-common.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int repack(ggml_tensor * t, const void * data, size_t data_size);

int repack_bf16_to_q8_0_32x32(ggml_tensor * t, const void * data, size_t data_size);
int repack_f32_to_q8_0_32x32(ggml_tensor * t, const void * data, size_t data_size);

}  // namespace ggml::cpu::riscv64_spacemit
