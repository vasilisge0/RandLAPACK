#pragma once

#include <variant>
#include <cuda_fp16.h>
using half = __half;

namespace RandLAPACK {

enum class NumericType : std::size_t { FP64 = 0, FP32 = 1, FP16 = 2, Count };
enum class IntType : std::size_t { INT64 = 0, INT32 = 1, Count };

using NumericPtrVariant = std::variant<double*, float*, __half*, uint32_t*>;
using IntPtrVariant     = std::variant<long int*, int*>;

} // namespace RandLAPACK
