#pragma once

#include <cuda_fp16.h>

#include <variant>
using half = __half;

namespace RandLAPACK {

enum class NumericType : std::size_t { FP64 = 0, FP32 = 1, FP16 = 2, Count };
enum class IntType : std::size_t { INT64 = 0, INT32 = 1, Count };

using NumericPtrVariant = std::variant<double*, float*, __half*, uint32_t*>;
using IntPtrVariant = std::variant<long int*, int*>;

template <typename T>
struct numeric_trait {
    static constexpr NumericType value = NumericType::Count;
};

template <>
struct numeric_trait<double> {
    static constexpr NumericType value = NumericType::FP64;
};
template <>
struct numeric_trait<float> {
    static constexpr NumericType value = NumericType::FP32;
};
template <>
struct numeric_trait<half> {
    static constexpr NumericType value = NumericType::FP16;
};

}  // namespace RandLAPACK
