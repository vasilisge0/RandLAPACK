#pragma once

#include <cuda_fp16.h>

#include <variant>

using half = __half;

namespace RandLAPACK {

enum class NumericType : std::size_t { FP64 = 0, FP32 = 1, FP16 = 2, Count };
enum class IntType : std::size_t { INT64 = 0, INT32 = 1, Count };

using NumericPtrVariant = std::variant<double*, float*, __half*, uint32_t*>;
using IntPtrVariant = std::variant<long int*, int*>;

// Type mappings between concrete and enum types.

template <typename T>
struct to_enum_type {
    static constexpr NumericType value = NumericType::Count;
};

template <>
struct to_enum_type<double> {
    static constexpr NumericType value = NumericType::FP64;
};
template <>
struct to_enum_type<float> {
    static constexpr NumericType value = NumericType::FP32;
};
template <>
struct to_enum_type<half> {
    static constexpr NumericType value = NumericType::FP16;
};

template <typename T>
constexpr NumericType numeric_type_of = to_enum_type<T>::value;

template <NumericType N>
struct to_concrete_type;

template <>
struct to_concrete_type<NumericType::FP64> {
    using value = double;
};
template <>
struct to_concrete_type<NumericType::FP32> {
    using value = float;
};
template <>
struct to_concrete_type<NumericType::FP16> {
    using value = half;
};

template <IntType I>
struct to_concrete_integer_type;
template <>
struct to_concrete_integer_type<IntType::INT64> {
    using value = int64_t;
};
template <>
struct to_concrete_integer_type<IntType::INT32> {
    using value = int32_t;
};

// Metaprogramming utilities for kernel registration.

// Compile-time value list.
template <auto... Vs>
struct vlist {};

// Compile-time for_each.
template <auto... Vs, typename F>
void for_each(vlist<Vs...>, F&& f) {
    (f.template operator()<Vs>(), ...);
}

// Lists of all numeric types and integer types for iteration.
using all_numerics =
    vlist<NumericType::FP64, NumericType::FP32, NumericType::FP16>;
using all_ints = vlist<IntType::INT64, IntType::INT32>;

}  // namespace RandLAPACK
