#pragma once

namespace RandLAPACK {
namespace experimental {

// Strip one layer of parentheses — used to pass comma-containing signatures as a single arg.
#define REMOVE_PARENS(...) __VA_ARGS__

// ---------------------------------------------------------------------------
// Flat iterators — no nested macro calls, so no preprocessor blue-paint issue.
// Each calls F(T1, ..., TN, extra...) for all 3^N combinations.
// ---------------------------------------------------------------------------
#define FOR_ALL_SINGLES(F, ...)         \
    F(double, ##__VA_ARGS__)            \
    F(float,  ##__VA_ARGS__)            \
    F(__half, ##__VA_ARGS__)            \
    F(uint32_t, ##__VA_ARGS__)

#define FOR_ALL_PAIRS(F, ...)           \
    F(double, double, ##__VA_ARGS__)    \
    F(double, float,  ##__VA_ARGS__)    \
    F(double, __half, ##__VA_ARGS__)    \
    F(float,  double, ##__VA_ARGS__)    \
    F(float,  float,  ##__VA_ARGS__)    \
    F(float,  __half, ##__VA_ARGS__)    \
    F(__half, double, ##__VA_ARGS__)    \
    F(__half, float,  ##__VA_ARGS__)    \
    F(__half, __half, ##__VA_ARGS__)    \
    F(uint32_t, double, ##__VA_ARGS__)  \
    F(uint32_t, float,  ##__VA_ARGS__)  \
    F(uint32_t, __half, ##__VA_ARGS__)

#define FOR_ALL_TRIPLES(F, ...)                  \
    F(double, double, double, ##__VA_ARGS__)     \
    F(double, double, float,  ##__VA_ARGS__)     \
    F(double, double, __half, ##__VA_ARGS__)     \
    F(double, float,  double, ##__VA_ARGS__)     \
    F(double, float,  float,  ##__VA_ARGS__)     \
    F(double, float,  __half, ##__VA_ARGS__)     \
    F(double, __half, double, ##__VA_ARGS__)     \
    F(double, __half, float,  ##__VA_ARGS__)     \
    F(double, __half, __half, ##__VA_ARGS__)     \
    F(float,  double, double, ##__VA_ARGS__)     \
    F(float,  double, float,  ##__VA_ARGS__)     \
    F(float,  double, __half, ##__VA_ARGS__)     \
    F(float,  float,  double, ##__VA_ARGS__)     \
    F(float,  float,  float,  ##__VA_ARGS__)     \
    F(float,  float,  __half, ##__VA_ARGS__)     \
    F(float,  __half, double, ##__VA_ARGS__)     \
    F(float,  __half, float,  ##__VA_ARGS__)     \
    F(float,  __half, __half, ##__VA_ARGS__)     \
    F(__half, double, double, ##__VA_ARGS__)     \
    F(__half, double, float,  ##__VA_ARGS__)     \
    F(__half, double, __half, ##__VA_ARGS__)     \
    F(__half, float,  double, ##__VA_ARGS__)     \
    F(__half, float,  float,  ##__VA_ARGS__)     \
    F(__half, float,  __half, ##__VA_ARGS__)     \
    F(__half, __half, double, ##__VA_ARGS__)     \
    F(__half, __half, float,  ##__VA_ARGS__)     \
    F(__half, __half, __half, ##__VA_ARGS__)     \
    F(uint32_t, double, double, ##__VA_ARGS__)   \
    F(uint32_t, double, float,  ##__VA_ARGS__)   \
    F(uint32_t, double, __half, ##__VA_ARGS__)   \
    F(uint32_t, float,  double, ##__VA_ARGS__)   \
    F(uint32_t, float,  float,  ##__VA_ARGS__)   \
    F(uint32_t, float,  __half, ##__VA_ARGS__)   \
    F(uint32_t, __half, double, ##__VA_ARGS__)   \
    F(uint32_t, __half, float,  ##__VA_ARGS__)   \
    F(uint32_t, __half, __half, ##__VA_ARGS__)

// ---------------------------------------------------------------------------
// Explicit instantiation helpers.
// Pass the parameter list wrapped in () to avoid comma ambiguity.
// ---------------------------------------------------------------------------
#define INSTANTIATE_1(T1,         func, sig)  template void func<T1>            (REMOVE_PARENS sig);
#define INSTANTIATE_2(T1, T2,     func, sig)  template void func<T1, T2>        (REMOVE_PARENS sig);
#define INSTANTIATE_3(T1, T2, T3, func, sig)  template void func<T1, T2, T3>    (REMOVE_PARENS sig);

// ---------------------------------------------------------------------------
// INSTANTIATE_ALL — counts placeholder args to determine template arity.
//
// Usage (placeholder names should match the function's template param names):
//   INSTANTIATE_ALL(func, (sig), value_t)          // 1 param  →  3 instantiations
//   INSTANTIATE_ALL(func, (sig), T1, T2)           // 2 params →  9 instantiations
//   INSTANTIATE_ALL(func, (sig), A_T, X_T, Y_T)   // 3 params → 27 instantiations
// ---------------------------------------------------------------------------
#define _NARGS_IMPL(_1, _2, _3, N, ...) N
#define _NARGS(...)  _NARGS_IMPL(__VA_ARGS__, 3, 2, 1, ~)

#define _FOR_ALL_1   FOR_ALL_SINGLES
#define _FOR_ALL_2   FOR_ALL_PAIRS
#define _FOR_ALL_3   FOR_ALL_TRIPLES
#define _INST_N_1    INSTANTIATE_1
#define _INST_N_2    INSTANTIATE_2
#define _INST_N_3    INSTANTIATE_3

#define _INST_ALL_EXPAND(N, func, sig)    _FOR_ALL_##N(_INST_N_##N, func, sig)
#define _INST_ALL_DISPATCH(N, func, sig)  _INST_ALL_EXPAND(N, func, sig)
#define INSTANTIATE_ALL(func, sig, ...)   _INST_ALL_DISPATCH(_NARGS(__VA_ARGS__), func, sig)

// ---------------------------------------------------------------------------
// FP-only iterators (double, float, __half) and INSTANTIATE_ALL_FP.
// ---------------------------------------------------------------------------
#define FOR_ALL_FP_SINGLES(F, ...)           \
    F(double, ##__VA_ARGS__)                 \
    F(float,  ##__VA_ARGS__)                 \
    F(__half, ##__VA_ARGS__)

#define FOR_ALL_FP_PAIRS(F, ...)             \
    F(double, double, ##__VA_ARGS__)         \
    F(double, float,  ##__VA_ARGS__)         \
    F(double, __half, ##__VA_ARGS__)         \
    F(float,  double, ##__VA_ARGS__)         \
    F(float,  float,  ##__VA_ARGS__)         \
    F(float,  __half, ##__VA_ARGS__)         \
    F(__half, double, ##__VA_ARGS__)         \
    F(__half, float,  ##__VA_ARGS__)         \
    F(__half, __half, ##__VA_ARGS__)

#define FOR_ALL_FP_TRIPLES(F, ...)               \
    F(double, double, double, ##__VA_ARGS__)     \
    F(double, double, float,  ##__VA_ARGS__)     \
    F(double, double, __half, ##__VA_ARGS__)     \
    F(double, float,  double, ##__VA_ARGS__)     \
    F(double, float,  float,  ##__VA_ARGS__)     \
    F(double, float,  __half, ##__VA_ARGS__)     \
    F(double, __half, double, ##__VA_ARGS__)     \
    F(double, __half, float,  ##__VA_ARGS__)     \
    F(double, __half, __half, ##__VA_ARGS__)     \
    F(float,  double, double, ##__VA_ARGS__)     \
    F(float,  double, float,  ##__VA_ARGS__)     \
    F(float,  double, __half, ##__VA_ARGS__)     \
    F(float,  float,  double, ##__VA_ARGS__)     \
    F(float,  float,  float,  ##__VA_ARGS__)     \
    F(float,  float,  __half, ##__VA_ARGS__)     \
    F(float,  __half, double, ##__VA_ARGS__)     \
    F(float,  __half, float,  ##__VA_ARGS__)     \
    F(float,  __half, __half, ##__VA_ARGS__)     \
    F(__half, double, double, ##__VA_ARGS__)     \
    F(__half, double, float,  ##__VA_ARGS__)     \
    F(__half, double, __half, ##__VA_ARGS__)     \
    F(__half, float,  double, ##__VA_ARGS__)     \
    F(__half, float,  float,  ##__VA_ARGS__)     \
    F(__half, float,  __half, ##__VA_ARGS__)     \
    F(__half, __half, double, ##__VA_ARGS__)     \
    F(__half, __half, float,  ##__VA_ARGS__)     \
    F(__half, __half, __half, ##__VA_ARGS__)

#define _FOR_ALL_FP_1   FOR_ALL_FP_SINGLES
#define _FOR_ALL_FP_2   FOR_ALL_FP_PAIRS
#define _FOR_ALL_FP_3   FOR_ALL_FP_TRIPLES

#define _INST_FP_ALL_EXPAND(N, func, sig)   _FOR_ALL_FP_##N(_INST_N_##N, func, sig)
#define _INST_FP_ALL_DISPATCH(N, func, sig) _INST_FP_ALL_EXPAND(N, func, sig)
#define INSTANTIATE_ALL_FP(func, sig, ...)  _INST_FP_ALL_DISPATCH(_NARGS(__VA_ARGS__), func, sig)

#define FOR_ALL_INDEX_TYPES(F, ...)      \
    F(int32_t, ##__VA_ARGS__)            \
    F(int64_t, ##__VA_ARGS__)

#define FOR_ALL_DEVICES(F, ...)         \
    F(Device::CPU,  ##__VA_ARGS__)      \
    F(Device::OMP,  ##__VA_ARGS__)      \
    F(Device::CUDA, ##__VA_ARGS__)

#define _INST_DEVICE_TYPE(device, value_t, func, sig) \
    template void func<device, value_t>(REMOVE_PARENS sig);

#define INSTANTIATE_ALL_DEVICES_AND_TYPES(func, sig) \
    FOR_ALL_DEVICES(_INST_DEVICE_TYPE_OUTER, func, sig)

#define _INST_DEVICE_TYPE_OUTER(device, func, sig) \
    _INST_DEVICE_TYPE(device, double,   func, sig)  \
    _INST_DEVICE_TYPE(device, float,    func, sig)  \
    _INST_DEVICE_TYPE(device, __half,   func, sig)  \
    _INST_DEVICE_TYPE(device, uint32_t, func, sig)

#define _INST_DTI(device, value_t, index_t, func, sig) \
    template void func<device, value_t, index_t>(REMOVE_PARENS sig);

#define _INST_DTI_INDEX(index_t, device, value_t, func, sig) \
    _INST_DTI(device, value_t, index_t, func, sig)

#define _INST_DTI_TYPE(value_t, device, func, sig)      \
    _INST_DTI_INDEX(int32_t, device, value_t, func, sig) \
    _INST_DTI_INDEX(int64_t, device, value_t, func, sig)

#define _INST_DTI_DEVICE(device, func, sig)          \
    _INST_DTI_TYPE(double,   device, func, sig)      \
    _INST_DTI_TYPE(float,    device, func, sig)      \
    _INST_DTI_TYPE(__half,   device, func, sig)      \
    _INST_DTI_TYPE(uint32_t, device, func, sig)

#define INSTANTIATE_ALL_DEVICES_TYPES_INDEX(func, sig) \
    FOR_ALL_DEVICES(_INST_DTI_DEVICE, func, sig)

// ---------------------------------------------------------------------------
// INSTANTIATE_ALL_DEVICE_PAIRS_FP_PAIRS
// For template<Device ds, Device dt, typename vi, typename vo>
// Generates 3 * 3 * 3 * 3 = 81 instantiations.
// ---------------------------------------------------------------------------
#define _INST_DPDP(ds, dt, vi, vo, func, sig) \
    template void func<ds, dt, vi, vo>(REMOVE_PARENS sig);

#define _INST_DPDP_VIN(vi, ds, dt, func, sig)    \
    _INST_DPDP(ds, dt, vi, double, func, sig)     \
    _INST_DPDP(ds, dt, vi, float,  func, sig)     \
    _INST_DPDP(ds, dt, vi, __half, func, sig)

#define _INST_DPDP_DTAR(dt, ds, func, sig)       \
    _INST_DPDP_VIN(double, ds, dt, func, sig)    \
    _INST_DPDP_VIN(float,  ds, dt, func, sig)    \
    _INST_DPDP_VIN(__half, ds, dt, func, sig)

#define _INST_DPDP_DSRC(ds, func, sig)            \
    _INST_DPDP_DTAR(Device::CPU,  ds, func, sig)  \
    _INST_DPDP_DTAR(Device::OMP,  ds, func, sig)  \
    _INST_DPDP_DTAR(Device::CUDA, ds, func, sig)

#define INSTANTIATE_ALL_DEVICE_PAIRS_FP_PAIRS(func, sig) \
    FOR_ALL_DEVICES(_INST_DPDP_DSRC, func, sig)

}   // namespace experimental
}   // namespace RandLAPACK