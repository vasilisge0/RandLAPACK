#pragma once

#include "device.hh"
#include "precision.hh"
#include "status.hh"

namespace RandLAPACK {

template <Device device, typename value_t>
Status malloc(size_t len, value_t** values) {
    if constexpr (device == RandLAPACK::Device::CPU) {
        *values = (value_t*)::malloc((sizeof **values) * len);
        return (*values != nullptr) ? RandLAPACK::SUCCESS : RandLAPACK::FAILURE;
    } else if constexpr (device == RandLAPACK::Device::CUDA) {
#if USE_CUDA
        cudaError_t status;
        status = cudaMalloc((void**)values, len * (sizeof **values));
        return (status == 0) ? RandLAPACK::SUCCESS : RandLAPACK::FAILURE;
#else
        throw std::invalid_argument("Compiled without Device::CUDA.");
#endif
    } else {
        throw std::invalid_argument(
            "Device has to be either RandLAPACK::Device::CPU or "
            "RandLAPACK::Device::CUDA");
    }
}  // malloc

#define _INST_MALLOC(device, T) \
    template Status malloc<device, T>(size_t len, T** values);

#define INST_MALLOC_ALL_TYPES(device) \
    _INST_MALLOC(device, double)      \
    _INST_MALLOC(device, float)       \
    _INST_MALLOC(device, half)        \
    _INST_MALLOC(device, int)

INST_MALLOC_ALL_TYPES(Device::CPU)
INST_MALLOC_ALL_TYPES(Device::CUDA)

#undef _INST_MALLOC
#undef INST_MALLOC_ALL_TYPES

template <Device device, typename value_t>
Status free(value_t* values) {
    if constexpr (device == RandLAPACK::Device::CPU) {
        ::free(values);
        values = nullptr;
        return RandLAPACK::SUCCESS;
    } else if constexpr (device == RandLAPACK::Device::CUDA) {
#if USE_CUDA
        cudaError_t status = cudaFree((void*)values);
        values = nullptr;
        return (status == 0) ? RandLAPACK::SUCCESS : RandLAPACK::FAILURE;
#else
        throw std::invalid_argument("Compiled without Device::CUDA.");
#endif
    } else {
        throw std::invalid_argument(
            "Device has to be either RandLAPACK::Device::CPU or "
            "RandLAPACK::Device::CUDA");
    }
}  // free

#define _INST_FREE(device, T) \
    template Status free<device, T>(T* values);

#define INST_FREE_ALL_TYPES(device) \
    _INST_FREE(device, double)      \
    _INST_FREE(device, float)       \
    _INST_FREE(device, half)        \
    _INST_FREE(device, int)

INST_FREE_ALL_TYPES(Device::CPU)
INST_FREE_ALL_TYPES(Device::CUDA)

#undef _INST_FREE
#undef INST_FREE_ALL_TYPES

}  // namespace RandLAPACK