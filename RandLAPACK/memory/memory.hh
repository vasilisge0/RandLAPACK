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
template Status malloc<Device::CPU, double>(size_t len, double** values);
template Status malloc<Device::CPU, float>(size_t len, float** values);
template Status malloc<Device::CPU, half>(size_t len, half** values);
template Status malloc<Device::CPU, int>(size_t len, int** values);
template Status malloc<Device::CUDA, double>(size_t len, double** values);
template Status malloc<Device::CUDA, float>(size_t len, float** values);
template Status malloc<Device::CUDA, half>(size_t len, half** values);
template Status malloc<Device::CUDA, int>(size_t len, int** values);

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

template Status free<Device::CPU, double>(double* values);
template Status free<Device::CPU, float>(float* values);
template Status free<Device::CPU, half>(half* values);
template Status free<Device::CPU, int>(int* values);
template Status free<Device::CUDA, double>(double* values);
template Status free<Device::CUDA, float>(float* values);
template Status free<Device::CUDA, half>(half* values);
template Status free<Device::CUDA, int>(int* values);

struct StridedStorage {
    dim<2> size_ = {0, 0};
    NumericPtrVariant values_;
    Layout layout_ = RandLAPACK::Layout::COL_MAJOR;
    size_t lead_dim_ = 0;
    StridedStorage() = default;
    StridedStorage(dim<2> size, size_t lead_dim, Layout layout)
        : size_{size}, lead_dim_{lead_dim}, layout_{layout} {}
    StridedStorage(dim<2> size, size_t lead_dim, Layout layout,
                   NumericPtrVariant values)
        : size_{size}, lead_dim_{lead_dim}, layout_{layout}, values_{values} {}
};  // struct StridedStorage

struct CsrStorage {
    dim<2> size_ = {0, 0};
    size_t nnz_;
    NumericPtrVariant values_;
    IntPtrVariant row_ptrs_;
    IntPtrVariant col_idxs_;
    CsrStorage() = default;
    CsrStorage(dim<2> size, size_t nnz, NumericType value_type,
               IntType index_type)
        : size_{size}, nnz_{nnz} {}
    CsrStorage(dim<2> size, size_t nnz, NumericPtrVariant values,
               IntPtrVariant row_ptrs, IntPtrVariant col_idxs, IntType)
        : size_{size},
          nnz_{nnz},
          values_{values},
          row_ptrs_{row_ptrs},
          col_idxs_{col_idxs} {}
};  // struct CsrStorage

enum class DenseFormat : std::size_t { STRIDED = 0, Count };
enum class SparseFormat : std::size_t { CSR = 1, Count };
enum class MatrixFormat : std::size_t { STRIDED = 0, CSR = 1, Count };

using DenseStorage = std::variant<StridedStorage>;
using SparseStorage = std::variant<CsrStorage>;

}  // namespace RandLAPACK