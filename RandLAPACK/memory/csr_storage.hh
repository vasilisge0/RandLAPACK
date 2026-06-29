#pragma once

#include "device.hh"
#include "dimensions.hh"
#include "memory.hh"
#include "precision.hh"
#include "storage_types.hh"
#include "type_macros.hh"

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>

#include "storage_cuda_kernels.cuh"
#endif

#include "storage_cpu_kernels.hh"

namespace RandLAPACK {

template <Device device, typename value_t, typename index_t>
void initialize_csr_storage(dim<2> size, SparseStorage& source) {
    CsrStorage& S = std::get<CsrStorage>(source);

    value_t* t;
    malloc<device, value_t>(S.nnz_, &t);
    S.values_ = t;

    index_t* rp;
    malloc<device, index_t>(size[0] + 1, &rp);
    S.row_ptrs_ = rp;

    index_t* ci;
    malloc<device, index_t>(S.nnz_, &ci);
    S.col_idxs_ = ci;
}  //  initialize_csr_storage

INSTANTIATE_ALL_DEVICES_TYPES_INDEX(initialize_csr_storage,
                                    (dim<2>, SparseStorage&))

template <Device device, typename value_t, typename index_t>
void free_csr_storage(SparseStorage& source) {
    CsrStorage& S = std::get<CsrStorage>(source);

    value_t* v = std::get<value_t*>(S.values_);
    if (v != nullptr) free<device, value_t>(v);

    index_t* rp = std::get<index_t*>(S.row_ptrs_);
    if (rp != nullptr) free<device, index_t>(rp);

    index_t* ci = std::get<index_t*>(S.col_idxs_);
    if (ci != nullptr) free<device, index_t>(ci);
}  //  free_csr_storage

INSTANTIATE_ALL_DEVICES_TYPES_INDEX(free_csr_storage, (SparseStorage&))

template <Device device_source, Device device_target, typename value_in_t,
          typename value_out_t, typename index_in_t, typename index_out_t>
void copy_csr_storage(dim<2> size, SparseStorage& source,
                      SparseStorage& target) {
    if constexpr (device_source == RandLAPACK::Device::CPU &&
                  device_target == RandLAPACK::Device::CPU) {
        RandLAPACK::cpu::csr_copy_impl<value_in_t, value_out_t, index_in_t,
                                       index_out_t>(size, source, target);
    } else if constexpr (device_source == RandLAPACK::Device::CUDA &&
                         device_target == RandLAPACK::Device::CUDA) {
#ifdef __CUDACC__
        RandLAPACK::cuda::csr_copy_impl<value_in_t, value_out_t, index_in_t,
                                        index_out_t>(size, source, target);
#else
        throw std::invalid_argument("Compiled without CUDA support.");
#endif
    } else {
        throw std::invalid_argument(
            "Copying between the specified devices is not supported.");
    }
}  //  copy_csr_storage

}  // namespace RandLAPACK