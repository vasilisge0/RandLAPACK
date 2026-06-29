#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "csr_storage.hh"
#include "dense_storage.hh"
#include "dimensions.hh"
#include "memory.hh"
#include "precision.hh"
#include "storage_types.hh"
#include "type_macros.hh"

namespace RandLAPACK {
namespace cuda {

template <typename value_t, typename value_out_t>
__global__ void copy_kernel(size_t nrows, size_t ncols, value_t* mtx_in,
                            int ld_in, value_out_t* mtx_out, int ld_out) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < nrows) && (col < ncols)) {
        if constexpr (std::is_same_v<value_t, value_out_t>) {
            mtx_out[row + ld_out * col] = mtx_in[row + ld_in * col];
        } else {
            mtx_out[row + ld_out * col] =
                static_cast<value_out_t>(mtx_in[row + ld_in * col]);
        }
    }
}

#define INST_COPY_KERNEL(T1, T2)                                           \
    template __global__ void copy_kernel<T1, T2>(size_t, size_t, T1*, int, \
                                                 T2*, int);

FOR_ALL_FP_PAIRS(INST_COPY_KERNEL)
FOR_ALL_INDEX_PAIRS(INST_COPY_KERNEL)

template <typename value_in_t, typename value_out_t>
__host__ void copy_launcher(dim<2> size, value_in_t* mtx_in, int ld_in,
                            value_out_t* mtx_out, int ld_out) {
    dim3 block_size(16, 16);
    dim3 grid_size((size[1] + block_size.x - 1) / block_size.x,
                   (size[0] + block_size.y - 1) / block_size.y);
    copy_kernel<<<grid_size, block_size>>>(size[0], size[1], mtx_in, ld_in,
                                           mtx_out, ld_out);
}

#define INST_COPY_LAUNCHER(value_in_t, value_out_t)                \
    template __host__ void copy_launcher<value_in_t, value_out_t>( \
        dim<2>, value_in_t*, int, value_out_t*, int);

FOR_ALL_FP_PAIRS(INST_COPY_LAUNCHER)
FOR_ALL_INDEX_PAIRS(INST_COPY_LAUNCHER)

template <typename value_in_t, typename value_out_t>
__host__ void dense_copy_impl(dim<2> size, DenseStorage& source,
                              DenseStorage& target) {
    StridedStorage& x = std::get<StridedStorage>(source);
    StridedStorage& y = std::get<StridedStorage>(target);
    copy_launcher(size, std::get<value_in_t*>(x.values_),
                  x.lead_dim_, std::get<value_out_t*>(y.values_),
                  y.lead_dim_);
}

#define INST_DENSE_COPY_IMPL(T1, T2)                                      \
    template __host__ void dense_copy_impl<T1, T2>(dim<2>, DenseStorage&, \
                                                   DenseStorage&);

FOR_ALL_FP_PAIRS(INST_DENSE_COPY_IMPL)

template <typename value_in_t, typename value_out_t, typename index_in_t,
          typename index_out_t>
__host__ void csr_copy_impl(dim<2> size, SparseStorage& source,
                            SparseStorage& target) {
    if ((static_cast<size_t>(source.index()) ==
         static_cast<size_t>(RandLAPACK::MatrixFormat::CSR)) &&
        (static_cast<size_t>(target.index()) ==
         static_cast<size_t>(RandLAPACK::MatrixFormat::CSR))) {
        CsrStorage& x = std::get<CsrStorage>(source);
        CsrStorage& y = std::get<CsrStorage>(target);
        copy_launcher(dim<2>{x.nnz_, static_cast<size_t>(1)},
                      std::get<value_out_t*>(x.values_), x.nnz_,
                      std::get<value_in_t*>(y.values_), x.nnz_);
        copy_launcher(dim<2>{size[0] + 1, static_cast<size_t>(1)},
                      std::get<index_out_t*>(x.row_ptrs_), size[0] + 1,
                      std::get<index_in_t*>(y.row_ptrs_), size[0] + 1);
        copy_launcher(dim<2>{x.nnz_, static_cast<size_t>(1)},
                      std::get<index_out_t*>(x.col_idxs_), x.nnz_,
                      std::get<index_in_t*>(y.col_idxs_), x.nnz_);
    } else {
        throw std::runtime_error(
            "Copying values from different storage types is not supported by "
            "csr_copy_impl.");
    }
}  // copy_impl

#define INST_CSR_COPY_IMPL(value_in_t, value_out_t, index_in_t, index_out_t) \
    template __host__ void                                                   \
    csr_copy_impl<value_in_t, value_out_t, index_in_t, index_out_t>(         \
        dim<2>, SparseStorage&, SparseStorage&);

FOR_ALL_FP_PAIRS_INDEX_PAIRS(INST_CSR_COPY_IMPL)

}  // namespace cuda
}  // namespace RandLAPACK