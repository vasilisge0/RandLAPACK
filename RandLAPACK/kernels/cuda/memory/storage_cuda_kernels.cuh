#pragma once

#include "precision.hh"
#include "memory.hh"
#include "dimensions.hh"
#include "type_macros.hh"
#include "precision.hh"
#include "dense_storage.hh"
#include "csr_storage.hh"

namespace RandLAPACK {
namespace cuda {

template <typename value_t, typename value_out_t>
__global__ void copy_kernel(size_t nrows, size_t ncols, value_t* mtx_in, int ld_in, value_out_t* mtx_out, int ld_out)
{
    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < nrows) && (col < ncols)) {
        if constexpr (std::is_same_v<value_t, value_out_t>) {
            mtx_out[row + ld_out * col] = mtx_in[row + ld_in * col];
        }
        else {
            mtx_out[row + ld_out * col] = static_cast<value_out_t>(mtx_in[row + ld_in * col]);
        }
    }
}   // copy_kernel

#define INST_COPY_KERNEL(T1, T2) \
    template __global__ void copy_kernel<T1, T2>(size_t, size_t, T1*, int, T2*, int);

FOR_ALL_FP_PAIRS(INST_COPY_KERNEL)

template <typename value_in_t, typename value_out_t>
__host__ void copy_impl(dim<2> size, StorageVariant& source, StorageVariant& target)
{
    std::visit([&](auto& x, auto& y) {
        using Tx = std::decay_t<decltype(x)>;
        using Ty = std::decay_t<decltype(y)>;
        if constexpr ((std::is_same_v<Tx, DenseStorage>) && (std::is_same_v<Ty, DenseStorage>)) {
            copy_launcher(size, std::get<value_out_t*>(x.values_), x.lead_dim_, std::get<value_in_t*>(y.values_), y.lead_dim_);
        }
        else if constexpr ((std::is_same_v<Tx, CsrStorage>) && (std::is_same_v<Ty, CsrStorage>)) {
            copy_launcher({x.nnz_, 1}, std::get<value_out_t*>(x.values_), x.nnz_, std::get<value_in_t*>(y.values_), x.nnz_);
            copy_launcher({size[0]+1, 1}, std::get<int*>(x.row_ptrs_), size[0]+1, std::get<int*>(y.row_ptrs_), size[0]+1);
            copy_launcher({x.nnz_, 1}, std::get<int*>(x.col_idxs_), x.nnz_, std::get<int*>(y.col_idxs_), x.nnz_);
        }
        else {
            throw std::runtime_error("Copying values from matrixs of different types.");
        }
    }, source, target);
}   // copy_impl

#define INST_COPY_IMPL(T1, T2) \
    template __host__ void copy_impl<T1, T2>(dim<2>, StorageVariant&, StorageVariant&);

FOR_ALL_FP_PAIRS(INST_COPY_IMPL)

}   // namespace cuda
}   // namespace RandLAPACK