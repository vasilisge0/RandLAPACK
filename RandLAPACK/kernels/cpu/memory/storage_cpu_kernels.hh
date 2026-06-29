#pragma once

#include "dimensions.hh"
#include "memory.hh"
#include "storage_types.hh"

namespace RandLAPACK {
namespace cpu {

template <typename value_in_t, typename value_out_t>
void strided_copy_kernel(dim<2> size, value_in_t* values_in, size_t ld_in,
                         value_out_t* values_out, size_t ld_out) {
    size_t nrows = size[0];
    size_t ncols = size[1];
    if constexpr (std::is_same<value_in_t, value_out_t>::value) {
        for (size_t j = 0; j < ncols; ++j) {
            for (size_t i = 0; i < nrows; ++i) {
                values_out[i + j * ld_out] = values_in[i + j * ld_in];
            }
        }
    } else {
        for (size_t j = 0; j < ncols; ++j) {
            for (size_t i = 0; i < nrows; ++i) {
                values_out[i + j * ld_out] =
                    static_cast<value_out_t>(values_in[i + j * ld_in]);
            }
        }
    }
}

template <typename value_in_t, typename value_out_t, typename index_in_t,
          typename index_out_t>
__host__ void csr_copy_impl(dim<2> size, SparseStorage& source,
                            SparseStorage& target) {
    CsrStorage& x = std::get<CsrStorage>(source);
    CsrStorage& y = std::get<CsrStorage>(target);
    strided_copy_kernel({x.nnz_, static_cast<size_t>(1)},
                        std::get<value_in_t*>(x.values_), x.nnz_,
                        std::get<value_out_t*>(y.values_), x.nnz_);
    strided_copy_kernel({size[0] + 1, static_cast<size_t>(1)},
                        std::get<index_in_t*>(x.row_ptrs_), size[0] + 1,
                        std::get<index_out_t*>(y.row_ptrs_), size[0] + 1);
    strided_copy_kernel({x.nnz_, static_cast<size_t>(1)},
                        std::get<index_in_t*>(x.col_idxs_), x.nnz_,
                        std::get<index_out_t*>(y.col_idxs_), x.nnz_);
}

}  // namespace cpu
}  // namespace RandLAPACK