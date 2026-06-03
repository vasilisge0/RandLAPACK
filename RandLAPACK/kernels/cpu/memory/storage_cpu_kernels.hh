#pragma once

#include "dimensions.hh"

namespace RandLAPACK {
namespace cpu {

template <typename value_in_t, typename value_out_t>
void copy_impl(dim<2> size, value_in_t* values_in, size_t ld_in, value_out_t* values_out, size_t ld_out)
{
    size_t nrows = size[0];
    size_t ncols = size[1];
    if constexpr (std::is_same<value_in_t, value_out_t>::value) {
        for (size_t j = 0; j < ncols; ++j) {
            for (size_t i = 0; i < nrows; ++i) {
                values_out[i + j * ld_out] = values_in[i + j * ld_in];
            }
        }
    }
    else {
        for (size_t j = 0; j < ncols; ++j) {
            for (size_t i = 0; i < nrows; ++i) {
                values_out[i + j * ld_out] = static_cast<value_out_t>(values_in[i + j * ld_in]);
            }
        }
    }
}   // end of copy_kernel()

}  // namespace cpu
}  // namespace RandLAPACK