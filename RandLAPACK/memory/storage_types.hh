#pragma once

#include <cstdint>
#include <variant>

#include "dimensions.hh"
#include "precision.hh"

namespace RandLAPACK {

struct StridedStorage {
    dim<2> size_ = {0, 0};
    size_t lead_dim_ = 0;
    size_t num_elems_ = 0;
    Layout layout_ = Layout::COL_MAJOR;
    NumericPtrVariant values_;

    StridedStorage() = default;
    StridedStorage(NumericType store_type, Layout layout, dim<2> size,
                   size_t lead_dim)
        : size_{size}, lead_dim_{lead_dim}, layout_{layout} {
        switch (store_type) {
            case NumericType::FP64: values_ = static_cast<double*>(nullptr); break;
            case NumericType::FP32: values_ = static_cast<float*>(nullptr); break;
            case NumericType::FP16: values_ = static_cast<half*>(nullptr); break;
            default: break;
        }
    }
    StridedStorage(Layout layout, dim<2> size, size_t lead_dim,
                   NumericPtrVariant values)
        : size_{size}, lead_dim_{lead_dim}, layout_{layout}, values_{values} {}
};

struct CsrStorage {
    dim<2> size_ = {0, 0};
    size_t nnz_;
    NumericPtrVariant values_;
    IntPtrVariant row_ptrs_;
    IntPtrVariant col_idxs_;

    CsrStorage() = default;
    CsrStorage(NumericType value_type, IntType index_type, dim<2> size,
               size_t nnz)
        : size_{size}, nnz_{nnz} {
        switch (value_type) {
            case NumericType::FP64: values_ = static_cast<double*>(nullptr); break;
            case NumericType::FP32: values_ = static_cast<float*>(nullptr); break;
            case NumericType::FP16: values_ = static_cast<half*>(nullptr); break;
            default: break;
        }
        switch (index_type) {
            case IntType::INT64:
                row_ptrs_ = static_cast<int64_t*>(nullptr);
                col_idxs_ = static_cast<int64_t*>(nullptr);
                break;
            case IntType::INT32:
                row_ptrs_ = static_cast<int32_t*>(nullptr);
                col_idxs_ = static_cast<int32_t*>(nullptr);
                break;
            default: break;
        }
    }
    CsrStorage(dim<2> size, size_t nnz, NumericPtrVariant values,
               IntPtrVariant row_ptrs, IntPtrVariant col_idxs)
        : size_{size},
          nnz_{nnz},
          values_{values},
          row_ptrs_{row_ptrs},
          col_idxs_{col_idxs} {}
};

enum class DenseFormat : std::size_t { STRIDED = 0, Count };
enum class SparseFormat : std::size_t { CSR = 1, Count };
enum class MatrixFormat : std::size_t { STRIDED = 0, CSR = 1, Count };

using DenseStorage = std::variant<StridedStorage>;
using SparseStorage = std::variant<CsrStorage>;

}  // namespace RandLAPACK
