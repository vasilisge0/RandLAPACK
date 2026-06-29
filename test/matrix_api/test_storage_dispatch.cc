#include <gtest/gtest.h>

#include <cstring>
#include <memory>

#include "matrix.hh"

using namespace RandLAPACK;

class TestStorageDispatch : public ::testing::Test {
protected:
    Context cpu_context{Device::CPU};
};

// --- Dense initialization tests ---

TEST_F(TestStorageDispatch, DenseInitFP64) {
    auto mat = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {4, 3});
    EXPECT_EQ(mat.get_format(), MatrixFormat::STRIDED);
    EXPECT_EQ(mat.get_device(), Device::CPU);
    EXPECT_EQ(mat.get_value_type(), NumericType::FP64);

    dim<2> size = mat.get_size();
    EXPECT_EQ(size.nrows, 4);
    EXPECT_EQ(size.ncols, 3);
}

TEST_F(TestStorageDispatch, DenseInitRowMajor) {
    auto mat = Dense::create_strided(cpu_context, Layout::ROW_MAJOR,
                                     NumericType::FP64, {3, 5});
    EXPECT_EQ(mat.get_format(), MatrixFormat::STRIDED);

    dim<2> size = mat.get_size();
    EXPECT_EQ(size.nrows, 3);
    EXPECT_EQ(size.ncols, 5);
}

TEST_F(TestStorageDispatch, DenseInitCustomLeadDim) {
    size_t ld = 10;
    auto mat = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {4, 3}, ld);
    EXPECT_EQ(mat.get_format(), MatrixFormat::STRIDED);

    dim<2> size = mat.get_size();
    EXPECT_EQ(size.nrows, 4);
    EXPECT_EQ(size.ncols, 3);
}

// --- Dense copy tests ---

TEST_F(TestStorageDispatch, DenseCopySameTypeFP64) {
    auto src = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {3, 2});
    StridedView<double> src_view(src);
    for (size_t j = 0; j < 2; ++j)
        for (size_t i = 0; i < 3; ++i)
            src_view.values_[i + j * 3] = static_cast<double>(i + j * 3 + 1);

    auto dst = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {3, 2});
    dst.copy_from(src);

    StridedView<double> dst_view(dst);
    for (size_t j = 0; j < 2; ++j)
        for (size_t i = 0; i < 3; ++i)
            EXPECT_DOUBLE_EQ(dst_view.values_[i + j * 3],
                             src_view.values_[i + j * 3]);
}

TEST_F(TestStorageDispatch, DenseCopyLargerMatrix) {
    size_t m = 50, n = 30;
    auto src = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {m, n});
    StridedView<double> src_view(src);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            src_view.values_[i + j * m] = static_cast<double>(i * n + j);

    auto dst = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {m, n});
    dst.copy_from(src);

    StridedView<double> dst_view(dst);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            EXPECT_DOUBLE_EQ(dst_view.values_[i + j * m],
                             src_view.values_[i + j * m]);
}

TEST_F(TestStorageDispatch, DenseCopyLargerMatrix2) {
    size_t m = 150, n = 230;
    auto src = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {m, n});
    StridedView<double> src_view(src);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            src_view.values_[i + j * m] = static_cast<double>(i * n + j);

    auto dst = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {m, n});
    dst.copy_from(src);

    StridedView<double> dst_view(dst);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            EXPECT_DOUBLE_EQ(dst_view.values_[i + j * m],
                             src_view.values_[i + j * m]);
}

TEST_F(TestStorageDispatch, DenseCopySizeMismatchThrows) {
    auto src = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {3, 2});
    auto dst = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {4, 2});
    EXPECT_THROW(dst.copy_from(src), std::invalid_argument);
}

// --- Dense StridedView tests ---

TEST_F(TestStorageDispatch, StridedViewAccessFP64) {
    auto mat = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {4, 3});
    StridedView<double> view(mat);
    EXPECT_EQ(view.size_.nrows, 4);
    EXPECT_EQ(view.size_.ncols, 3);
    EXPECT_EQ(view.lead_dim_, 4);
    EXPECT_EQ(view.device_, Device::CPU);
    EXPECT_NE(view.values_, nullptr);
}

TEST_F(TestStorageDispatch, StridedViewWriteAndRead) {
    auto mat = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {2, 2});
    StridedView<double> view(mat);
    view.values_[0] = 1.0;
    view.values_[1] = 2.0;
    view.values_[2] = 3.0;
    view.values_[3] = 4.0;

    StridedView<double> view2(mat);
    EXPECT_DOUBLE_EQ(view2.values_[0], 1.0);
    EXPECT_DOUBLE_EQ(view2.values_[1], 2.0);
    EXPECT_DOUBLE_EQ(view2.values_[2], 3.0);
    EXPECT_DOUBLE_EQ(view2.values_[3], 4.0);
}

TEST_F(TestStorageDispatch, StridedViewWrongTypeThrows) {
    auto mat = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {3, 2});
    EXPECT_THROW(
        {
            StridedView<float> bad_view(mat);
            (void)bad_view;
        },
        std::bad_variant_access);
}

// --- Dispatch singleton tests ---

TEST_F(TestStorageDispatch, DenseDispatchSingletonIsSame) {
    const auto& d1 = dispatch::Dense::get();
    const auto& d2 = dispatch::Dense::get();
    EXPECT_EQ(&d1, &d2);
}

TEST_F(TestStorageDispatch, SparseDispatchSingletonIsSame) {
    const auto& s1 = dispatch::Sparse::get();
    const auto& s2 = dispatch::Sparse::get();
    EXPECT_EQ(&s1, &s2);
}

// --- Valid lookups return non-null function pointers ---

TEST_F(TestStorageDispatch, DenseLookupInitializeReturnsNonNull) {
    auto fn = dispatch::Dense::get().lookup_initialize(
        MatrixFormat::STRIDED, Device::CPU, NumericType::FP64);
    EXPECT_NE(fn, nullptr);
}

TEST_F(TestStorageDispatch, DenseLookupFreeReturnsNonNull) {
    auto fn = dispatch::Dense::get().lookup_free(
        MatrixFormat::STRIDED, Device::CPU, NumericType::FP64);
    EXPECT_NE(fn, nullptr);
}

TEST_F(TestStorageDispatch, DenseLookupCopyReturnsNonNull) {
    auto fn = dispatch::Dense::get().lookup_copy(
        MatrixFormat::STRIDED, MatrixFormat::STRIDED, Device::CPU, Device::CPU,
        NumericType::FP64, NumericType::FP64);
    EXPECT_NE(fn, nullptr);
}

TEST_F(TestStorageDispatch, SparseLookupInitializeReturnsNonNull) {
    auto fn = dispatch::Sparse::get().lookup_initialize(
        MatrixFormat::CSR, Device::CPU, NumericType::FP64, IntType::INT64);
    EXPECT_NE(fn, nullptr);
}

TEST_F(TestStorageDispatch, SparseLookupFreeReturnsNonNull) {
    auto fn = dispatch::Sparse::get().lookup_free(
        MatrixFormat::CSR, Device::CPU, NumericType::FP64, IntType::INT64);
    EXPECT_NE(fn, nullptr);
}

TEST_F(TestStorageDispatch, SparseLookupCopyReturnsNonNull) {
    auto fn = dispatch::Sparse::get().lookup_copy(
        MatrixFormat::CSR, MatrixFormat::CSR, Device::CPU, Device::CPU,
        NumericType::FP64, NumericType::FP64, IntType::INT64, IntType::INT64);
    EXPECT_NE(fn, nullptr);
}

// --- Unsupported lookup throws ---

TEST_F(TestStorageDispatch, SparseLookupUnsupportedFormatThrows) {
    EXPECT_THROW(dispatch::Sparse::get().lookup_initialize(
                     MatrixFormat::STRIDED, Device::CPU, NumericType::FP64,
                     IntType::INT64),
                 std::runtime_error);
}

// --- Dense lifecycle: repeated create/destroy ---

TEST_F(TestStorageDispatch, DenseCreateAndDestroyMultiple) {
    for (int i = 0; i < 10; ++i) {
        auto mat = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                         NumericType::FP64, {8, 8});
        StridedView<double> view(mat);
        for (size_t k = 0; k < 64; ++k)
            view.values_[k] = static_cast<double>(k);
    }
}

// --- Descriptor round-trip ---

TEST_F(TestStorageDispatch, DenseDescriptorRoundTrip) {
    auto mat = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {4, 3});
    MatrixDescriptor desc = mat.get_descriptor();
    EXPECT_EQ(desc.get_format(), MatrixFormat::STRIDED);
}

// --- Sparse initialization tests ---

TEST_F(TestStorageDispatch, SparseInitCSR_FP64_INT64) {
    auto mat = Sparse::create_csr(cpu_context, NumericType::FP64, IntType::INT64,
                                  {4, 5}, 7);
    EXPECT_EQ(mat.get_format(), MatrixFormat::CSR);
    EXPECT_EQ(mat.get_device(), Device::CPU);
    EXPECT_EQ(mat.get_value_type(), NumericType::FP64);
    EXPECT_EQ(mat.get_index_type(), IntType::INT64);

    dim<2> size = mat.get_size();
    EXPECT_EQ(size.nrows, 4);
    EXPECT_EQ(size.ncols, 5);
}

TEST_F(TestStorageDispatch, SparseInitCSR_FP32_INT32) {
    auto mat = Sparse::create_csr(cpu_context, NumericType::FP32, IntType::INT32,
                                  {3, 6}, 5);
    EXPECT_EQ(mat.get_format(), MatrixFormat::CSR);
    EXPECT_EQ(mat.get_value_type(), NumericType::FP32);
    EXPECT_EQ(mat.get_index_type(), IntType::INT32);

    dim<2> size = mat.get_size();
    EXPECT_EQ(size.nrows, 3);
    EXPECT_EQ(size.ncols, 6);
}

TEST_F(TestStorageDispatch, SparseInitCSR_FP64_INT32) {
    auto mat = Sparse::create_csr(cpu_context, NumericType::FP64, IntType::INT32,
                                  {10, 8}, 20);
    EXPECT_EQ(mat.get_value_type(), NumericType::FP64);
    EXPECT_EQ(mat.get_index_type(), IntType::INT32);
}

// --- Sparse CsrView tests ---

TEST_F(TestStorageDispatch, CsrViewAccessFP64_INT64) {
    size_t nrows = 3, ncols = 4, nnz = 5;
    auto mat = Sparse::create_csr(cpu_context, NumericType::FP64, IntType::INT64,
                                  {nrows, ncols}, nnz);

    CsrView<double, int64_t> view(mat);
    EXPECT_EQ(view.size_.nrows, nrows);
    EXPECT_EQ(view.size_.ncols, ncols);
    EXPECT_EQ(view.nnz_, nnz);
    EXPECT_EQ(view.device_, Device::CPU);
    EXPECT_NE(view.values_, nullptr);
    EXPECT_NE(view.row_ptrs_, nullptr);
    EXPECT_NE(view.col_idxs_, nullptr);
}

TEST_F(TestStorageDispatch, CsrViewWriteAndRead) {
    size_t nrows = 3, ncols = 4, nnz = 5;
    auto mat = Sparse::create_csr(cpu_context, NumericType::FP64, IntType::INT64,
                                  {nrows, ncols}, nnz);

    CsrView<double, int64_t> view(mat);
    for (size_t i = 0; i < nnz; ++i)
        view.values_[i] = static_cast<double>(i + 1);
    view.row_ptrs_[0] = 0;
    view.row_ptrs_[1] = 2;
    view.row_ptrs_[2] = 3;
    view.row_ptrs_[3] = 5;
    view.col_idxs_[0] = 0;
    view.col_idxs_[1] = 1;
    view.col_idxs_[2] = 2;
    view.col_idxs_[3] = 1;
    view.col_idxs_[4] = 3;

    CsrView<double, int64_t> view2(mat);
    for (size_t i = 0; i < nnz; ++i)
        EXPECT_DOUBLE_EQ(view2.values_[i], static_cast<double>(i + 1));
    EXPECT_EQ(view2.row_ptrs_[0], 0);
    EXPECT_EQ(view2.row_ptrs_[1], 2);
    EXPECT_EQ(view2.row_ptrs_[2], 3);
    EXPECT_EQ(view2.row_ptrs_[3], 5);
}

TEST_F(TestStorageDispatch, CsrViewWrongValueTypeThrows) {
    auto mat = Sparse::create_csr(cpu_context, NumericType::FP64, IntType::INT64,
                                  {3, 4}, 5);
    using WrongValueView = CsrView<float, int64_t>;
    EXPECT_THROW(
        { WrongValueView view(mat); (void)view; },
        std::bad_variant_access);
}

TEST_F(TestStorageDispatch, CsrViewWrongIndexTypeThrows) {
    auto mat = Sparse::create_csr(cpu_context, NumericType::FP64, IntType::INT64,
                                  {3, 4}, 5);
    using WrongIndexView = CsrView<double, int32_t>;
    EXPECT_THROW(
        { WrongIndexView view(mat); (void)view; },
        std::bad_variant_access);
}

// --- Sparse copy tests ---

TEST_F(TestStorageDispatch, SparseCopySameType_FP64_INT64) {
    size_t nrows = 3, ncols = 4, nnz = 5;
    auto src = Sparse::create_csr(cpu_context, NumericType::FP64, IntType::INT64,
                                  {nrows, ncols}, nnz);

    CsrView<double, int64_t> src_view(src);
    for (size_t i = 0; i < nnz; ++i)
        src_view.values_[i] = static_cast<double>(i + 1);
    src_view.row_ptrs_[0] = 0;
    src_view.row_ptrs_[1] = 2;
    src_view.row_ptrs_[2] = 3;
    src_view.row_ptrs_[3] = 5;
    src_view.col_idxs_[0] = 0;
    src_view.col_idxs_[1] = 1;
    src_view.col_idxs_[2] = 2;
    src_view.col_idxs_[3] = 1;
    src_view.col_idxs_[4] = 3;

    auto dst = Sparse::create_csr(cpu_context, NumericType::FP64, IntType::INT64,
                                  {nrows, ncols}, nnz);
    dst.copy_from(src);

    CsrView<double, int64_t> dst_view(dst);
    for (size_t i = 0; i < nnz; ++i)
        EXPECT_DOUBLE_EQ(dst_view.values_[i], src_view.values_[i]);
    for (size_t i = 0; i <= nrows; ++i)
        EXPECT_EQ(dst_view.row_ptrs_[i], src_view.row_ptrs_[i]);
    for (size_t i = 0; i < nnz; ++i)
        EXPECT_EQ(dst_view.col_idxs_[i], src_view.col_idxs_[i]);
}

TEST_F(TestStorageDispatch, SparseCopySizeMismatchThrows) {
    auto src = Sparse::create_csr(cpu_context, NumericType::FP64, IntType::INT64,
                                  {3, 4}, 5);
    auto dst = Sparse::create_csr(cpu_context, NumericType::FP64, IntType::INT64,
                                  {4, 4}, 5);
    EXPECT_THROW(dst.copy_from(src), std::invalid_argument);
}

// --- Sparse lifecycle ---

TEST_F(TestStorageDispatch, SparseCreateAndDestroyMultiple) {
    for (int i = 0; i < 10; ++i) {
        auto mat = Sparse::create_csr(cpu_context, NumericType::FP64,
                                      IntType::INT64, {5, 5}, 10);
    }
}

// --- Sparse descriptor round-trip ---

TEST_F(TestStorageDispatch, SparseDescriptorRoundTrip) {
    auto mat = Sparse::create_csr(cpu_context, NumericType::FP32, IntType::INT32,
                                  {3, 5}, 8);
    MatrixDescriptor desc = mat.get_descriptor();
    EXPECT_EQ(desc.get_format(), MatrixFormat::CSR);
}

// --- Dense cross-type copy tests ---

TEST_F(TestStorageDispatch, DenseCopyFP64toFP32) {
    auto src = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {3, 2});
    StridedView<double> src_view(src);
    for (size_t j = 0; j < 2; ++j)
        for (size_t i = 0; i < 3; ++i)
            src_view.values_[i + j * 3] = static_cast<double>(i + j * 3) + 0.5;

    auto dst = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP32, {3, 2});
    dst.copy_from(src);

    StridedView<float> dst_view(dst);
    for (size_t j = 0; j < 2; ++j)
        for (size_t i = 0; i < 3; ++i)
            EXPECT_NEAR(static_cast<double>(dst_view.values_[i + j * 3]),
                        src_view.values_[i + j * 3], 1e-6);
}

TEST_F(TestStorageDispatch, DenseCopyFP32toFP64) {
    auto src = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP32, {2, 3});
    StridedView<float> src_view(src);
    for (size_t j = 0; j < 3; ++j)
        for (size_t i = 0; i < 2; ++i)
            src_view.values_[i + j * 2] = static_cast<float>(i + j * 2) + 0.25f;

    auto dst = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {2, 3});
    dst.copy_from(src);

    StridedView<double> dst_view(dst);
    for (size_t j = 0; j < 3; ++j)
        for (size_t i = 0; i < 2; ++i)
            EXPECT_NEAR(dst_view.values_[i + j * 2],
                        static_cast<double>(src_view.values_[i + j * 2]), 1e-6);
}

TEST_F(TestStorageDispatch, DenseCopyFP32toFP32) {
    auto src = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP32, {4, 2});
    StridedView<float> src_view(src);
    for (size_t j = 0; j < 2; ++j)
        for (size_t i = 0; i < 4; ++i)
            src_view.values_[i + j * 4] = static_cast<float>(i + j * 4 + 1);

    auto dst = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP32, {4, 2});
    dst.copy_from(src);

    StridedView<float> dst_view(dst);
    for (size_t j = 0; j < 2; ++j)
        for (size_t i = 0; i < 4; ++i)
            EXPECT_FLOAT_EQ(dst_view.values_[i + j * 4],
                            src_view.values_[i + j * 4]);
}

// --- Dense create_from tests ---

TEST_F(TestStorageDispatch, DenseCreateFromCopiesData) {
    auto src = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {3, 2});
    StridedView<double> src_view(src);
    for (size_t k = 0; k < 6; ++k)
        src_view.values_[k] = static_cast<double>(k + 1);

    auto dst = Dense::create_from(src);

    StridedView<double> dst_view(dst);
    for (size_t k = 0; k < 6; ++k)
        EXPECT_DOUBLE_EQ(dst_view.values_[k], src_view.values_[k]);
}

TEST_F(TestStorageDispatch, DenseCreateFromWithContext) {
    auto src = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {4, 3});
    StridedView<double> src_view(src);
    for (size_t k = 0; k < 12; ++k)
        src_view.values_[k] = static_cast<double>(k * 2.5);

    auto dst = Dense::create_from(cpu_context, src);

    StridedView<double> dst_view(dst);
    for (size_t k = 0; k < 12; ++k)
        EXPECT_DOUBLE_EQ(dst_view.values_[k], src_view.values_[k]);
    EXPECT_EQ(dst.get_device(), Device::CPU);
}

// --- Sparse create_from tests ---

TEST_F(TestStorageDispatch, SparseCreateFromCopiesData) {
    size_t nrows = 3, ncols = 4, nnz = 5;
    auto src = Sparse::create_csr(cpu_context, NumericType::FP64, IntType::INT64,
                                  {nrows, ncols}, nnz);

    CsrView<double, int64_t> src_view(src);
    for (size_t i = 0; i < nnz; ++i)
        src_view.values_[i] = static_cast<double>(i + 1);
    src_view.row_ptrs_[0] = 0;
    src_view.row_ptrs_[1] = 2;
    src_view.row_ptrs_[2] = 3;
    src_view.row_ptrs_[3] = 5;
    src_view.col_idxs_[0] = 0;
    src_view.col_idxs_[1] = 1;
    src_view.col_idxs_[2] = 2;
    src_view.col_idxs_[3] = 1;
    src_view.col_idxs_[4] = 3;

    auto dst = Sparse::create_from(cpu_context, src);

    CsrView<double, int64_t> dst_view(dst);
    for (size_t i = 0; i < nnz; ++i)
        EXPECT_DOUBLE_EQ(dst_view.values_[i], src_view.values_[i]);
    for (size_t i = 0; i <= nrows; ++i)
        EXPECT_EQ(dst_view.row_ptrs_[i], src_view.row_ptrs_[i]);
    for (size_t i = 0; i < nnz; ++i)
        EXPECT_EQ(dst_view.col_idxs_[i], src_view.col_idxs_[i]);
}

// --- Move semantics tests ---

TEST_F(TestStorageDispatch, DenseMoveConstructor) {
    auto src = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {3, 2});
    StridedView<double> src_view(src);
    for (size_t k = 0; k < 6; ++k)
        src_view.values_[k] = static_cast<double>(k + 1);

    Dense dst(std::move(src));

    StridedView<double> dst_view(dst);
    for (size_t k = 0; k < 6; ++k)
        EXPECT_DOUBLE_EQ(dst_view.values_[k], static_cast<double>(k + 1));
}

TEST_F(TestStorageDispatch, DenseMoveAssignment) {
    auto src = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {3, 2});
    StridedView<double> src_view(src);
    for (size_t k = 0; k < 6; ++k)
        src_view.values_[k] = static_cast<double>(k + 10);

    Dense dst;
    dst = std::move(src);

    StridedView<double> dst_view(dst);
    for (size_t k = 0; k < 6; ++k)
        EXPECT_DOUBLE_EQ(dst_view.values_[k], static_cast<double>(k + 10));
}

TEST_F(TestStorageDispatch, DenseMoveAssignmentFreesOldMemory) {
    auto dst = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {4, 4});
    auto src = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {2, 2});
    StridedView<double> src_view(src);
    src_view.values_[0] = 42.0;

    dst = std::move(src);

    StridedView<double> dst_view(dst);
    EXPECT_DOUBLE_EQ(dst_view.values_[0], 42.0);
    dim<2> size = dst.get_size();
    EXPECT_EQ(size.nrows, 2);
    EXPECT_EQ(size.ncols, 2);
}

TEST_F(TestStorageDispatch, SparseMoveConstructor) {
    size_t nrows = 3, ncols = 4, nnz = 5;
    auto src = Sparse::create_csr(cpu_context, NumericType::FP64, IntType::INT64,
                                  {nrows, ncols}, nnz);

    CsrView<double, int64_t> src_view(src);
    for (size_t i = 0; i < nnz; ++i)
        src_view.values_[i] = static_cast<double>(i + 1);

    Sparse dst(std::move(src));

    CsrView<double, int64_t> dst_view(dst);
    for (size_t i = 0; i < nnz; ++i)
        EXPECT_DOUBLE_EQ(dst_view.values_[i], static_cast<double>(i + 1));
}

TEST_F(TestStorageDispatch, SparseMoveAssignment) {
    size_t nrows = 3, ncols = 4, nnz = 5;
    auto src = Sparse::create_csr(cpu_context, NumericType::FP64, IntType::INT64,
                                  {nrows, ncols}, nnz);

    CsrView<double, int64_t> src_view(src);
    for (size_t i = 0; i < nnz; ++i)
        src_view.values_[i] = static_cast<double>(i + 10);

    Sparse dst;
    dst = std::move(src);

    CsrView<double, int64_t> dst_view(dst);
    for (size_t i = 0; i < nnz; ++i)
        EXPECT_DOUBLE_EQ(dst_view.values_[i], static_cast<double>(i + 10));
}

// --- Sparse cross-type copy tests ---

TEST_F(TestStorageDispatch, SparseCopySameType_FP32_INT32) {
    size_t nrows = 3, ncols = 4, nnz = 5;
    auto src = Sparse::create_csr(cpu_context, NumericType::FP32, IntType::INT32,
                                  {nrows, ncols}, nnz);

    CsrView<float, int32_t> src_view(src);
    for (size_t i = 0; i < nnz; ++i)
        src_view.values_[i] = static_cast<float>(i + 1);
    src_view.row_ptrs_[0] = 0;
    src_view.row_ptrs_[1] = 2;
    src_view.row_ptrs_[2] = 3;
    src_view.row_ptrs_[3] = 5;
    src_view.col_idxs_[0] = 0;
    src_view.col_idxs_[1] = 1;
    src_view.col_idxs_[2] = 2;
    src_view.col_idxs_[3] = 1;
    src_view.col_idxs_[4] = 3;

    auto dst = Sparse::create_csr(cpu_context, NumericType::FP32, IntType::INT32,
                                  {nrows, ncols}, nnz);
    dst.copy_from(src);

    CsrView<float, int32_t> dst_view(dst);
    for (size_t i = 0; i < nnz; ++i)
        EXPECT_FLOAT_EQ(dst_view.values_[i], src_view.values_[i]);
    for (size_t i = 0; i <= nrows; ++i)
        EXPECT_EQ(dst_view.row_ptrs_[i], src_view.row_ptrs_[i]);
    for (size_t i = 0; i < nnz; ++i)
        EXPECT_EQ(dst_view.col_idxs_[i], src_view.col_idxs_[i]);
}
