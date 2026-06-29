#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstring>
#include <memory>

#include "matrix.hh"

using namespace RandLAPACK;

class TestStorageDispatchGPU : public ::testing::Test {
protected:
    Context gpu_context{Device::CUDA};
    Context cpu_context{Device::CPU};
};

// --- Dense GPU initialization tests ---

TEST_F(TestStorageDispatchGPU, DenseInitFP64_GPU) {
    auto mat = Dense::create_strided(gpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {4, 3});
    EXPECT_EQ(mat.get_format(), MatrixFormat::STRIDED);
    EXPECT_EQ(mat.get_device(), Device::CUDA);
    EXPECT_EQ(mat.get_value_type(), NumericType::FP64);

    dim<2> size = mat.get_size();
    EXPECT_EQ(size.nrows, 4);
    EXPECT_EQ(size.ncols, 3);
}

TEST_F(TestStorageDispatchGPU, DenseInitRowMajor_GPU) {
    auto mat = Dense::create_strided(gpu_context, Layout::ROW_MAJOR,
                                     NumericType::FP64, {3, 5});
    EXPECT_EQ(mat.get_format(), MatrixFormat::STRIDED);
    EXPECT_EQ(mat.get_device(), Device::CUDA);

    dim<2> size = mat.get_size();
    EXPECT_EQ(size.nrows, 3);
    EXPECT_EQ(size.ncols, 5);
}

TEST_F(TestStorageDispatchGPU, DenseInitCustomLeadDim_GPU) {
    size_t ld = 10;
    auto mat = Dense::create_strided(gpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {4, 3}, ld);
    EXPECT_EQ(mat.get_format(), MatrixFormat::STRIDED);
    EXPECT_EQ(mat.get_device(), Device::CUDA);

    dim<2> size = mat.get_size();
    EXPECT_EQ(size.nrows, 4);
    EXPECT_EQ(size.ncols, 3);
}

// --- Dense GPU copy tests ---

TEST_F(TestStorageDispatchGPU, DenseCopySameTypeFP64_GPU) {
    size_t m = 3, n = 2;
    auto src_cpu = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                         NumericType::FP64, {m, n});
    StridedView<double> src_cpu_view(src_cpu);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            src_cpu_view.values_[i + j * m] =
                static_cast<double>(i + j * m + 1);

    auto src_gpu = Dense::create_strided(gpu_context, Layout::COL_MAJOR,
                                         NumericType::FP64, {m, n});
    StridedView<double> src_gpu_view(src_gpu);
    cudaMemcpy(src_gpu_view.values_, src_cpu_view.values_,
               m * n * sizeof(double), cudaMemcpyHostToDevice);

    auto dst_gpu = Dense::create_strided(gpu_context, Layout::COL_MAJOR,
                                         NumericType::FP64, {m, n});
    dst_gpu.copy_from(src_gpu);

    auto dst_cpu = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                         NumericType::FP64, {m, n});
    StridedView<double> dst_gpu_view(dst_gpu);
    StridedView<double> dst_cpu_view(dst_cpu);
    cudaMemcpy(dst_cpu_view.values_, dst_gpu_view.values_,
               m * n * sizeof(double), cudaMemcpyDeviceToHost);

    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            EXPECT_DOUBLE_EQ(dst_cpu_view.values_[i + j * m],
                             src_cpu_view.values_[i + j * m]);
}

TEST_F(TestStorageDispatchGPU, DenseCopyLargerMatrix_GPU) {
    size_t m = 50, n = 30;
    auto src_cpu = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                         NumericType::FP64, {m, n});
    StridedView<double> src_cpu_view(src_cpu);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            src_cpu_view.values_[i + j * m] = static_cast<double>(i * n + j);

    auto src_gpu = Dense::create_strided(gpu_context, Layout::COL_MAJOR,
                                         NumericType::FP64, {m, n});
    StridedView<double> src_gpu_view(src_gpu);
    cudaMemcpy(src_gpu_view.values_, src_cpu_view.values_,
               m * n * sizeof(double), cudaMemcpyHostToDevice);

    auto dst_gpu = Dense::create_strided(gpu_context, Layout::COL_MAJOR,
                                         NumericType::FP64, {m, n});
    dst_gpu.copy_from(src_gpu);

    auto dst_cpu = Dense::create_strided(cpu_context, Layout::COL_MAJOR,
                                         NumericType::FP64, {m, n});
    StridedView<double> dst_gpu_view(dst_gpu);
    StridedView<double> dst_cpu_view(dst_cpu);
    cudaMemcpy(dst_cpu_view.values_, dst_gpu_view.values_,
               m * n * sizeof(double), cudaMemcpyDeviceToHost);

    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            EXPECT_DOUBLE_EQ(dst_cpu_view.values_[i + j * m],
                             src_cpu_view.values_[i + j * m]);
}

TEST_F(TestStorageDispatchGPU, DenseCopySizeMismatchThrows_GPU) {
    auto src = Dense::create_strided(gpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {3, 2});
    auto dst = Dense::create_strided(gpu_context, Layout::COL_MAJOR,
                                     NumericType::FP64, {4, 2});
    EXPECT_THROW(dst.copy_from(src), std::invalid_argument);
}

// --- Dense GPU lifecycle ---

TEST_F(TestStorageDispatchGPU, DenseCreateAndDestroyMultiple_GPU) {
    for (int i = 0; i < 10; ++i) {
        auto mat = Dense::create_strided(gpu_context, Layout::COL_MAJOR,
                                         NumericType::FP64, {8, 8});
    }
}

// --- GPU dispatch lookups ---

TEST_F(TestStorageDispatchGPU, DenseLookupInitializeCUDA_ReturnsNonNull) {
    auto fn = dispatch::Dense::get().lookup_initialize(
        MatrixFormat::STRIDED, Device::CUDA, NumericType::FP64);
    EXPECT_NE(fn, nullptr);
}

TEST_F(TestStorageDispatchGPU, DenseLookupFreeCUDA_ReturnsNonNull) {
    auto fn = dispatch::Dense::get().lookup_free(
        MatrixFormat::STRIDED, Device::CUDA, NumericType::FP64);
    EXPECT_NE(fn, nullptr);
}

TEST_F(TestStorageDispatchGPU, DenseLookupCopyCUDA_ReturnsNonNull) {
    auto fn = dispatch::Dense::get().lookup_copy(
        MatrixFormat::STRIDED, MatrixFormat::STRIDED, Device::CUDA,
        Device::CUDA, NumericType::FP64, NumericType::FP64);
    EXPECT_NE(fn, nullptr);
}

TEST_F(TestStorageDispatchGPU, SparseLookupInitializeCUDA_ReturnsNonNull) {
    auto fn = dispatch::Sparse::get().lookup_initialize(
        MatrixFormat::CSR, Device::CUDA, NumericType::FP64, IntType::INT64);
    EXPECT_NE(fn, nullptr);
}

TEST_F(TestStorageDispatchGPU, SparseLookupFreeCUDA_ReturnsNonNull) {
    auto fn = dispatch::Sparse::get().lookup_free(
        MatrixFormat::CSR, Device::CUDA, NumericType::FP64, IntType::INT64);
    EXPECT_NE(fn, nullptr);
}
