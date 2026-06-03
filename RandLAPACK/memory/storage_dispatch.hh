#pragma once

#include <array>

#include "precision.hh"
#include "device.hh"
#include "dimensions.hh"
#include "memory.hh"
#include "dense_storage.hh"
#include "csr_storage.hh"

namespace RandLAPACK {
namespace detail {

using initialize_dense_fn = void (*)(dim<2> size, DenseStorage& source);
using free_dense_fn = void (*)(DenseStorage& source);
using copy_dense_fn = void (*)(dim<2> size, DenseStorage& source, DenseStorage& target);
using initialize_sparse_fn = void (*)(dim<2> size, SparseStorage& source);
using free_sparse_fn = void (*)(SparseStorage& source);
using copy_sparse_fn = void (*)(dim<2> size, SparseStorage& source, SparseStorage& target);

struct StorageRegistry {
  public:
    static const StorageRegistry& get() {
        static const StorageRegistry instance;
        return instance;
    }

    initialize_dense_fn lookup_initialize_dense(MatrixFormat format, Device dev, NumericType numeric_type) const {
        initialize_dense_fn fn = dense_initialize_table_[encode_init_free_dense(format, dev, numeric_type)];
        if (!fn)
            throw std::runtime_error("Unsupported storage type.");
        return fn;
    }

    free_dense_fn lookup_free_dense(MatrixFormat format, Device dev, NumericType numeric_type) const {
        free_dense_fn fn = dense_free_table_[encode_init_free_dense(format, dev, numeric_type)];
        if (!fn)
            throw std::runtime_error("Unsupported storage type.");
        return fn;
    }

    copy_dense_fn lookup_copy_dense(MatrixFormat format_source, MatrixFormat format_target, Device source_dev, Device target_dev, NumericType numeric_type_source, NumericType numeric_type_target) const {
        copy_dense_fn fn = dense_copy_table_[encode_copy_dense(format_source, format_target, source_dev, target_dev, numeric_type_source, numeric_type_target)];
        if (!fn)
            throw std::runtime_error("Unsupported storage type.");
        return fn;
    }

    initialize_sparse_fn lookup_sparse_initialize(MatrixFormat format_target, Device dev, NumericType numeric_type, IntType index_type) const {
        initialize_sparse_fn fn = sparse_initialize_table_[encode_init_free_csr(format_target, dev, numeric_type, index_type)];
        if (!fn)
            throw std::runtime_error("Unsupported storage type.");
        return fn;
    }

    free_sparse_fn lookup_sparse_free(MatrixFormat format_target, Device dev, NumericType numeric_type, IntType index_type) const {
        free_sparse_fn fn = sparse_free_table_[encode_init_free_csr(format_target, dev, numeric_type, index_type)];
        if (!fn)
            throw std::runtime_error("Unsupported storage type.");
        return fn;
    }

    copy_sparse_fn lookup_copy_sparse(MatrixFormat source_format, MatrixFormat target_format, Device source_dev, Device target_dev, NumericType source_numeric_type, NumericType target_numeric_type, IntType source_index_type, IntType target_index_type) const {
        copy_sparse_fn fn = sparse_copy_table_[encode_copy_sparse(source_format, target_format, source_dev, target_dev, source_numeric_type, target_numeric_type, source_index_type, target_index_type)];
        if (!fn)
            throw std::runtime_error("Unsupported storage type.");
        return fn;
    }

  private:
    StorageRegistry() {
        dense_initialize_table_.fill(nullptr);
        dense_free_table_.fill(nullptr);
        sparse_initialize_table_.fill(nullptr);
        sparse_free_table_.fill(nullptr);
        sparse_copy_table_.fill(nullptr);

        struct DenseInitEntries {
            MatrixFormat format;
            Device dev;
            NumericType numeric_type;
            initialize_dense_fn fn;
        };
        static const DenseInitEntries dense_init_entries[] = {
            {MatrixFormat::STRIDED,  Device::CPU, NumericType::FP64,  &initialize_strided_storage<Device::CPU, double>},
            {MatrixFormat::STRIDED,  Device::CPU, NumericType::FP32,  &initialize_strided_storage<Device::CPU,  float>},
            {MatrixFormat::STRIDED,  Device::CPU, NumericType::FP16,  &initialize_strided_storage<Device::CPU,   half>},
            {MatrixFormat::STRIDED, Device::CUDA, NumericType::FP64, &initialize_strided_storage<Device::CUDA, double>},
            {MatrixFormat::STRIDED, Device::CUDA, NumericType::FP32, &initialize_strided_storage<Device::CUDA,  float>},
            {MatrixFormat::STRIDED, Device::CUDA, NumericType::FP16, &initialize_strided_storage<Device::CUDA,   half>},
        };
        for (auto& e : dense_init_entries)
            dense_initialize_table_[encode_init_free_dense(e.format, e.dev, e.numeric_type)] = e.fn;

        struct DenseFreeEntries {
            MatrixFormat format;
            Device dev;
            NumericType numeric_type;
            free_dense_fn fn;
        };
        static const DenseFreeEntries dense_free_entries[] = {
            {MatrixFormat::STRIDED,  Device::CPU, NumericType::FP64,  &free_strided_storage<Device::CPU, double>},
            {MatrixFormat::STRIDED,  Device::CPU, NumericType::FP32,  &free_strided_storage<Device::CPU,  float>},
            {MatrixFormat::STRIDED,  Device::CPU, NumericType::FP16,  &free_strided_storage<Device::CPU,   half>},
            {MatrixFormat::STRIDED, Device::CUDA, NumericType::FP64, &free_strided_storage<Device::CUDA, double>},
            {MatrixFormat::STRIDED, Device::CUDA, NumericType::FP32, &free_strided_storage<Device::CUDA,  float>},
            {MatrixFormat::STRIDED, Device::CUDA, NumericType::FP16, &free_strided_storage<Device::CUDA,   half>},
        };
        for (auto& e : dense_free_entries)
            dense_free_table_[encode_init_free_dense(e.format, e.dev, e.numeric_type)] = e.fn;

        struct DenseCopyEntries {
            MatrixFormat source_format;
            MatrixFormat target_format;
            Device source_dev;
            Device target_dev;
            NumericType source_type;
            NumericType target_type;
            copy_dense_fn fn;
        };
        static const DenseCopyEntries dense_copy_entries[] = {
            {MatrixFormat::STRIDED, MatrixFormat::STRIDED, Device::CPU, Device::CPU, NumericType::FP64, NumericType::FP64, &copy_strided_storage<Device::CPU, Device::CPU, double, double>},
            {MatrixFormat::STRIDED, MatrixFormat::STRIDED, Device::CPU, Device::CPU, NumericType::FP64, NumericType::FP64, &copy_strided_storage<Device::CPU, Device::CPU, double,  float>},
            {MatrixFormat::STRIDED, MatrixFormat::STRIDED, Device::CPU, Device::CPU, NumericType::FP64, NumericType::FP64, &copy_strided_storage<Device::CPU, Device::CPU, double,   half>},
            {MatrixFormat::STRIDED, MatrixFormat::STRIDED, Device::CPU, Device::CPU, NumericType::FP32, NumericType::FP64, &copy_strided_storage<Device::CPU, Device::CPU,  float, double>},
            {MatrixFormat::STRIDED, MatrixFormat::STRIDED, Device::CPU, Device::CPU, NumericType::FP32, NumericType::FP32, &copy_strided_storage<Device::CPU, Device::CPU,  float,  float>},
            {MatrixFormat::STRIDED, MatrixFormat::STRIDED, Device::CPU, Device::CPU, NumericType::FP32, NumericType::FP32, &copy_strided_storage<Device::CPU, Device::CPU,  float,   half>},
            {MatrixFormat::STRIDED, MatrixFormat::STRIDED, Device::CPU, Device::CPU, NumericType::FP16, NumericType::FP16, &copy_strided_storage<Device::CPU, Device::CPU,   half, double>},
            {MatrixFormat::STRIDED, MatrixFormat::STRIDED, Device::CPU, Device::CPU, NumericType::FP16, NumericType::FP16, &copy_strided_storage<Device::CPU, Device::CPU,   half,  float>},
            {MatrixFormat::STRIDED, MatrixFormat::STRIDED, Device::CPU, Device::CPU, NumericType::FP16, NumericType::FP16, &copy_strided_storage<Device::CPU, Device::CPU,   half,   half>},
        };
        for (auto& e : dense_copy_entries)
            dense_copy_table_[encode_copy_dense(e.source_format, e.target_format, e.source_dev, e.target_dev, e.source_type, e.target_type)] = e.fn;

        sparse_initialize_table_.fill(nullptr);
        sparse_free_table_.fill(nullptr);

        struct SparseInitEntries {
            MatrixFormat format;
            Device dev;
            NumericType numeric_type;
            IntType index_type;
            initialize_sparse_fn fn;
        };
        static const SparseInitEntries sparse_init_entries[] = {
            {MatrixFormat::CSR,  Device::CPU, NumericType::FP64, IntType::INT64,  &initialize_csr_storage<Device::CPU, double, int64_t>},
            {MatrixFormat::CSR,  Device::CPU, NumericType::FP32, IntType::INT64,  &initialize_csr_storage<Device::CPU,  float, int64_t>},
            {MatrixFormat::CSR,  Device::CPU, NumericType::FP16, IntType::INT64,  &initialize_csr_storage<Device::CPU,   half, int64_t>},
            {MatrixFormat::CSR,  Device::CPU, NumericType::FP64, IntType::INT32,  &initialize_csr_storage<Device::CPU, double, int32_t>},
            {MatrixFormat::CSR,  Device::CPU, NumericType::FP32, IntType::INT32,  &initialize_csr_storage<Device::CPU,  float, int32_t>},
            {MatrixFormat::CSR,  Device::CPU, NumericType::FP16, IntType::INT32,  &initialize_csr_storage<Device::CPU,   half, int32_t>},
            {MatrixFormat::CSR, Device::CUDA, NumericType::FP64, IntType::INT64, &initialize_csr_storage<Device::CUDA, double, int64_t>},
            {MatrixFormat::CSR, Device::CUDA, NumericType::FP32, IntType::INT64, &initialize_csr_storage<Device::CUDA,  float, int64_t>},
            {MatrixFormat::CSR, Device::CUDA, NumericType::FP16, IntType::INT64, &initialize_csr_storage<Device::CUDA,   half, int64_t>},
            {MatrixFormat::CSR, Device::CUDA, NumericType::FP64, IntType::INT32, &initialize_csr_storage<Device::CUDA, double, int32_t>},
            {MatrixFormat::CSR, Device::CUDA, NumericType::FP32, IntType::INT32, &initialize_csr_storage<Device::CUDA,  float, int32_t>},
            {MatrixFormat::CSR, Device::CUDA, NumericType::FP16, IntType::INT32, &initialize_csr_storage<Device::CUDA,   half, int32_t>},
        };
        for (auto& e : sparse_init_entries)
            sparse_initialize_table_[encode_init_free_csr(e.format, e.dev, e.numeric_type, e.index_type)] = e.fn;

        struct SparseFreeEntries {
            MatrixFormat format;
            Device dev;
            NumericType numeric_type;
            IntType index_type;
            free_sparse_fn fn;
        };
        static const SparseFreeEntries sparse_free_entries[] = {
            {MatrixFormat::CSR,  Device::CPU, NumericType::FP64, IntType::INT64,  &free_csr_storage<Device::CPU, double, int64_t>},
            // {MatrixFormat::CSR,  Device::CPU, NumericType::FP32, IntType::INT64,  &free_csr_storage<Device::CPU,  float, int64_t>},
            // {MatrixFormat::CSR,  Device::CPU, NumericType::FP16, IntType::INT64,  &free_csr_storage<Device::CPU,   half, int64_t>},
            // {MatrixFormat::CSR,  Device::CPU, NumericType::FP64, IntType::INT32,  &free_csr_storage<Device::CPU, double, int32_t>},
            // {MatrixFormat::CSR,  Device::CPU, NumericType::FP32, IntType::INT32,  &free_csr_storage<Device::CPU,  float, int32_t>},
            // {MatrixFormat::CSR,  Device::CPU, NumericType::FP16, IntType::INT32,  &free_csr_storage<Device::CPU,   half, int32_t>},
            // {MatrixFormat::CSR, Device::CUDA, NumericType::FP64, IntType::INT64, &free_csr_storage<Device::CUDA, double, int64_t>},
            // {MatrixFormat::CSR, Device::CUDA, NumericType::FP32, IntType::INT64, &free_csr_storage<Device::CUDA,  float, int64_t>},
            // {MatrixFormat::CSR, Device::CUDA, NumericType::FP16, IntType::INT64, &free_csr_storage<Device::CUDA,   half, int64_t>},
            // {MatrixFormat::CSR, Device::CUDA, NumericType::FP64, IntType::INT32, &free_csr_storage<Device::CUDA, double, int32_t>},
            // {MatrixFormat::CSR, Device::CUDA, NumericType::FP32, IntType::INT32, &free_csr_storage<Device::CUDA,  float, int32_t>},
            // {MatrixFormat::CSR, Device::CUDA, NumericType::FP16, IntType::INT32, &free_csr_storage<Device::CUDA,   half, int32_t>},
        };
        for (auto& e : sparse_free_entries)
            sparse_free_table_[encode_init_free_csr(e.format, e.dev, e.numeric_type, e.index_type)] = e.fn;

        struct SparseCopyEntries {
            MatrixFormat source_format;
            MatrixFormat target_format;
            Device source_dev;
            Device target_dev;
            NumericType source_numeric_type;
            NumericType target_numeric_type;
            IntType source_index_type;
            IntType target_index_type;
            copy_sparse_fn fn;
        };
        static const SparseCopyEntries sparse_copy_entries[] = {
            // {MatrixFormat::CSR, MatrixFormat::CSR, Device::CUDA, Device::CPU, NumericType::FP64, NumericType::FP64, IntType::INT64, IntType::INT64, &copy_csr_storage<Device::CUDA, Device::CPU, double, double, int64
        };
        for (auto& e : sparse_copy_entries)
            sparse_copy_table_[encode_copy_sparse(e.source_format, e.target_format, e.source_dev, e.target_dev, e.source_numeric_type, e.target_numeric_type, e.source_index_type, e.target_index_type)] = e.fn;
    }

    // static constexpr std::size_t encode_init_free_dense(Device dev, NumericType numeric_type) {
    //     return (static_cast<std::size_t>(dev)) * num_numeric_types_ + static_cast<std::size_t>(numeric_type);
    // }

    static constexpr std::size_t encode_init_free_csr(MatrixFormat format, Device dev, NumericType numeric_type, IntType index_type) {
        return (static_cast<std::size_t>(format)) * num_devices_ * num_numeric_types_ * num_index_types_ +
               (static_cast<std::size_t>(dev)) * num_numeric_types_ * num_index_types_ +
               (static_cast<std::size_t>(numeric_type)) * num_index_types_ + static_cast<std::size_t>(index_type);
    }

    static constexpr std::size_t encode_init_free_dense(MatrixFormat format, Device dev, NumericType numeric_type) {
        return (static_cast<std::size_t>(format)) * num_devices_ * num_numeric_types_ + (static_cast<std::size_t>(dev)) * num_numeric_types_ + static_cast<std::size_t>(numeric_type);
    }

    static constexpr std::size_t encode_copy_dense(MatrixFormat source_format, MatrixFormat target_format, Device source_dev, Device target_dev, NumericType source_type, NumericType target_type) {
        return (static_cast<std::size_t>(source_format)) * num_devices_ * num_devices_ * num_numeric_types_ * num_numeric_types_ +
               (static_cast<std::size_t>(target_format)) * num_devices_ * num_numeric_types_ * num_numeric_types_ +
               (static_cast<std::size_t>(source_dev)) * num_numeric_types_ * num_numeric_types_ +
               (static_cast<std::size_t>(target_dev)) * num_numeric_types_ * num_numeric_types_ +
               (static_cast<std::size_t>(source_type)) * num_numeric_types_ +
               (static_cast<std::size_t>(target_type));
    }

    static constexpr std::size_t encode_copy_sparse(MatrixFormat source_format, MatrixFormat target_format, Device source_dev, Device target_dev, NumericType source_numeric_type, NumericType target_numeric_type, IntType source_index_type, IntType target_index_type) {
        return (static_cast<std::size_t>(source_format)) * num_devices_ * num_devices_ * num_numeric_types_ * num_numeric_types_ * num_index_types_ * num_index_types_ +
               (static_cast<std::size_t>(target_format)) * num_devices_ * num_numeric_types_ * num_numeric_types_ * num_index_types_ * num_index_types_ +
               (static_cast<std::size_t>(source_dev)) * num_numeric_types_ * num_numeric_types_ * num_index_types_ * num_index_types_ +
               (static_cast<std::size_t>(target_dev)) * num_numeric_types_ * num_numeric_types_ * num_index_types_ * num_index_types_ +
               (static_cast<std::size_t>(source_numeric_type)) * num_numeric_types_ * num_index_types_ * num_index_types_ +
               (static_cast<std::size_t>(target_numeric_type)) * num_index_types_ * num_index_types_ +
               (static_cast<std::size_t>(source_index_type)) * num_index_types_ +
               static_cast<std::size_t>(target_index_type);
    }

    static constexpr std::size_t num_format_types_ = static_cast<std::size_t>(DenseFormat::Count);
    static constexpr std::size_t num_devices_ = static_cast<std::size_t>(Device::Count);
    static constexpr std::size_t num_numeric_types_ = static_cast<std::size_t>(NumericType::Count);
    static constexpr std::size_t num_init_free_dense_elems_ = num_format_types_ * num_devices_ * num_numeric_types_;
    static constexpr std::size_t num_copy_dense_elems_ = num_format_types_ * num_format_types_ * num_devices_ * num_devices_ * num_numeric_types_ * num_numeric_types_;
    static constexpr std::size_t num_index_types_ = static_cast<std::size_t>(IntType::Count);
    static constexpr std::size_t num_init_free_sparse_elems_ = num_devices_ * num_numeric_types_ * num_index_types_;
    static constexpr std::size_t num_copy_sparse_elems_ = num_devices_ * num_devices_ * num_numeric_types_ * num_numeric_types_ * num_index_types_ * num_index_types_;

    std::array<initialize_dense_fn, num_init_free_dense_elems_>   dense_initialize_table_{};
    std::array<free_dense_fn, num_init_free_dense_elems_>         dense_free_table_{};
    std::array<copy_dense_fn, num_copy_dense_elems_>              dense_copy_table_{};
    std::array<initialize_sparse_fn, num_init_free_sparse_elems_> sparse_initialize_table_{};
    std::array<free_sparse_fn, num_init_free_sparse_elems_>       sparse_free_table_{};
    std::array<copy_sparse_fn, num_copy_sparse_elems_>            sparse_copy_table_{};
}; // StorageRegistry

} // namespace detail
} // namespace RandLAPACK}