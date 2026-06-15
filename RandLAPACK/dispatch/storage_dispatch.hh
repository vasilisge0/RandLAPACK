#pragma once

#include <array>
#include <initializer_list>
#include <utility>

#include "csr_storage.hh"
#include "dense_storage.hh"
#include "device.hh"
#include "dimensions.hh"
#include "memory.hh"
#include "precision.hh"

namespace RandLAPACK {
namespace dispatch {

using initialize_dense_fn = void (*)(dim<2> size, DenseStorage& source);
using free_dense_fn = void (*)(DenseStorage& source);
using copy_dense_fn = void (*)(dim<2> size, DenseStorage& source,
                               DenseStorage& target);
using initialize_sparse_fn = void (*)(dim<2> size, SparseStorage& source);
using free_sparse_fn = void (*)(SparseStorage& source);
using copy_sparse_fn = void (*)(dim<2> size, SparseStorage& source,
                                SparseStorage& target);

template <typename Fn, std::size_t N>
static void fill(std::array<Fn, N>& table,
                 std::initializer_list<std::pair<std::size_t, Fn>> entries) {
    table.fill(nullptr);
    for (auto [idx, fn] : entries) table[idx] = fn;
}

// Mixed-radix encoder using Horner's method.
// Usage: encode(v0, d1, v1, d2, v2, ..., dn, vn)
// where v0..vn are dimension values and d1..dn are the sizes of each
// subsequent dimension.
static constexpr std::size_t encode(std::size_t acc) { return acc; }

template <typename T, typename... Rest>
static constexpr std::size_t encode(std::size_t acc, std::size_t dim, T next,
                                    Rest... rest) {
    return encode(acc * dim + static_cast<std::size_t>(next), rest...);
}

// Dense dispatch struct. This struct contains function pointer tables for
// Dense matrices.
//
// Every function that can be dispatched on Dense matrices
// should have the following private members:
// 1) A function pointer type defined above (e.g., initialize_dense_fn)
// 2) A fill_table function (e.g., fill_initialize_table) that fills the
// functions into the appropriate locations in the table. The fill function
// should be called in the constructor of the dispatch struct.
// 3) An encode function that encodes the input parameters into a unique index
// for the function pointer table.
//
// It also contains the following public members:
// 1) A get() function that returns a reference to the singleton instance of the
// dispatch struct.
// 2) A lookup function for each function pointer table (e.g.,
// lookup_initialize) that returns the appropriate function pointer based on the
// input parameters.
struct Dense {
public:
    static const Dense& get() {
        static const Dense instance;
        return instance;
    }

    initialize_dense_fn lookup_initialize(MatrixFormat format, Device dev,
                                          NumericType numeric_type) const {
        initialize_dense_fn fn =
            initialize_table_[encode_init_free(format, dev, numeric_type)];
        if (!fn) throw std::runtime_error("Unsupported storage type.");
        return fn;
    }

    free_dense_fn lookup_free(MatrixFormat format, Device dev,
                              NumericType numeric_type) const {
        free_dense_fn fn =
            free_table_[encode_init_free(format, dev, numeric_type)];
        if (!fn) throw std::runtime_error("Unsupported storage type.");
        return fn;
    }

    copy_dense_fn lookup_copy(MatrixFormat source_format,
                              MatrixFormat target_format, Device source_dev,
                              Device target_dev, NumericType source_type,
                              NumericType target_type) const {
        copy_dense_fn fn =
            copy_table_[encode_copy(source_format, target_format, source_dev,
                                    target_dev, source_type, target_type)];
        if (!fn) throw std::runtime_error("Unsupported storage type.");
        return fn;
    }

private:
    Dense() {
        fill_initialize_table();

        fill_free_table();

        fill_copy_table();
    }

    // Number of parameters for dense storage. Those are used to compute the
    // size of function tables.
    static constexpr std::size_t num_formats_ =
        static_cast<std::size_t>(DenseFormat::Count);
    static constexpr std::size_t num_devices_ =
        static_cast<std::size_t>(Device::Count);
    static constexpr std::size_t num_numeric_types_ =
        static_cast<std::size_t>(NumericType::Count);
    static constexpr std::size_t num_init_free_elems_ =
        num_formats_ * num_devices_ * num_numeric_types_;
    static constexpr std::size_t num_copy_elems_ =
        num_formats_ * num_formats_ * num_devices_ * num_devices_ *
        num_numeric_types_ * num_numeric_types_;

    static constexpr std::size_t encode_init_free(MatrixFormat format,
                                                  Device dev,
                                                  NumericType numeric_type) {
        return encode(static_cast<std::size_t>(format), num_devices_, dev,
                      num_numeric_types_, numeric_type);
    }

    static constexpr std::size_t encode_copy(MatrixFormat source_format,
                                             MatrixFormat target_format,
                                             Device source_dev,
                                             Device target_dev,
                                             NumericType source_type,
                                             NumericType target_type) {
        return encode(static_cast<std::size_t>(source_format), num_formats_,
                      target_format, num_devices_, source_dev, num_devices_,
                      target_dev, num_numeric_types_, source_type,
                      num_numeric_types_, target_type);
    }

    void fill_initialize_table() {
        fill(initialize_table_,
             {
                 {encode_init_free(MatrixFormat::STRIDED, Device::CPU,
                                   NumericType::FP64),
                  &initialize_strided_storage<Device::CPU, double>},
                 {encode_init_free(MatrixFormat::STRIDED, Device::CPU,
                                   NumericType::FP32),
                  &initialize_strided_storage<Device::CPU, float>},
                 {encode_init_free(MatrixFormat::STRIDED, Device::CPU,
                                   NumericType::FP16),
                  &initialize_strided_storage<Device::CPU, half>},
                 {encode_init_free(MatrixFormat::STRIDED, Device::CUDA,
                                   NumericType::FP64),
                  &initialize_strided_storage<Device::CUDA, double>},
                 {encode_init_free(MatrixFormat::STRIDED, Device::CUDA,
                                   NumericType::FP32),
                  &initialize_strided_storage<Device::CUDA, float>},
                 {encode_init_free(MatrixFormat::STRIDED, Device::CUDA,
                                   NumericType::FP16),
                  &initialize_strided_storage<Device::CUDA, half>},
             });
    }

    void fill_free_table() {
        fill(free_table_,
             {
                 {encode_init_free(MatrixFormat::STRIDED, Device::CPU,
                                   NumericType::FP64),
                  &free_strided_storage<Device::CPU, double>},
                 {encode_init_free(MatrixFormat::STRIDED, Device::CPU,
                                   NumericType::FP32),
                  &free_strided_storage<Device::CPU, float>},
                 {encode_init_free(MatrixFormat::STRIDED, Device::CPU,
                                   NumericType::FP16),
                  &free_strided_storage<Device::CPU, half>},
                 {encode_init_free(MatrixFormat::STRIDED, Device::CUDA,
                                   NumericType::FP64),
                  &free_strided_storage<Device::CUDA, double>},
                 {encode_init_free(MatrixFormat::STRIDED, Device::CUDA,
                                   NumericType::FP32),
                  &free_strided_storage<Device::CUDA, float>},
                 {encode_init_free(MatrixFormat::STRIDED, Device::CUDA,
                                   NumericType::FP16),
                  &free_strided_storage<Device::CUDA, half>},
             });
    }

    void fill_copy_table() {
        fill(
            copy_table_,
            {
                {encode_copy(MatrixFormat::STRIDED, MatrixFormat::STRIDED,
                             Device::CPU, Device::CPU, NumericType::FP64,
                             NumericType::FP64),
                 &copy_strided_storage<Device::CPU, Device::CPU, double,
                                       double>},
                {encode_copy(MatrixFormat::STRIDED, MatrixFormat::STRIDED,
                             Device::CPU, Device::CPU, NumericType::FP64,
                             NumericType::FP64),
                 &copy_strided_storage<Device::CPU, Device::CPU, double,
                                       float>},
                {encode_copy(MatrixFormat::STRIDED, MatrixFormat::STRIDED,
                             Device::CPU, Device::CPU, NumericType::FP64,
                             NumericType::FP64),
                 &copy_strided_storage<Device::CPU, Device::CPU, double, half>},
                {encode_copy(MatrixFormat::STRIDED, MatrixFormat::STRIDED,
                             Device::CPU, Device::CPU, NumericType::FP32,
                             NumericType::FP64),
                 &copy_strided_storage<Device::CPU, Device::CPU, float,
                                       double>},
                {encode_copy(MatrixFormat::STRIDED, MatrixFormat::STRIDED,
                             Device::CPU, Device::CPU, NumericType::FP32,
                             NumericType::FP32),
                 &copy_strided_storage<Device::CPU, Device::CPU, float, float>},
                {encode_copy(MatrixFormat::STRIDED, MatrixFormat::STRIDED,
                             Device::CPU, Device::CPU, NumericType::FP32,
                             NumericType::FP32),
                 &copy_strided_storage<Device::CPU, Device::CPU, float, half>},
                {encode_copy(MatrixFormat::STRIDED, MatrixFormat::STRIDED,
                             Device::CPU, Device::CPU, NumericType::FP16,
                             NumericType::FP16),
                 &copy_strided_storage<Device::CPU, Device::CPU, half, double>},
                {encode_copy(MatrixFormat::STRIDED, MatrixFormat::STRIDED,
                             Device::CPU, Device::CPU, NumericType::FP16,
                             NumericType::FP16),
                 &copy_strided_storage<Device::CPU, Device::CPU, half, float>},
                {encode_copy(MatrixFormat::STRIDED, MatrixFormat::STRIDED,
                             Device::CPU, Device::CPU, NumericType::FP16,
                             NumericType::FP16),
                 &copy_strided_storage<Device::CPU, Device::CPU, half, half>},
            });
    }

    std::array<initialize_dense_fn, num_init_free_elems_> initialize_table_{};
    std::array<free_dense_fn, num_init_free_elems_> free_table_{};
    std::array<copy_dense_fn, num_copy_elems_> copy_table_{};
};  // Dense

struct Sparse {
public:
    static const Sparse& get() {
        static const Sparse instance;
        return instance;
    }

    initialize_sparse_fn lookup_initialize(MatrixFormat format, Device dev,
                                           NumericType numeric_type,
                                           IntType index_type) const {
        initialize_sparse_fn fn = initialize_table_[encode_init_free(
            format, dev, numeric_type, index_type)];
        if (!fn) throw std::runtime_error("Unsupported storage type.");
        return fn;
    }

    free_sparse_fn lookup_free(MatrixFormat format, Device dev,
                               NumericType numeric_type,
                               IntType index_type) const {
        free_sparse_fn fn = free_table_[encode_init_free(
            format, dev, numeric_type, index_type)];
        if (!fn) throw std::runtime_error("Unsupported storage type.");
        return fn;
    }

    copy_sparse_fn lookup_copy(MatrixFormat source_format,
                               MatrixFormat target_format, Device source_dev,
                               Device target_dev,
                               NumericType source_numeric_type,
                               NumericType target_numeric_type,
                               IntType source_index_type,
                               IntType target_index_type) const {
        copy_sparse_fn fn = copy_table_[encode_copy(
            source_format, target_format, source_dev, target_dev,
            source_numeric_type, target_numeric_type, source_index_type,
            target_index_type)];
        if (!fn) throw std::runtime_error("Unsupported storage type.");
        return fn;
    }

private:
    Sparse() {
        fill(initialize_table_,
             {
                 {encode_init_free(MatrixFormat::CSR, Device::CPU,
                                   NumericType::FP64, IntType::INT64),
                  &initialize_csr_storage<Device::CPU, double, int64_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CPU,
                                   NumericType::FP32, IntType::INT64),
                  &initialize_csr_storage<Device::CPU, float, int64_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CPU,
                                   NumericType::FP16, IntType::INT64),
                  &initialize_csr_storage<Device::CPU, half, int64_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CPU,
                                   NumericType::FP64, IntType::INT32),
                  &initialize_csr_storage<Device::CPU, double, int32_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CPU,
                                   NumericType::FP32, IntType::INT32),
                  &initialize_csr_storage<Device::CPU, float, int32_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CPU,
                                   NumericType::FP16, IntType::INT32),
                  &initialize_csr_storage<Device::CPU, half, int32_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CUDA,
                                   NumericType::FP64, IntType::INT64),
                  &initialize_csr_storage<Device::CUDA, double, int64_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CUDA,
                                   NumericType::FP32, IntType::INT64),
                  &initialize_csr_storage<Device::CUDA, float, int64_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CUDA,
                                   NumericType::FP16, IntType::INT64),
                  &initialize_csr_storage<Device::CUDA, half, int64_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CUDA,
                                   NumericType::FP64, IntType::INT32),
                  &initialize_csr_storage<Device::CUDA, double, int32_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CUDA,
                                   NumericType::FP32, IntType::INT32),
                  &initialize_csr_storage<Device::CUDA, float, int32_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CUDA,
                                   NumericType::FP16, IntType::INT32),
                  &initialize_csr_storage<Device::CUDA, half, int32_t>},
             });

        fill(free_table_,
             {
                 {encode_init_free(MatrixFormat::CSR, Device::CPU,
                                   NumericType::FP64, IntType::INT64),
                  &free_csr_storage<Device::CPU, double, int64_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CPU,
                                   NumericType::FP32, IntType::INT64),
                  &free_csr_storage<Device::CPU, float, int64_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CPU,
                                   NumericType::FP16, IntType::INT64),
                  &free_csr_storage<Device::CPU, half, int64_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CPU,
                                   NumericType::FP64, IntType::INT32),
                  &free_csr_storage<Device::CPU, double, int32_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CPU,
                                   NumericType::FP32, IntType::INT32),
                  &free_csr_storage<Device::CPU, float, int32_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CPU,
                                   NumericType::FP16, IntType::INT32),
                  &free_csr_storage<Device::CPU, half, int32_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CUDA,
                                   NumericType::FP64, IntType::INT64),
                  &free_csr_storage<Device::CUDA, double, int64_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CUDA,
                                   NumericType::FP32, IntType::INT64),
                  &free_csr_storage<Device::CUDA, float, int64_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CUDA,
                                   NumericType::FP16, IntType::INT64),
                  &free_csr_storage<Device::CUDA, half, int64_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CUDA,
                                   NumericType::FP64, IntType::INT32),
                  &free_csr_storage<Device::CUDA, double, int32_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CUDA,
                                   NumericType::FP32, IntType::INT32),
                  &free_csr_storage<Device::CUDA, float, int32_t>},
                 {encode_init_free(MatrixFormat::CSR, Device::CUDA,
                                   NumericType::FP16, IntType::INT32),
                  &free_csr_storage<Device::CUDA, half, int32_t>},
             });

        fill(
            copy_table_,
            {
                {encode_copy(MatrixFormat::CSR, MatrixFormat::CSR, Device::CUDA,
                             Device::CPU, NumericType::FP64, NumericType::FP64,
                             IntType::INT64, IntType::INT64),
                 &copy_csr_storage<Device::CUDA, Device::CPU, double, double,
                                   int64_t, int64_t>},
            });
    }

    static constexpr std::size_t num_devices_ =
        static_cast<std::size_t>(Device::Count);
    static constexpr std::size_t num_numeric_types_ =
        static_cast<std::size_t>(NumericType::Count);
    static constexpr std::size_t num_index_types_ =
        static_cast<std::size_t>(IntType::Count);
    static constexpr std::size_t num_format_types_ =
        static_cast<std::size_t>(SparseFormat::Count);
    static constexpr std::size_t num_init_free_elems_ =
        num_format_types_ * num_devices_ * num_numeric_types_ *
        num_index_types_;
    static constexpr std::size_t num_copy_elems_ =
        num_format_types_ * num_format_types_ * num_devices_ * num_devices_ *
        num_numeric_types_ * num_numeric_types_ * num_index_types_ *
        num_index_types_;

    static constexpr std::size_t encode_init_free(MatrixFormat format,
                                                  Device dev,
                                                  NumericType numeric_type,
                                                  IntType index_type) {
        return encode(static_cast<std::size_t>(format), num_devices_, dev,
                      num_numeric_types_, numeric_type, num_index_types_,
                      index_type);
    }

    static constexpr std::size_t encode_copy(
        MatrixFormat source_format, MatrixFormat target_format,
        Device source_dev, Device target_dev, NumericType source_numeric_type,
        NumericType target_numeric_type, IntType source_index_type,
        IntType target_index_type) {
        return encode(static_cast<std::size_t>(source_format),
                      num_format_types_, target_format, num_devices_,
                      source_dev, num_devices_, target_dev, num_numeric_types_,
                      source_numeric_type, num_numeric_types_,
                      target_numeric_type, num_index_types_, source_index_type,
                      num_index_types_, target_index_type);
    }

    std::array<initialize_sparse_fn, num_init_free_elems_> initialize_table_{};
    std::array<free_sparse_fn, num_init_free_elems_> free_table_{};
    std::array<copy_sparse_fn, num_copy_elems_> copy_table_{};
};  // Sparse

}  // namespace dispatch
}  // namespace RandLAPACK
