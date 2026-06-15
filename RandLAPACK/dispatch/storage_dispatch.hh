#pragma once

#include <array>

#include "csr_storage.hh"
#include "dense_storage.hh"
#include "device.hh"
#include "dimensions.hh"
#include "memory.hh"
#include "precision.hh"

namespace RandLAPACK {

using all_devices = vlist<Device::CPU, Device::CUDA>;

namespace dispatch {

using initialize_dense_fn = void (*)(dim<2> size, DenseStorage& source);
using free_dense_fn = void (*)(DenseStorage& source);
using copy_dense_fn = void (*)(dim<2> size, DenseStorage& source,
                               DenseStorage& target);
using initialize_sparse_fn = void (*)(dim<2> size, SparseStorage& source);
using free_sparse_fn = void (*)(SparseStorage& source);
using copy_sparse_fn = void (*)(dim<2> size, SparseStorage& source,
                                SparseStorage& target);

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
        initialize_table_.fill(nullptr);
        for_each(all_devices{}, [&]<Device Dev>() {
            for_each(all_numerics{}, [&]<NumericType Nt>() {
                using T = typename to_concrete_numeric_type<Nt>::value;
                initialize_table_[encode_init_free(MatrixFormat::STRIDED, Dev,
                                                   Nt)] =
                    &initialize_strided_storage<Dev, T>;
            });
        });
    }

    void fill_free_table() {
        free_table_.fill(nullptr);
        for_each(all_devices{}, [&]<Device Dev>() {
            for_each(all_numerics{}, [&]<NumericType Nt>() {
                using T = typename to_concrete_numeric_type<Nt>::value;
                free_table_[encode_init_free(MatrixFormat::STRIDED, Dev, Nt)] =
                    &free_strided_storage<Dev, T>;
            });
        });
    }

    void fill_copy_table() {
        copy_table_.fill(nullptr);
        for_each(all_devices{}, [&]<Device SrcDev>() {
            for_each(all_devices{}, [&]<Device TgtDev>() {
                for_each(all_numerics{}, [&]<NumericType SrcNt>() {
                    for_each(all_numerics{}, [&]<NumericType TgtNt>() {
                        using source_float_t =
                            typename to_concrete_numeric_type<SrcNt>::value;
                        using target_float_t =
                            typename to_concrete_numeric_type<TgtNt>::value;
                        copy_table_[encode_copy(MatrixFormat::STRIDED,
                                                MatrixFormat::STRIDED, SrcDev,
                                                TgtDev, SrcNt, TgtNt)] =
                            &copy_strided_storage<
                                SrcDev, TgtDev, source_float_t, target_float_t>;
                    });
                });
            });
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
        fill_initialize_table();
        fill_free_table();
        fill_copy_table();
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

    void fill_initialize_table() {
        initialize_table_.fill(nullptr);
        for_each(all_devices{}, [&]<Device Dev>() {
            for_each(all_numerics{}, [&]<NumericType Nt>() {
                for_each(all_ints{}, [&]<IntType It>() {
                    using T = typename to_concrete_numeric_type<Nt>::value;
                    using I = typename to_concrete_integer_type<It>::value;
                    initialize_table_[encode_init_free(MatrixFormat::CSR, Dev,
                                                       Nt, It)] =
                        &initialize_csr_storage<Dev, T, I>;
                });
            });
        });
    }

    void fill_free_table() {
        free_table_.fill(nullptr);
        for_each(all_devices{}, [&]<Device Dev>() {
            for_each(all_numerics{}, [&]<NumericType Nt>() {
                for_each(all_ints{}, [&]<IntType It>() {
                    using T = typename to_concrete_numeric_type<Nt>::value;
                    using I = typename to_concrete_integer_type<It>::value;
                    free_table_[encode_init_free(MatrixFormat::CSR, Dev, Nt,
                                                 It)] =
                        &free_csr_storage<Dev, T, I>;
                });
            });
        });
    }

    void fill_copy_table() {
        copy_table_.fill(nullptr);
        for_each(all_devices{}, [&]<Device SrcDev>() {
            for_each(all_devices{}, [&]<Device TgtDev>() {
                for_each(all_numerics{}, [&]<NumericType SrcNt>() {
                    for_each(all_numerics{}, [&]<NumericType TgtNt>() {
                        for_each(all_ints{}, [&]<IntType source_int_tt>() {
                            for_each(all_ints{}, [&]<IntType target_int_tt>() {
                                // using source_float_t =
                                //     typename
                                //     to_concrete_numeric_type<SrcNt>::value;
                                // using target_float_t =
                                //     typename
                                //     to_concrete_numeric_type<TgtNt>::value;
                                // using source_int_t =
                                //     typename
                                //     to_concrete_integer_type<source_int_tt>::value;
                                // using target_int_t =
                                //     typename
                                //     to_concrete_integer_type<target_int_tt>::value;
                                // copy_table_[encode_copy(
                                //     MatrixFormat::CSR, MatrixFormat::CSR,
                                //     SrcDev, TgtDev, SrcNt, TgtNt,
                                //     source_int_tt, target_int_tt)] =
                                //     &copy_csr_storage<
                                //         SrcDev, TgtDev, source_float_t,
                                //         target_float_t, source_int_t,
                                //         target_int_t>;
                            });
                        });
                    });
                });
            });
        });
    }

    std::array<initialize_sparse_fn, num_init_free_elems_> initialize_table_{};
    std::array<free_sparse_fn, num_init_free_elems_> free_table_{};
    std::array<copy_sparse_fn, num_copy_elems_> copy_table_{};
};  // Sparse

}  // namespace dispatch

namespace detail {

// Helper functions definitions for initialization and freeing of matrix
// storage. These functions will be called by the constructors and destructors
// of MatrixInternals to initialize and free the matrix storage, and these
// functions will lookup the correct function pointer from the MatrixRegistry to
// call the correct initialize and free functions for different matrix types,
// devices, and numeric types.

NumericType get_numeric_type(DenseStorage& source) {
    return static_cast<NumericType>(
        std::get<StridedStorage>(source).values_.index());
}

NumericType get_numeric_type(SparseStorage& source) {
    return static_cast<NumericType>(
        std::get<CsrStorage>(source).values_.index());
}

IntType get_index_type(SparseStorage& source) {
    return static_cast<IntType>(std::get<CsrStorage>(source).row_ptrs_.index());
}

dim<2> get_size(DenseStorage& source) {
    return std::get<StridedStorage>(source).size_;
}

dim<2> get_size(SparseStorage& source) {
    return std::get<CsrStorage>(source).size_;
}

void initialize_dense_storage(dim<2> size, Device dev, DenseStorage& source) {
    auto fn = dispatch::Dense::get().lookup_initialize(
        static_cast<MatrixFormat>(source.index()), dev,
        get_numeric_type(source));
    fn(size, source);
}

void free_dense_storage(Device dev, DenseStorage& source) {
    auto fn = dispatch::Dense::get().lookup_free(
        static_cast<MatrixFormat>(source.index()), dev,
        get_numeric_type(source));
    fn(source);
}

void copy_dense_storage(dim<2> size, Device source_dev, Device target_dev,
                        DenseStorage& source, DenseStorage& target) {
    auto fn = dispatch::Dense::get().lookup_copy(
        static_cast<MatrixFormat>(source.index()),
        static_cast<MatrixFormat>(target.index()), source_dev, target_dev,
        get_numeric_type(source), get_numeric_type(target));
    fn(size, source, target);
}

void initialize_sparse_storage(Device dev, SparseStorage& source) {
    auto fn = dispatch::Sparse::get().lookup_initialize(
        static_cast<MatrixFormat>(source.index()), dev,
        get_numeric_type(source), get_index_type(source));
    fn(get_size(source), source);
}

void free_sparse_storage(Device dev, SparseStorage& source) {
    auto fn = dispatch::Sparse::get().lookup_free(
        static_cast<MatrixFormat>(source.index()), dev,
        get_numeric_type(source), get_index_type(source));
    fn(source);
}

void copy_sparse_storage(dim<2> size, Device source_dev, Device target_dev,
                         SparseStorage& source, SparseStorage& target) {
    auto fn = dispatch::Sparse::get().lookup_copy(
        static_cast<MatrixFormat>(source.index()),
        static_cast<MatrixFormat>(target.index()), source_dev, target_dev,
        get_numeric_type(source), get_numeric_type(target),
        get_index_type(source), get_index_type(target));
    fn(size, source, target);
}

}  // namespace detail

}  // namespace RandLAPACK