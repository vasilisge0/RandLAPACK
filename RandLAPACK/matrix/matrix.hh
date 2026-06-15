#pragma once

#include <array>
#include <memory>
#include <variant>

#include "context.hh"
#include "csr_storage.hh"
#include "dense_storage.hh"
#include "device.hh"
#include "dimensions.hh"
#include "matrix_impl.hh"
#include "matrix_registry.hh"
#include "memory.hh"
#include "precision.hh"
#include "status.hh"
#include "storage_dispatch.hh"

namespace RandLAPACK {

// Descriptors used for matrix objects

struct StridedDescriptor {
    dim<2> size_ = {0, 0};
    size_t lead_dim_ = 0;
    Layout layout_ = Layout::COL_MAJOR;
    NumericType store_type_;

    StridedDescriptor() = delete;
    StridedDescriptor(Layout layout, NumericType store_type, size_t lead_dim)
        : lead_dim_(lead_dim), layout_(layout), store_type_(store_type) {}
};

struct CsrDescriptor {
    dim<2> size_ = {0, 0};
    size_t nnz_ = 0;
    NumericType store_type_;
    IntType index_type_;

    CsrDescriptor() = delete;
    CsrDescriptor(size_t nnz, NumericType store_type, IntType index_type)
        : index_type_(index_type), nnz_(nnz), store_type_(store_type) {}
};

struct MatrixDescriptor {
    std::variant<StridedDescriptor, CsrDescriptor> storage_descriptor_;

    MatrixDescriptor() = delete;

    MatrixDescriptor(StridedDescriptor dense_descr)
        : storage_descriptor_(dense_descr) {}

    MatrixDescriptor(CsrDescriptor csr_descr)
        : storage_descriptor_(csr_descr) {}

    MatrixFormat get_format() const {
        return static_cast<MatrixFormat>(storage_descriptor_.index());
    }

    dim<2> get_size() const {
        return std::visit([](auto&& arg) -> dim<2> { return arg.size_; },
                          storage_descriptor_);
    }
};

struct Dense;

// View objects used to access dense matrices without copying data. These are
// used as arguments to kernels, and are constructed from Dense objects. They do
// not own any data, and are not responsible for deallocating any data.

template <typename value_t>
struct StridedView {
    Device device_;
    dim<2> size_ = {0, 0};
    size_t lead_dim_ = 0;
    size_t num_elems_ = 0;
    Layout layout_ = Layout::COL_MAJOR;
    value_t* values_ = nullptr;

    StridedView(RandLAPACK::Dense& source) {
        StridedStorage storage = std::get<StridedStorage>(source.storage_);
        value_t* values = std::get<value_t*>(storage.values_);
        size_ = storage.size_;
        lead_dim_ = storage.lead_dim_;
        layout_ = storage.layout_;
        values_ = values;
        device_ = source.device_;
        num_elems_ = storage.num_elems_;
    }
};

// Matrix objects that own memory.

struct Dense {
    template <typename value_t>
    friend struct StridedView;

private:
    Device device_;
    DenseStorage storage_;

protected:
    Dense(Context& context, MatrixFormat matrix_type, NumericType store_type,
          Layout layout, dim<2> size, size_t lead_dim) {
        if (matrix_type == MatrixFormat::STRIDED) {
            storage_ = StridedStorage(store_type, layout, size, lead_dim);
        } else {
            throw std::invalid_argument("Unsupported matrix type.");
        }
        device_ = context.get_device();
        detail::initialize_dense_storage(size, context.get_device(), storage_);
    }

    Dense(Context& context, NumericType store_type, std::string& filename) {
        // @TODO: add implementation
    }

    Dense(Context& context, MatrixDescriptor descriptor) {
        // @TODO: add implementation
    }

public:
    static std::unique_ptr<Dense> create_strided(Context& context,
                                                 Layout layout,
                                                 NumericType store_precision,
                                                 dim<2> size) {
        Dense* tmp = new Dense(context, MatrixFormat::STRIDED, store_precision,
                               layout, size, size[0]);
        return std::unique_ptr<Dense>(tmp);
    }

    static std::unique_ptr<Dense> create_strided(Context& context,
                                                 Layout layout,
                                                 NumericType store_precision,
                                                 dim<2> size, size_t lead_dim) {
        Dense* tmp = new Dense(context, MatrixFormat::STRIDED, store_precision,
                               layout, size, lead_dim);
        return std::unique_ptr<Dense>(tmp);
    }

    static std::unique_ptr<Dense> create_strided(Context& context,
                                                 Layout layout,
                                                 NumericType store_precision,
                                                 std::string& filename) {
        Dense* tmp = new Dense(context, store_precision, filename);
        return std::unique_ptr<Dense>(tmp);
    }

    static std::unique_ptr<Dense> create(Context& context,
                                         MatrixDescriptor descriptor) {
        Dense* tmp = new Dense(context, descriptor);
        return std::unique_ptr<Dense>(tmp);
    }

    static std::unique_ptr<Dense> create_from(Context& context, Dense& source) {
        std::unique_ptr<Dense> target =
            create(context, source.get_descriptor());
        target->copy_from(source);
        return target;
    }

    static std::unique_ptr<Dense> create_from(Dense& source) {
        Context context = Context(source.get_device());
        std::unique_ptr<Dense> target =
            create(context, source.get_descriptor());
        target->copy_from(source);
        return target;
    }

    void copy_from(Dense& source) {
        if (get_format() != source.get_format())
            throw std::invalid_argument(
                "Copying values from matrices of different types");

        dim<2> size = get_size();
        if (size != source.get_size())
            throw std::invalid_argument(
                "Copying from matrix of incompatible size.");

        detail::copy_dense_storage(size, source.device_, device_,
                                   source.storage_, storage_);
    }

    MatrixDescriptor get_descriptor() const {
        NumericType data_type = get_value_type();
        return std::visit(
            [data_type](auto&& arg) -> MatrixDescriptor {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, StridedStorage>) {
                    return MatrixDescriptor(StridedDescriptor(
                        arg.layout_, data_type, arg.lead_dim_));
                } else {
                    throw std::invalid_argument("Unsupported matrix format.");
                }
            },
            storage_);
    }

    dim<2> get_size() const {
        return std::visit([](auto&& arg) -> dim<2> { return arg.size_; },
                          storage_);
    };

    MatrixFormat get_format() const {
        return static_cast<MatrixFormat>(storage_.index());
    }

    Device get_device() const { return device_; }

    NumericType get_value_type() const {
        return std::visit(
            [](auto&& arg) -> NumericType {
                return std::visit(
                    [](auto&& subarg) -> NumericType {
                        return numeric_type_of<std::remove_pointer_t<
                            std::decay_t<decltype(subarg)>>>;
                    },
                    arg.values_);
            },
            storage_);
    }

    ~Dense() { detail::free_dense_storage(device_, storage_); }
};

template <typename value_t>
StridedView<value_t> as_strided_view(Dense& source) {
    StridedStorage storage = std::get<StridedStorage>(source.storage_);
    return StridedView<value_t>{
        source.device_,    storage.size_,
        storage.lead_dim_, storage.num_elems_,
        storage.layout_,   std::get<value_t*>(storage.values_)};
}

struct Sparse {
    template <typename value_t, typename index_t>
    friend class CsrView;

private:
    Device device_;
    SparseStorage storage_;

protected:
    Sparse(Context& context, MatrixDescriptor descriptor) {
        device_ = context.get_device();
        std::visit(
            [this](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, CsrDescriptor>) {
                    storage_ = CsrStorage(arg.store_type_, arg.index_type_,
                                          arg.size_, arg.nnz_);
                } else {
                    throw std::invalid_argument("Unsupported matrix format.");
                }
            },
            descriptor.storage_descriptor_);
    }

public:
    static std::unique_ptr<Sparse> create_csr(Context& context,
                                              NumericType value_type,
                                              IntType index_type, dim<2> size,
                                              size_t nnz) {
        Sparse* tmp = new Sparse(context, MatrixDescriptor(CsrDescriptor(
                                              nnz, value_type, index_type)));
        return std::unique_ptr<Sparse>(tmp);
    }

    static std::unique_ptr<Sparse> create(Context& context,
                                          MatrixDescriptor descriptor) {
        Sparse* tmp = new Sparse(context, descriptor);
        return std::unique_ptr<Sparse>(tmp);
    }

    static std::unique_ptr<Sparse> create_from(Context& context,
                                               Sparse& source) {
        std::unique_ptr<Sparse> target =
            create(context, source.get_descriptor());
        target->copy_from(source);
        return target;
    }

    void copy_from(Sparse& source) {
        if (get_format() != source.get_format())
            throw std::invalid_argument(
                "Copying values from matrices of different types");

        dim<2> size = get_size();
        if (size != source.get_size())
            throw std::invalid_argument(
                "Copying from matrix of incompatible size.");

        detail::copy_sparse_storage(size, source.device_, device_,
                                    source.storage_, storage_);
    }

    MatrixDescriptor get_descriptor() const {
        NumericType value_type = get_value_type();
        IntType index_type = get_index_type();
        return std::visit(
            [value_type, index_type](auto&& arg) -> MatrixDescriptor {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, CsrStorage>) {
                    return MatrixDescriptor(
                        CsrDescriptor(arg.nnz_, value_type, index_type));
                } else {
                    throw std::invalid_argument("Unsupported matrix format.");
                }
            },
            storage_);
    }

    dim<2> get_size() const {
        return std::visit([](auto&& arg) -> dim<2> { return arg.size_; },
                          storage_);
    }

    MatrixFormat get_format() const {
        return static_cast<MatrixFormat>(storage_.index());
    }

    Device get_device() const { return device_; }

    NumericType get_value_type() const {
        return std::visit(
            [](auto&& arg) -> NumericType {
                return std::visit(
                    [](auto&& subarg) -> NumericType {
                        return numeric_type_of<std::remove_pointer_t<
                            std::decay_t<decltype(subarg)>>>;
                    },
                    arg.values_);
            },
            storage_);
    }

    IntType get_index_type() const {
        return std::visit(
            [](auto&& arg) -> IntType {
                return static_cast<IntType>(arg.row_ptrs_.index());
            },
            storage_);
    }

    ~Sparse() { detail::free_sparse_storage(device_, storage_); }
};

template <typename value_t, typename index_t>
struct CsrView {
    Device device_;
    dim<2> size_;
    size_t nnz_;
    NumericPtrVariant values_;
    IntPtrVariant row_ptrs_;
    IntPtrVariant col_idxs_;

    CsrView(Sparse& source) {
        CsrStorage storage = std::get<CsrStorage>(source.storage_);
        values_ = std::get<value_t*>(storage.values_);
        row_ptrs_ = std::get<index_t*>(storage.row_ptrs_);
        col_idxs_ = std::get<index_t*>(storage.col_idxs_);
        size_ = storage.size_;
        nnz_ = storage.nnz_;
        device_ = source.device_;
    }
};

template <typename value_t, typename index_t>
CsrView<value_t, index_t> as_csr_view(Sparse& source) {
    CsrStorage storage = std::get<CsrStorage>(source.storage_);
    return CsrView<value_t, index_t>{source.device_,    storage.size_,
                                     storage.nnz_,      storage.values_,
                                     storage.row_ptrs_, storage.col_idxs_};
}

using Matrix = std::variant<Dense, Sparse>;

}  // namespace RandLAPACK