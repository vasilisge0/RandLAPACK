#pragma once

#include <array>
#include <memory>
#include <optional>
#include <variant>

#include "context.hh"
#include "csr_storage.hh"
// #include "dense_storage.hh"
#include "device.hh"
#include "dimensions.hh"
// #include "matrix_impl.hh"
// #include "matrix_registry.hh"
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
    CsrDescriptor(dim<2> size, size_t nnz, NumericType store_type,
                   IntType index_type)
        : size_(size),
          nnz_(nnz),
          store_type_(store_type),
          index_type_(index_type) {}
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
        device_ = *source.device_;
        num_elems_ = storage.num_elems_;
    }
};

// Matrix objects that own memory.

struct Dense {
    template <typename value_t>
    friend struct StridedView;

private:
    std::optional<Device> device_;
    DenseStorage storage_;

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

public:
    Dense() = default;

    Dense(Dense&& other) noexcept
        : device_(other.device_), storage_(other.storage_) {
        other.device_ = std::nullopt;
        other.storage_ = StridedStorage();
    }

    Dense& operator=(Dense&& other) noexcept {
        if (this != &other) {
            if (device_)
                detail::free_dense_storage(*device_, storage_);
            device_ = other.device_;
            storage_ = other.storage_;
            other.device_ = std::nullopt;
            other.storage_ = StridedStorage();
        }
        return *this;
    }

    Dense(const Dense&) = delete;
    Dense& operator=(const Dense&) = delete;

    static Dense create_strided(Context& context, Layout layout,
                                NumericType store_precision, dim<2> size) {
        return Dense(context, MatrixFormat::STRIDED, store_precision, layout,
                     size, size[0]);
    }

    static Dense create_strided(Context& context, Layout layout,
                                NumericType store_precision, dim<2> size,
                                size_t lead_dim) {
        return Dense(context, MatrixFormat::STRIDED, store_precision, layout,
                     size, lead_dim);
    }

    static Dense create_from(Context& context, Dense& source) {
        Dense target = create_strided(
            context,
            std::get<StridedStorage>(source.storage_).layout_,
            source.get_value_type(), source.get_size(),
            std::get<StridedStorage>(source.storage_).lead_dim_);
        target.copy_from(source);
        return target;
    }

    static Dense create_from(Dense& source) {
        Context context = Context(source.get_device());
        return create_from(context, source);
    }

    void copy_from(Dense& source) {
        if (get_format() != source.get_format())
            throw std::invalid_argument(
                "Copying values from matrices of different types");

        dim<2> size = get_size();
        if (size != source.get_size())
            throw std::invalid_argument(
                "Copying from matrix of incompatible size.");

        detail::copy_dense_storage(size, *source.device_, *device_,
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

    Device get_device() const { return *device_; }

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

    ~Dense() {
        if (device_)
            detail::free_dense_storage(*device_, storage_);
    }
};

template <typename value_t>
StridedView<value_t> as_strided_view(Dense& source) {
    StridedStorage storage = std::get<StridedStorage>(source.storage_);
    return StridedView<value_t>{
        *source.device_,   storage.size_,
        storage.lead_dim_, storage.num_elems_,
        storage.layout_,   std::get<value_t*>(storage.values_)};
}

struct Sparse {
    template <typename value_t, typename index_t>
    friend struct CsrView;

private:
    std::optional<Device> device_;
    SparseStorage storage_;

    Sparse(Context& context, NumericType value_type, IntType index_type,
           dim<2> size, size_t nnz) {
        device_ = context.get_device();
        storage_ = CsrStorage(value_type, index_type, size, nnz);
        detail::initialize_sparse_storage(*device_, storage_);
    }

public:
    Sparse() = default;

    Sparse(Sparse&& other) noexcept
        : device_(other.device_), storage_(other.storage_) {
        other.device_ = std::nullopt;
        other.storage_ = CsrStorage();
    }

    Sparse& operator=(Sparse&& other) noexcept {
        if (this != &other) {
            if (device_)
                detail::free_sparse_storage(*device_, storage_);
            device_ = other.device_;
            storage_ = other.storage_;
            other.device_ = std::nullopt;
            other.storage_ = CsrStorage();
        }
        return *this;
    }

    Sparse(const Sparse&) = delete;
    Sparse& operator=(const Sparse&) = delete;

    static Sparse create_csr(Context& context, NumericType value_type,
                              IntType index_type, dim<2> size, size_t nnz) {
        return Sparse(context, value_type, index_type, size, nnz);
    }

    static Sparse create_from(Context& context, Sparse& source) {
        CsrStorage& src_storage = std::get<CsrStorage>(source.storage_);
        Sparse target = create_csr(context, source.get_value_type(),
                                   source.get_index_type(),
                                   src_storage.size_, src_storage.nnz_);
        target.copy_from(source);
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

        detail::copy_sparse_storage(size, *source.device_, *device_,
                                    source.storage_, storage_);
    }

    MatrixDescriptor get_descriptor() const {
        NumericType value_type = get_value_type();
        IntType index_type = get_index_type();
        return std::visit(
            [value_type, index_type](auto&& arg) -> MatrixDescriptor {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, CsrStorage>) {
                    return MatrixDescriptor(CsrDescriptor(
                        arg.size_, arg.nnz_, value_type, index_type));
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
        return std::visit(
            [](auto&& arg) -> MatrixFormat {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, CsrStorage>) {
                    return MatrixFormat::CSR;
                } else {
                    throw std::invalid_argument("Unknown sparse format.");
                }
            },
            storage_);
    }

    Device get_device() const { return *device_; }

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

    ~Sparse() {
        if (device_)
            detail::free_sparse_storage(*device_, storage_);
    }
};

template <typename value_t, typename index_t>
struct CsrView {
    Device device_;
    dim<2> size_{0, 0};
    size_t nnz_ = 0;
    value_t* values_ = nullptr;
    index_t* row_ptrs_ = nullptr;
    index_t* col_idxs_ = nullptr;

    CsrView(Sparse& source) {
        CsrStorage storage = std::get<CsrStorage>(source.storage_);
        values_ = std::get<value_t*>(storage.values_);
        row_ptrs_ = std::get<index_t*>(storage.row_ptrs_);
        col_idxs_ = std::get<index_t*>(storage.col_idxs_);
        size_ = storage.size_;
        nnz_ = storage.nnz_;
        device_ = *source.device_;
    }
};

template <typename value_t, typename index_t>
CsrView<value_t, index_t> as_csr_view(Sparse& source) {
    CsrStorage storage = std::get<CsrStorage>(source.storage_);
    return CsrView<value_t, index_t>{*source.device_,   storage.size_,
                                     storage.nnz_,      storage.values_,
                                     storage.row_ptrs_, storage.col_idxs_};
}

using MatrixHandle = std::variant<Dense*, Sparse*>;

}  // namespace RandLAPACK