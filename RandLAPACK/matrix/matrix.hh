#pragma once

#include <array>
#include <memory>
#include <variant>

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
    StridedDescriptor(size_t lead_dim, Layout layout, NumericType store_type)
        : lead_dim_(lead_dim), layout_(layout), store_type_(store_type) {}
};  // StridedDescriptor

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

// Helper functions definitions for initialization and freeing of matrix
// storage. These functions will be called by the constructors and destructors
// of MatrixInternals to initialize and free the matrix storage, and these
// functions will lookup the correct function pointer from the MatrixRegistry to
// call the correct initialize and free functions for different matrix types,
// devices, and numeric types.

namespace detail {

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
    auto fn = StorageRegistry::get().lookup_initialize_dense(
        static_cast<MatrixFormat>(source.index()), dev,
        get_numeric_type(source));
    fn(size, source);
}

void free_dense_storage(Device dev, DenseStorage& source) {
    auto fn = StorageRegistry::get().lookup_free_dense(
        static_cast<MatrixFormat>(source.index()), dev,
        get_numeric_type(source));
    fn(source);
}

void copy_dense_storage(dim<2> size, Device source_dev, Device target_dev,
                        DenseStorage& source, DenseStorage& target) {
    auto fn = StorageRegistry::get().lookup_copy_dense(
        static_cast<MatrixFormat>(static_cast<MatrixFormat>(source.index())),
        static_cast<MatrixFormat>(target.index()), source_dev, target_dev,
        get_numeric_type(source), get_numeric_type(target));
    fn(size, source, target);
}

void initialize_sparse_storage(Device dev, SparseStorage& source) {
    auto fn = StorageRegistry::get().lookup_sparse_initialize(
        static_cast<MatrixFormat>(source.index()), dev,
        get_numeric_type(source), get_index_type(source));
    fn(get_size(source), source);
}

void free_sparse_storage(Device dev, SparseStorage& source) {
    auto fn = StorageRegistry::get().lookup_sparse_free(
        static_cast<MatrixFormat>(source.index()), dev,
        get_numeric_type(source), get_index_type(source));
    fn(source);
}

void copy_sparse_storage(dim<2> size, Device source_dev, Device target_dev,
                         SparseStorage& source, SparseStorage& target) {
    auto fn = StorageRegistry::get().lookup_copy_sparse(
        static_cast<MatrixFormat>(source.index()),
        static_cast<MatrixFormat>(target.index()), source_dev, target_dev,
        get_numeric_type(source), get_numeric_type(target),
        get_index_type(source), get_index_type(target));
    fn(size, source, target);
}

}  // namespace detail

// Forward declaration of matrix views

struct StridedView;

struct CsrView;

struct DenseState {
   public:
    RandLAPACK::Device device_;
    RandLAPACK::DenseStorage storage_;

    size_t num_elems_ = 0;
    dim<2> size_ = {0, 0};

    DenseState() = default;

    DenseState(dim<2> size, Layout layout, Device device,
               NumericType store_precision)
        : device_(device), storage_(StridedStorage(size, size[0], layout)) {
        detail::initialize_dense_storage(size, device_, storage_);
    }

    DenseState(MatrixDescriptor descriptor, Device device) {
        device_ = device;
        size_ = descriptor.get_size();
        if (descriptor.get_format() == MatrixFormat::STRIDED) {
            StridedDescriptor dense_descr =
                std::get<StridedDescriptor>(descriptor.storage_descriptor_);
            storage_ = StridedStorage(size_, dense_descr.lead_dim_,
                                      dense_descr.layout_);
            detail::initialize_dense_storage(size_, device_, storage_);
        }
    }

    ~DenseState() { detail::free_dense_storage(device_, storage_); }

    void copy_from(DenseState& source) {
        if (storage_.index() != source.storage_.index())
            throw std::invalid_argument(
                "Copying values from matrices of different types");
        if (size_ != source.size_)
            throw std::invalid_argument(
                "Copying from matrix of incompatible size.");
        detail::copy_dense_storage(size_, source.device_, device_,
                                   source.storage_, storage_);
    }
};  // DenseState

struct SparseState {
   public:
    RandLAPACK::Device device_;
    RandLAPACK::SparseStorage storage_;

    SparseState() = default;

    SparseState(dim<2> size, size_t nnz, Device device,
                NumericType store_precision, IntType index_type)
        : device_(device) {
        storage_ = CsrStorage(size, nnz, store_precision, index_type);
        detail::initialize_sparse_storage(device_, storage_);
    }  // SparseState

    SparseState(MatrixDescriptor descriptor, Device device) {
        device_ = device;
        if (descriptor.get_format() == MatrixFormat::CSR) {
            CsrDescriptor sparse_descr =
                std::get<CsrDescriptor>(descriptor.storage_descriptor_);
            storage_ =
                CsrStorage(sparse_descr.size_, sparse_descr.nnz_,
                           sparse_descr.store_type_, sparse_descr.index_type_);
            detail::initialize_sparse_storage(device_, storage_);
        }
    }  // SparseState

    ~SparseState() {
        detail::free_sparse_storage(device_, storage_);
    }  // ~SparseState

    dim<2> get_size() const {
        return std::visit([](auto&& arg) -> dim<2> { return arg.size_; },
                          storage_);
    };

    void copy_from(SparseState& source) {
        if (storage_.index() != source.storage_.index())
            throw std::invalid_argument(
                "Copying values from matrices of different types");
        dim<2> size = std::visit([](auto&& arg) -> dim<2> { return arg.size_; },
                                 storage_);
        if (size != source.get_size())
            throw std::invalid_argument(
                "Copying from matrix of incompatible size.");
        detail::copy_sparse_storage(size, source.device_, device_,
                                    source.storage_, storage_);
    }
};  // SparseState

// Dense/Sparse matrix interfaces

struct Dense {
    friend struct StridedView;

   public:
    static std::unique_ptr<Dense> create_strided(dim<2> size,
                                                 RandLAPACK::Layout layout,
                                                 Device device,
                                                 NumericType store_precision) {
        Dense* tmp = new Dense(size, layout, device, store_precision);
        return std::unique_ptr<Dense>(tmp);
    }

    static std::unique_ptr<Dense> create_strided(std::string& filename,
                                                 RandLAPACK::Layout layout,
                                                 Device device,
                                                 NumericType store_precision) {}

    static std::unique_ptr<Dense> create(MatrixDescriptor descriptor,
                                         Device device) {
        Dense* tmp = new Dense(descriptor, device);
        return std::unique_ptr<Dense>(tmp);
    }

    static std::unique_ptr<Dense> create_from(Dense& source) {
        std::unique_ptr<Dense> target =
            create(source.get_descriptor(), source.state_.device_);
        target->copy_from(source);
        return target;
    }

    static std::unique_ptr<Dense> create_from(Device target_device,
                                              Dense& source) {
        MatrixDescriptor descriptor = source.get_descriptor();
        std::unique_ptr<Dense> target =
            Dense::create(descriptor, target_device);
        target->copy_from(source);
        return target;
    }

    void copy_from(Dense& source) { state_.copy_from(source.state_); }

    dim<2> get_size() const { return state_.size_; };

    MatrixDescriptor get_descriptor() const {
        if (state_.storage_.index() ==
            static_cast<std::size_t>(MatrixFormat::STRIDED)) {
            StridedStorage S =
                std::get<RandLAPACK::StridedStorage>(state_.storage_);
            return MatrixDescriptor(
                StridedDescriptor(S.lead_dim_, S.layout_,
                                  static_cast<NumericType>(S.values_.index())));

        } else {
            throw std::runtime_error("Invalid storage type for dense matrix.");
        }
    }  // get_descriptor

    DenseFormat get_format() const {
        return static_cast<DenseFormat>(state_.storage_.index());
    }

    Device get_device() const { return state_.device_; }

   protected:
    Dense(dim<2> size, RandLAPACK::Layout layout, Device device,
          NumericType compute_precision)
        : state_(size, layout, device, compute_precision) {}

    Dense(std::string& filename, Device device, NumericType compute_precision) {
    }

    Dense(MatrixDescriptor descriptor, Device device)
        : state_(descriptor, device) {}

   private:
    DenseState state_;
};  // Dense

struct Sparse {
    friend class CsrView;

   private:
    SparseState state_;
};  // Sparse

// View objects for dynamic dispatch of the correct apply call in
// sketch_general. A view object will perform a shallow copy of the relevant
// metadata and data pointers from the Matrix object, and this view object will
// be passed to the apply call in sketch_general.

struct StridedView {
    dim<2> size_;
    Layout layout_ = RandLAPACK::Layout::COL_MAJOR;
    NumericPtrVariant values_;
    Device device_;
    size_t lead_dim_ = 0;
    StridedView(Dense& source)
        : size_(source.state_.size_),
          lead_dim_(std::get<StridedStorage>(source.state_.storage_).lead_dim_),
          layout_(std::get<StridedStorage>(source.state_.storage_).layout_),
          values_(std::get<StridedStorage>(source.state_.storage_).values_),
          device_(source.state_.device_) {}
};  // StridedView

struct CsrView {
    // dim<2> size_;
    // NumericPtrVariant values_;
    // Device device_;
    // size_t nnz_ = 0;
    CsrView(RandLAPACK::Sparse& source) {}
    // : size_(source.state_.size_),
    // nnz_(std::get<CsrStorage>(source.state_.storage_).nnz_),
    //   values_(std::get<CsrStorage>(source.state_.storage_).values_),
    //   device_(source.state_.device_) {}
};  // CsrView

}  // namespace RandLAPACK