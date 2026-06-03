#pragma once

namespace RandLAPACK {
namespace detail {

struct MatrixRegistry {
public:
    static const MatrixRegistry& get() {
        static const MatrixRegistry instance;
        return instance;
    }

    initialize_fn lookup_initialize(MatrixType matrix_type, Device dev, NumericType numeric_type) const;
    free_fn       lookup_free      (MatrixType matrix_type, Device dev, NumericType numeric_type) const;

private:
    MatrixRegistry() {
        struct InitEntries { MatrixType matrix_type; Device dev; NumericType numeric_type; initialize_fn fn; };
            static const InitEntries init_entries[] = {
            {MatrixType::DENSE, Device::CPU,  NumericType::FP64, &cpu::initialize_dense_storage<double>},
            {MatrixType::DENSE, Device::CPU,  NumericType::FP32, &cpu::initialize_dense_storage<float>},
            {MatrixType::DENSE, Device::CPU,  NumericType::FP16, &cpu::initialize_dense_storage<half>},
            {MatrixType::CSR,   Device::CPU,  NumericType::FP64, &cpu::initialize_csr_storage<double>},
            {MatrixType::CSR,   Device::CPU,  NumericType::FP32, &cpu::initialize_csr_storage<float>},
            {MatrixType::CSR,   Device::CPU,  NumericType::FP16, &cpu::initialize_csr_storage<half>},
            {MatrixType::DENSE, Device::CUDA, NumericType::FP64, &cuda::initialize_dense_storage<double>},
            {MatrixType::DENSE, Device::CUDA, NumericType::FP32, &cuda::initialize_dense_storage<float>},
            {MatrixType::DENSE, Device::CUDA, NumericType::FP16, &cuda::initialize_dense_storage<half>},
            {MatrixType::CSR,   Device::CUDA, NumericType::FP64, &cuda::initialize_csr_storage<double>},
            {MatrixType::CSR,   Device::CUDA, NumericType::FP32, &cuda::initialize_csr_storage<float>},
            {MatrixType::CSR,   Device::CUDA, NumericType::FP16, &cuda::initialize_csr_storage<half>},
        };
        for (auto& e : init_entries) initialize_table_[encode_init_free(e.matrix_type, e.dev, e.numeric_type)] = e.fn;

        struct FreeEntries { MatrixType matrix_type; Device dev; NumericType numeric_type; free_fn fn; };
        static const FreeEntries free_entries[] = {
            {MatrixType::DENSE, Device::CPU,  NumericType::FP64, &cpu::free_dense_storage<double>},
            {MatrixType::DENSE, Device::CPU,  NumericType::FP32, &cpu::free_dense_storage<float>},
            {MatrixType::DENSE, Device::CPU,  NumericType::FP16, &cpu::free_dense_storage<half>},
            {MatrixType::CSR,   Device::CPU,  NumericType::FP64, &cpu::free_csr_storage<double>},
            {MatrixType::CSR,   Device::CPU,  NumericType::FP32, &cpu::free_csr_storage<float>},
            {MatrixType::CSR,   Device::CPU,  NumericType::FP16, &cpu::free_csr_storage<half>},
            {MatrixType::DENSE, Device::CUDA, NumericType::FP64, &cuda::free_dense_storage<double>},
            {MatrixType::DENSE, Device::CUDA, NumericType::FP32, &cuda::free_dense_storage<float>},
            {MatrixType::DENSE, Device::CUDA, NumericType::FP16, &cuda::free_dense_storage<half>},
            {MatrixType::CSR,   Device::CUDA, NumericType::FP64, &cuda::free_csr_storage<double>},
            {MatrixType::CSR,   Device::CUDA, NumericType::FP32, &cuda::free_csr_storage<float>},
            {MatrixType::CSR,   Device::CUDA, NumericType::FP16, &cuda::free_csr_storage<half>},
        };
        for (auto& e : free_entries) free_table_[encode_init_free(e.matrix_type, e.dev, e.numeric_type)] = e.fn;

    }   // MatrixRegistry

    static constexpr std::size_t num_matrix_types_    = static_cast<std::size_t>(MatrixType::Count);
    static constexpr std::size_t num_devices_         = static_cast<std::size_t>(Device::Count);
    static constexpr std::size_t num_numeric_types_   = static_cast<std::size_t>(NumericType::Count);
    static constexpr std::size_t num_init_free_elems_ = num_matrix_types_ * num_devices_ * num_numeric_types_;

    std::array<initialize_fn, num_init_free_elems_> initialize_table_{};
    std::array<free_fn,       num_init_free_elems_> free_table_{};

    static constexpr std::size_t encode_init_free(MatrixType matrix_type, Device device, NumericType numeric_type) {
        return (static_cast<std::size_t>(matrix_type) * num_devices_
               + static_cast<std::size_t>(device)) * num_numeric_types_
               + static_cast<std::size_t>(numeric_type);
    }
};

}   // detail



}   // RandLAPACK