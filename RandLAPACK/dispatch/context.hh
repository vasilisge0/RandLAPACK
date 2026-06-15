#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include <cstdint>
#include <variant>

#include "device.hh"
#include "status.hh"

namespace RandLAPACK {

struct CudaContext {
    CudaContext() {
        cudaError_t cuda_status = cudaStreamCreate(&stream_);
        if (cuda_status != cudaSuccess) {
            perror("Call to cudaStreamCreate failed.\n");
            exit(EXIT_FAILURE);
        }

        cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            perror("Call to cublasCreate failed.\n");
            exit(EXIT_FAILURE);
        }

        cusparseStatus_t cusparse_status = cusparseCreate(&cusparse_handle_);
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
            perror("Call to cusparseCreate failed.\n");
            exit(EXIT_FAILURE);
        }
    }

    CudaContext(cudaStream_t stream, cublasHandle_t cublas_handle,
                cusparseHandle_t cusparse_handle, bool enable_tf32)
        : stream_(stream),
          cublas_handle_(cublas_handle),
          cusparse_handle_(cusparse_handle),
          use_tf32_(enable_tf32) {
        cudaError_t cuda_status = cudaStreamCreate(&stream_);
        if (cuda_status != cudaSuccess) {
            perror("Call to cudaStreamCreate failed.\n");
            exit(EXIT_FAILURE);
        }

        cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            perror("Call to cublasCreate failed.\n");
            exit(EXIT_FAILURE);
        }

        cusparseStatus_t cusparse_status = cusparseCreate(&cusparse_handle_);
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
            perror("Call to cusparseCreate failed.\n");
            exit(EXIT_FAILURE);
        }

        if (enable_tf32) {
            cublasStatus_t cublas_status =
                cublasSetMathMode(cublas_handle_, CUBLAS_TF32_TENSOR_OP_MATH);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                perror("Call to cublasSetMathMode failed.\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    Status enable_tf32_math() {
        cublasStatus_t cublas_status =
            cublasSetMathMode(cublas_handle_, CUBLAS_TF32_TENSOR_OP_MATH);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            perror("Call to cublasSetMathMode failed\n");
            return FAILURE;
        }
        use_tf32_ = true;
        return SUCCESS;
    }

    Status disable_tf32_math() {
        cublasStatus_t cublas_status =
            cublasSetMathMode(cublas_handle_, CUBLAS_DEFAULT_MATH);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            perror("Call to cublasSetMathMode failed\n");
            return FAILURE;
        }
        use_tf32_ = false;
        return SUCCESS;
    }

    cudaStream_t stream_;
    cublasHandle_t cublas_handle_;
    cusparseHandle_t cusparse_handle_;
    bool use_tf32_ = false;  // specifics for tensorcore format
};

struct CpuContext {
    CpuContext() {}

    int num_threads_ = 1;
};

using DeviceContext = std::variant<CpuContext, CudaContext>;

struct Context {
public:
    Context() = delete;

    Context(Device device) {
        if (device == Device::CPU) {
            device_context_ = CpuContext();
        } else {
            device_context_ = CudaContext();
        }
    }

    ~Context() {
        std::visit(
            [](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, CudaContext>) {
                    cudaStreamDestroy(arg.stream_);
                    cublasDestroy(arg.cublas_handle_);
                    cusparseDestroy(arg.cusparse_handle_);
                }
            },
            device_context_);
    }

    Device get_device() {
        return std::visit(
            [](auto&& arg) -> Device {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, CpuContext>) {
                    return Device::CPU;
                } else if constexpr (std::is_same_v<T, CudaContext>) {
                    return Device::CUDA;
                }
            },
            device_context_);
    }

private:
    DeviceContext device_context_;
};

}  // namespace RandLAPACK