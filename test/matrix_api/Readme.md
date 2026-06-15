Matrix-API for RandLAPACK

The Matrix-API introduces the 3 following modules:

1. High-level matrix API: 
> Enables access to matrix objects without knowing the concrete types related
> to the backend device for which memory is owned and the type in which data are
> stored. This includes the Dense and Sparse structs and Matrix type defined as,
> variant<Dense, Sparse> to be used in function argument to switch between those
> two types of objects.

2. Internal API
> Includes functions on act on internal matrix storage objects such is
> DenseStorage / SparseStorage items. The correct kernel is selected via
> a lookup on a table of functions. A kernel is dispatched based
> matrix, device, and precision types.

3. Implementations to initialize, free and copy functions.
> Those kernels are registered to a Dense / Sparse registry.


To help pass data for matrix construction, we use the MatrixDescriptor struct.

Internals of a matrix are accessed via transformations to StridedView or CsrView.

files added:

    dispatch:
        context.hh:         Defines Context object used for communicating
                            runtime execution information such as backend, external
                            library handles and logging.                  
        storage_dispatch.hh Contains Dense/Sparse registries that store tables
                            containing available kernels. The suitable function is
                            accessed via the lookup() function.
    matrix:
        dimensions.hh       Defines the dim<2> struct for communicating matrix
                            dimensions.
        matrix.hh           Defines the following types:
                                - Dense / Sparse owning structs memory
                                - Matrix variant std::variant<Dense, Sparse>
                                - StridedView / CsrView non-owning structs used
                                  to extract matrix information in a function
                                  implemented in RandLAPACK
                                - StridedDescriptor / CsrDescriptor to communicate
                                  information to pass metadata.
                                - MatrixDescriptor variant
                                  std::variant<StridedDescriptor, CsrDescriptor>
    kernels
        cpu
            memory
                storage_cpu_kernel.hh     Implements copy CPU kernels.
        cuda
            memory
                storage_cuda_kernels.cuh  Implements copy CUDA kernels.
    memory
        csr_storage.hh      Implements initialize, free and copy functions for sparse objects.
        dense_storage.hh    Implements initialize, free and copy functions for dense objects.
        memory.hh           Implements allocators and and storage methods.
        type_macros.hh      Defines macros for instantiating template functions.
    precision
        precision.hh        Defines enum types and traits for converting between
                            enum and concrete types.
    status
        status.hh           Defines SUCCESS / FAILURE status values.
    test
        matrix_api
            test_cqrrpt_matrix.cc   Modifies test_cqrrpt.cc to use the matrix API.

You can run the tests defines in this file as:

path-to-randlapack-build/bin/RandLAPACK_tests --gtest_filter="TestCQRRPTMatrix.CQRRPT_full_rank_no_hqrrp"
path-to-randlapack-build/bin/RandLAPACK_tests --gtest_filter="TestCQRRPTMatrix.CQRRPT_low_rank_with_hqrrp"
path-to-randlapack-build/bin/RandLAPACK_tests --gtest_filter="TestCQRRPTMatrix.CQRRPT_low_rank_with_bqrrp"
path-to-randlapack-build/bin/RandLAPACK_tests --gtest_filter="TestCQRRPTMatrix.CQRRPT_bad_orth"
path-to-randlapack-build/bin/RandLAPACK_tests --gtest_filter="TestCQRRPTMatrix.CQRRPT_orthogonalization_mode_low_rank"