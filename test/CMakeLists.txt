message(STATUS "Checking for regression tests ...")
find_package(GTest)

set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)

    set(RandLAPACK_test_srcs
        comps/test_determiter.cc
        comps/test_orth.cc
        comps/test_qb.cc
        comps/test_preconditioners.cc
        comps/test_rf.cc
        comps/test_syrf.cc
        comps/test_rpchol.cc
        drivers/test_krill.cc
        drivers/test_rsvd.cc
        drivers/test_cqrrpt.cc
        drivers/test_cqrrt.cc
        drivers/test_bqrrp.cc
        drivers/test_revd2.cc
        drivers/test_hqrrp.cc
        drivers/test_abrik.cc
        drivers/test_orth_linop.cc
        matrix_api/test_cqrrpt_matrix.cc
        matrix_api/test_storage_dispatch.cc
        misc/test_util.cc
        misc/test_pdkernels.cc
        linops/test_linops.cc
        linops/test_linop_unified.cc
        misc/test_gen.cc
        misc/test_memory_tracker.cc
        linops/test_linop_block_views.cc
    )
    
    # Create non-CUDA test executable
    add_executable(RandLAPACK_tests ${RandLAPACK_test_srcs})
    target_link_libraries(RandLAPACK_tests RandLAPACK GTest::GTest GTest::Main)
    gtest_discover_tests(RandLAPACK_tests)

    # Check if CUDA is required and setup CUDA-specific tests
    if (RequireCUDA AND CUDAToolkit_FOUND)
        set(RandLAPACK_test_cu_srcs
            comps/test_util_gpu.cu
            drivers/test_cqrrpt_gpu.cu
            drivers/test_bqrrp_gpu.cu
            matrix_api/test_storage_dispatch_gpu.cu
        )

        # Create CUDA test executables
        set_source_files_properties(${RandLAPACK_test_cu_srcs} PROPERTIES LANGUAGE CUDA)
        add_executable(RandLAPACK_tests_gpu ${RandLAPACK_test_cu_srcs})
        target_link_libraries(RandLAPACK_tests_gpu RandLAPACK GTest::GTest GTest::Main CUDA::cusolver)

        # Discover CUDA tests
        gtest_discover_tests(RandLAPACK_tests_gpu)

        # RandBLAS CUDA kernel tests
        set(RANDBLAS_CUDA_KERNEL_SRCS
            ../RandBLAS/test/kernels/test_sparse_skops.cc
        )
        set_source_files_properties(${RANDBLAS_CUDA_KERNEL_SRCS} PROPERTIES LANGUAGE CUDA)
        add_executable(randblas_cuda_kernel_tests ${RANDBLAS_CUDA_KERNEL_SRCS})
        target_link_libraries(randblas_cuda_kernel_tests RandLAPACK GTest::GTest GTest::Main CUDA::cudart)
        gtest_discover_tests(randblas_cuda_kernel_tests)

        # RandBLAS sparse sketch construction benchmark (GPU vs CPU)
        set(RANDBLAS_SPARSE_BENCH_SRC ../RandBLAS/test/kernels/bench_sparse_skops_gpu.cc)
        set_source_files_properties(${RANDBLAS_SPARSE_BENCH_SRC} PROPERTIES LANGUAGE CUDA)
        add_executable(bench_sparse_skops_gpu ${RANDBLAS_SPARSE_BENCH_SRC})
        target_link_libraries(bench_sparse_skops_gpu RandLAPACK CUDA::cudart CUDA::cusparse)
    endif()

    # Create additional tests
    add_executable(test_pcgls comps/test_pcgls.cc)
    target_link_libraries(test_pcgls RandLAPACK GTest::GTest)
    add_test(NAME test_pcgls COMMAND test_pcgls 15 100)
else()
    message(WARNING "GTest not found - tests will be disabled. Install GTest to enable regression tests.")
endif()

message(STATUS "Checking for regression tests ... ${tmp}")