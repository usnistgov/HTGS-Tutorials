add_definitions(-DUSE_CUDA)

set(TUTORIAL_UTILS_SRC
        ../../tutorial-utils/util-matrix.h
        ../../tutorial-utils/util-matrix.cpp
        ../../tutorial-utils/util-filesystem.cpp
        ../../tutorial-utils/util-filesystem.h
        ../../tutorial-utils/SimpleClock.h
        ../../tutorial-utils/enums/MatrixType.h
        ../../tutorial-utils/enums/MatrixType.cpp
        ../../tutorial-utils/util-cuda.cpp
        ../../tutorial-utils/util-cuda.h)

cuda_add_executable(matrixMul-in-memory-cuda-alternate
        tutorial4-matrix-mul-in-memory-cuda-alternate-main.cpp
        data/MatrixRequestData.h
        data/MatrixBlockData.h
        data/MatrixBlockMulData.h
        memory/MatrixMemoryRule.h
        memory/MatrixAllocator.h
        memory/CudaMatrixAllocator.h
        rules/MatrixLoadRule.h
        rules/MatrixDistributeRule.h
        rules/MatrixAccumulateRule.h
        rules/MatrixOutputRule.h
        tasks/MatrixMulBlkTask.h
        tasks/MatrixMulBlkTask.cpp
        tasks/MatrixAccumTask.h
        tasks/MatrixAccumTask.cpp
        tasks/OutputTask.h
        tasks/OutputTask.cpp
        tasks/MatrixCopyInTask.h
        tasks/MatrixCopyInTask.cpp
        tasks/MatrixCopyOutTask.h
        tasks/MatrixCopyOutTask.cpp
        tasks/ReadMatrixTask.h
        tasks/ReadMatrixTask.cpp
        ${TUTORIAL_UTILS_SRC})

target_link_libraries(matrixMul-in-memory-cuda-alternate openblas)
target_compile_definitions(matrixMul-in-memory-cuda-alternate PUBLIC -DUSE_CUDA)
cuda_add_cublas_to_target(matrixMul-in-memory-cuda-alternate)
target_link_libraries(matrixMul-in-memory-cuda-alternate ${CUDA_LIBRARIES})
target_link_libraries(matrixMul-in-memory-cuda-alternate cuda)
target_link_libraries(matrixMul-in-memory-cuda-alternate openblas)