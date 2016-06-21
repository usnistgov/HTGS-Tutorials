add_definitions(-DUSE_CUDA)

set(TUTORIAL_UTILS_SRC
        ../../tutorial-utils/util-matrix.h
        ../../tutorial-utils/util-matrix.cpp
        ../../tutorial-utils/util-filesystem.cpp
        ../../tutorial-utils/util-filesystem.h
        ../../tutorial-utils/SimpleClock.h
        ../../tutorial-utils/enums/MatrixType.h
        ../../tutorial-utils/enums/MatrixType.cpp)

cuda_add_executable(matrixMul-in-memory-cuda-exec-pipeline
        tutorial5-matrix-mul-in-memory-cuda-exec-pipeline-main.cpp
        data/MatrixRequestData.h
        data/MatrixBlockData.h
        data/MatrixBlockMulData.h
        tasks/ReadMatrixTask.h
        tasks/OutputTask.h
        memory/MatrixMemoryRule.h
        memory/MatrixAllocator.h
        memory/CudaMatrixAllocator.h
        rules/MatrixLoadRule.h
        rules/MatrixDistributeRule.h
        rules/MatrixLoopRule.h
        rules/MatrixAccumulateRule.h
        rules/MatrixOutputRule.h
        rules/MatrixDecompositionRule.h
        tasks/MatrixMulBlkTask.h
        tasks/MatrixAccumTask.h
        tasks/OutputTask.h
        tasks/MatrixCopyInTask.h
        tasks/MatrixCopyOutTask.h
        ${TUTORIAL_UTILS_SRC})

target_link_libraries(matrixMul-in-memory-cuda-exec-pipeline openblas)
target_compile_definitions(matrixMul-in-memory-cuda-exec-pipeline PUBLIC -DUSE_CUDA)
cuda_add_cublas_to_target(matrixMul-in-memory-cuda-exec-pipeline)
target_link_libraries(matrixMul-in-memory-cuda-exec-pipeline ${CUDA_LIBRARIES})
target_link_libraries(matrixMul-in-memory-cuda-exec-pipeline cuda)
target_link_libraries(matrixMul-in-memory-cuda-exec-pipeline openblas)