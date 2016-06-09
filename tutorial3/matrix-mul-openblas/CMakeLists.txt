add_executable(matrix-mul-openblas
        tutorial3-openblas.cpp
        data/MatrixRequestData.h
        data/MatrixBlockData.h
        data/MatrixBlockMulData.h
        tasks/ReadMatrixTask.h
        tasks/OutputTask.h
        memory/MatrixMemoryRule.h
        memory/MatrixAllocator.h
        rules/MatrixLoadRule.h
        rules/MatrixDistributeRule.h
        rules/MatrixLoopRule.h
        rules/MatrixAccumulateRule.h
        tasks/MatrixMulBlkTask.h tasks/OutputTask.h
        ../matrix-ops/utils/util-matrix.cpp
        ../matrix-ops/utils/util-matrix.h)

target_link_libraries(matrix-mul-openblas openblas)