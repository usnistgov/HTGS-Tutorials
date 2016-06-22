set(TUTORIAL_UTILS_SRC
        ../../tutorial-utils/util-matrix.h
        ../../tutorial-utils/util-matrix.cpp
        ../../tutorial-utils/util-filesystem.cpp
        ../../tutorial-utils/util-filesystem.h
        ../../tutorial-utils/SimpleClock.h
        ../../tutorial-utils/enums/MatrixType.h
        ../../tutorial-utils/enums/MatrixType.cpp)

add_executable(matrixMul-in-memory-openblas
        tutorial3-matrix-mul-in-memory-openblas-main.cpp
        data/MatrixRequestData.h
        data/MatrixBlockData.h
        data/MatrixBlockMulData.h
        tasks/ReadMatrixTask.h
        tasks/OutputTask.h
        memory/MatrixMemoryRule.h
        memory/MatrixAllocator.h
        rules/MatrixLoadRule.h
        rules/MatrixDistributeRule.h
        rules/MatrixAccumulateRule.h
        rules/MatrixOutputRule.h
        tasks/MatrixMulBlkTask.h
        tasks/MatrixAccumTask.h
        tasks/OutputTask.h
        ${TUTORIAL_UTILS_SRC})

target_link_libraries(matrixMul-in-memory-openblas openblas)