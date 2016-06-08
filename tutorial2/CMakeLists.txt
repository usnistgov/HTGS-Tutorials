
set(TUTORIAL_UTILS_SRC
        ../tutorial-utils/util-matrix.h
        ../tutorial-utils/util-matrix.cpp
        ../tutorial-utils/util-filesystem.cpp
        ../tutorial-utils/util-filesystem.h
        ../tutorial-utils/SimpleClock.h
        ../tutorial-utils/enums/MatrixType.h
        ../tutorial-utils/enums/MatrixType.cpp)

add_executable(tutorial2 tutorial2.cpp ${TUTORIAL_UTILS_SRC} data/MatrixRequestData.h data/MatrixBlockData.h data/MatrixBlockMulData.h tasks/ReadMatrixTask.h memory/MatrixMemoryRule.h memory/MatrixAllocator.h rules/MatrixLoadRule.h tasks/HadamardProductTask.h)