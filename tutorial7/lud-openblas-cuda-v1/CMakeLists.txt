
# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

#link_directories(${LIBOPENBLAS_LIBRARIES})

add_definitions(-DUSE_CUDA)


set(TUTORIAL_UTILS_SRC
        ../../tutorial-utils/util-matrix.h
        ../../tutorial-utils/util-matrix.cpp
        ../../tutorial-utils/util-filesystem.cpp
        ../../tutorial-utils/util-filesystem.h
        ../../tutorial-utils/SimpleClock.h
        ../../tutorial-utils/enums/MatrixType.h
        ../../tutorial-utils/enums/MatrixType.cpp
        ../../tutorial-utils/util-cuda.h
        ../../tutorial-utils/util-cuda.cpp)

cuda_add_executable(lud-in-memory-openblas-cuda-v1
        tutorial7-lud-in-memory-openblas-cuda-main.cpp
        ../common/data/MatrixRequestData.h
        ../common/data/MatrixBlockData.h
        ../common/data/MatrixBlockMulData.h
        ../common/data/MatrixFactorData.h
        ../common/data/MatrixFactorCudaData.h
        ../common/data/MatrixBlockMultiData.h
        ../common/data/MatrixBlockMulDataPreCopy.h
        tasks/MatrixCopyOutTask.h
        tasks/FactorUpperTask.h
        tasks/FactorLowerTask.h
        tasks/GausElimTask.h
        tasks/MatrixCopyInFactorTask.h
        tasks/MatrixCopyInGemmTask.h
        tasks/MatrixMulBlkTask.h
        rules/MatrixMulRule.h
        rules/UpdateRule.h
        rules/GausElimRule.h
        rules/GausElimRuleUpper.h
        rules/GausElimRuleLower.h
        rules/UpdateRuleMatMul.h
        memory/CudaMatrixAllocator.h
        memory/MatrixMemoryRule.h
        ${TUTORIAL_UTILS_SRC} )

#target_link_libraries(lud-in-memory-openblas-cuda-v1 openblas)
target_compile_definitions(lud-in-memory-openblas-cuda-v1 PUBLIC -DUSE_CUDA)
cuda_add_cublas_to_target(lud-in-memory-openblas-cuda-v1)
target_link_libraries(lud-in-memory-openblas-cuda-v1 ${CUDA_LIBRARIES})
target_link_libraries(lud-in-memory-openblas-cuda-v1 cuda)
target_link_libraries(lud-in-memory-openblas-cuda-v1 ${LIBOPENBLAS_LIBRARIES})
target_include_directories(lud-in-memory-openblas-cuda-v1 PUBLIC ${LIBOPENBLAS_INCLUDE_DIRS})
