
# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

link_directories(${OpenBLAS_LIBRARIES})

set(TUTORIAL_UTILS_SRC
        ../../tutorial-utils/util-matrix.h
        ../../tutorial-utils/util-matrix.cpp
        ../../tutorial-utils/util-filesystem.cpp
        ../../tutorial-utils/util-filesystem.h
        ../../tutorial-utils/SimpleClock.h
        ../../tutorial-utils/enums/MatrixType.h
        ../../tutorial-utils/enums/MatrixType.cpp)

add_executable(lud-in-memory-plasma
        tutorial6-lud-in-memory-openblas-main.cpp
        data/MatrixRequestData.h
        data/MatrixBlockData.h
        data/MatrixBlockMulData.h
        tasks/FactorUpperTask.h
        tasks/FactorLowerTask.h
        tasks/MatrixMulBlkTask.h
        data/MatrixFactorData.h
        rules/GausElimRuleLower.h
        rules/GausElimRuleUpper.h
        tasks/GausElimTask.h
        rules/MatrixMulRule.h
        rules/UpdateRule.h
        rules/GausElimRule.h
        ${TUTORIAL_UTILS_SRC} )

MESSAGE(STATUS "PLASMA LIBRARIES ================ ${PLASMA_LIBRARIES}")

target_link_libraries(lud-in-memory-plasma openblas)
target_link_libraries(lud-in-memory-plasma hwloc)
target_link_libraries(lud-in-memory-plasma ${PLASMA_LIBRARIES})

target_include_directories(lud-in-memory-plasma PUBLIC ${LAPACK_INCLUDES})
target_include_directories(lud-in-memory-plasma PUBLIC ${OpenBLAS_INCLUDE_DIR})
target_include_directories(lud-in-memory-plasma PUBLIC ${PLASMA_INCLUDES})
