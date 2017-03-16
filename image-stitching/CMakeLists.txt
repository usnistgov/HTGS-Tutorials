find_package(TIFF REQUIRED)
find_package(FFTW REQUIRED)

set(PARAMS_SRC
        tclap/Arg.h
        tclap/ArgException.h
        tclap/ArgTraits.h
        tclap/CmdLine.h
        tclap/CmdLineInterface.h
        tclap/CmdLineOutput.h
        tclap/Constraint.h
        tclap/DocBookOutput.h
        tclap/HelpVisitor.h
        tclap/IgnoreRestVisitor.h
        tclap/MultiArg.h
        tclap/MultiSwitchArg.h
        tclap/OptionalUnlabeledTracker.h
        tclap/StandardTraits.h
        tclap/StdOutput.h
        tclap/SwitchArg.h
        tclap/UnlabeledMultiArg.h
        tclap/UnlabeledValueArg.h
        tclap/ValueArg.h
        tclap/ValuesConstraint.h
        tclap/VersionVisitor.h
        tclap/Visitor.h
        tclap/XorHandler.h
        tclap/ZshCompletionOutput.h
        StitchingParams.cpp
        StitchingParams.h
        ParamValidators.h
        util-filesystem.h
        )

set(IMAGE_STITCHING_SRC
        cpu/data/FFTData.h
        cpu/data/PCIAMData.h
        cpu/memory/FFTMemoryRule.h
        cpu/memory/FFTWMemory.h
        cpu/memory/ReadMemory.h
        cpu/rules/GridDecompositionRule.h
        cpu/rules/StitchingRule.h
        cpu/tasks/FFTTask.cpp
        cpu/tasks/FFTTask.h
        cpu/tasks/PCIAMTask.cpp
        cpu/tasks/PCIAMTask.h
        cpu/tasks/ReadTask.cpp
        cpu/tasks/ReadTask.h
        )

set(STITCHING_CPP_LIBRARIES imageStitching ${FFTW_LIBRARIES} ${TIFF_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

## Edit Here to specify location of image stitching library
include_directories(/home/tjb3/local/include)
include_directories(.)
link_directories(/home/tjb3/local/lib)

add_executable(testCPUStitching testCPUStitching.cpp ${IMAGE_STITCHING_SRC} ${PARAMS_SRC})
add_executable(testCPUStitchingWithMemory testCPUStitchingWithMemory.cpp ${IMAGE_STITCHING_SRC} ${PARAMS_SRC})
add_executable(testStitchingAPI testStitchingAPI.cpp ${IMAGE_STITCHING_SRC})

target_link_libraries(testCPUStitching ${STITCHING_CPP_LIBRARIES})
target_link_libraries(testCPUStitchingWithMemory ${STITCHING_CPP_LIBRARIES})
target_link_libraries(testStitchingAPI ${STITCHING_CPP_LIBRARIES})

if (CUDA_FOUND)
    add_definitions(-DUSE_CUDA)

    set(IMAGE_STITCHING_CUDA_SRC
            cuda/data/CCFData.h
            cuda/data/FFTData.h
            cuda/data/PCIAMData.h
            cuda/memory/CudaMemory.h
            cuda/rules/GridDecompositionRule.h
            cuda/rules/StitchingRule.h
            cuda/tasks/CCFTask.cpp
            cuda/tasks/CCFTask.h
            cuda/tasks/FFTTask.cpp
            cuda/tasks/FFTTask.h
            cuda/tasks/PCIAMTask.cpp
            cuda/tasks/PCIAMTask.h
            cuda/tasks/ReadTask.cpp
            cuda/tasks/ReadTask.h
            )
    set(STITCHING_CUDA_LIBRARIES imageStitchingCuda ${FFTW_LIBRARIES} ${TIFF_LIBRARIES} -lcuda ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

    cuda_add_executable(testCudaStitching testCudaStitching.cpp ${IMAGE_STITCHING_CUDA_SRC})
    cuda_add_executable(testCudaStitchingExecutionPipeline testCudaStitchingExecutionPipeline.cpp ${IMAGE_STITCHING_CUDA_SRC})

    target_link_libraries(testCudaStitching ${STITCHING_CUDA_LIBRARIES})
    target_link_libraries(testCudaStitchingExecutionPipeline ${STITCHING_CUDA_LIBRARIES})

    cuda_add_cufft_to_target(testCudaStitching)
    cuda_add_cufft_to_target(testCudaStitchingExecutionPipeline)


endif (CUDA_FOUND)