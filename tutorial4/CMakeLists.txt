
if (CUDA_FOUND)
add_subdirectory(matrix-mul-cuda)
add_subdirectory(matrix-mul-in-memory-cuda)
add_subdirectory(matrix-mul-in-memory-cuda-alternate)
endif(CUDA_FOUND)