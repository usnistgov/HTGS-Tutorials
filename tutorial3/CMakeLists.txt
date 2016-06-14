add_subdirectory(matrix-mul)
add_subdirectory(matrix-mul-mmap)

if (OpenBLAS_FOUND)
    add_subdirectory(matrix-mul-openblas)
    add_subdirectory(matrix-mul-mmap-openblas)
endif(OpenBLAS_FOUND)