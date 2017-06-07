
# - Try to find libimageStitchingCuda
# Once done, this will define
#
#  LIB_ImageStitchingCuda_FOUND - system has the image stitching Cuda library
#  LIB_ImageStitchingCuda_INCLUDE_DIRS - the image stitching include directories
#  LIB_ImageStitchingCuda_LIBRARIES - link these to use image stitching
#
# Set the LIB_ImageStitchingCuda_USE_STATIC variable to specify if static libraries should
# be preferred to shared ones.

if(NOT USE_BUNDLED_LIB_ImageStitchingCuda)
    find_package(PkgConfig)
    if (PKG_CONFIG_FOUND)
        pkg_check_modules(PC_LIB_ImageStitchingCuda QUIET libimageStitchingCuda)
    endif()
else()
    set(PC_LIB_ImageStitchingCuda_INCLUDEDIR)
    set(PC_LIB_ImageStitchingCuda_INCLUDE_DIRS)
    set(PC_LIB_ImageStitchingCuda_LIBDIR)
    set(PC_LIB_ImageStitchingCuda_LIBRARY_DIRS)
    set(LIMIT_SEARCH NO_DEFAULT_PATH)
endif()

find_path(LIB_ImageStitchingCuda_INCLUDE_DIR fftw-image-tile.h
        HINTS ${PC_LIB_ImageStitchingCuda_INCLUDEDIR} ${PC_LIB_ImageStitchingCuda_INCLUDE_DIRS}
        ${LIMIT_SEARCH})

# If we're asked to use static linkage, add libimageStitchingCuda.a as a preferred library name.
if(LIB_ImageStitchingCuda_USE_STATIC)
    list(APPEND LIB_ImageStitchingCuda_NAMES
            "${CMAKE_STATIC_LIBRARY_PREFIX}imageStitchingCuda${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif(LIB_ImageStitchingCuda_USE_STATIC)

list(APPEND LIB_ImageStitchingCuda_NAMES imageStitchingCuda)

find_library(LIB_ImageStitchingCuda_LIBRARY NAMES ${LIB_ImageStitchingCuda_NAMES}
        HINTS ${PC_LIB_ImageStitchingCuda_LIBDIR} ${PC_LIB_ImageStitchingCuda_LIBRARY_DIRS}
        ${LIMIT_SEARCH})

mark_as_advanced(LIB_ImageStitchingCuda_INCLUDE_DIR LIB_ImageStitchingCuda_LIBRARY)

if(PC_LIB_ImageStitchingCuda_LIBRARIES)
    list(REMOVE_ITEM PC_LIB_ImageStitchingCuda_LIBRARIES imageStitchingCuda)
endif()

set(LIB_ImageStitchingCuda_LIBRARIES ${LIB_ImageStitchingCuda_LIBRARY} ${PC_LIB_ImageStitchingCuda_LIBRARIES})
set(LIB_ImageStitchingCuda_INCLUDE_DIRS ${LIB_ImageStitchingCuda_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)

# handle the QUIETLY and REQUIRED arguments and set LIB_ImageStitchingCuda_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(LibImageStitchingCuda DEFAULT_MSG
        LIB_ImageStitchingCuda_LIBRARY LIB_ImageStitchingCuda_INCLUDE_DIR)

mark_as_advanced(LIB_ImageStitchingCuda_INCLUDE_DIR LIB_ImageStitchingCuda_LIBRARY)