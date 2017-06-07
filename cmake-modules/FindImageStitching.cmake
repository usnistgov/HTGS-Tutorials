
# - Try to find libimageStitching
# Once done, this will define
#
#  LIB_ImageStitching_FOUND - system has the image stitching library
#  LIB_ImageStitching_INCLUDE_DIRS - the image stitching include directories
#  LIB_ImageStitching_LIBRARIES - link these to use libuv
#
# Set the LIB_ImageStitching_USE_STATIC variable to specify if static libraries should
# be preferred to shared ones.

if(NOT USE_BUNDLED_LIB_ImageStitching)
    find_package(PkgConfig)
    if (PKG_CONFIG_FOUND)
        pkg_check_modules(PC_LIB_ImageStitching QUIET libimageStitching)
    endif()
else()
    set(PC_LIB_ImageStitching_INCLUDEDIR)
    set(PC_LIB_ImageStitching_INCLUDE_DIRS)
    set(PC_LIB_ImageStitching_LIBDIR)
    set(PC_LIB_ImageStitching_LIBRARY_DIRS)
    set(LIMIT_SEARCH NO_DEFAULT_PATH)
endif()

find_path(LIB_ImageStitching_INCLUDE_DIR fftw-image-tile.h
        HINTS ${PC_LIB_ImageStitching_INCLUDEDIR} ${PC_LIB_ImageStitching_INCLUDE_DIRS}
        ${LIMIT_SEARCH})

# If we're asked to use static linkage, add libimageStitching.a as a preferred library name.
if(LIB_ImageStitching_USE_STATIC)
    list(APPEND LIB_ImageStitching_NAMES
            "${CMAKE_STATIC_LIBRARY_PREFIX}imageStitching${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif(LIB_ImageStitching_USE_STATIC)

list(APPEND LIB_ImageStitching_NAMES imageStitching)

find_library(LIB_ImageStitching_LIBRARY NAMES ${LIB_ImageStitching_NAMES}
        HINTS ${PC_LIB_ImageStitching_LIBDIR} ${PC_LIB_ImageStitching_LIBRARY_DIRS}
        ${LIMIT_SEARCH})

mark_as_advanced(LIB_ImageStitching_INCLUDE_DIR LIB_ImageStitching_LIBRARY)

if(PC_LIB_ImageStitching_LIBRARIES)
    list(REMOVE_ITEM PC_LIB_ImageStitching_LIBRARIES imageStitching)
endif()

set(LIB_ImageStitching_LIBRARIES ${LIB_ImageStitching_LIBRARY} ${PC_LIB_ImageStitching_LIBRARIES})
set(LIB_ImageStitching_INCLUDE_DIRS ${LIB_ImageStitching_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)

# handle the QUIETLY and REQUIRED arguments and set LIB_ImageStitching_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(LibImageStitching DEFAULT_MSG
        LIB_ImageStitching_LIBRARY LIB_ImageStitching_INCLUDE_DIR)

mark_as_advanced(LIB_ImageStitching_INCLUDE_DIR LIB_ImageStitching_LIBRARY)