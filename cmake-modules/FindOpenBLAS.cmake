
# - Try to find openblas
# Once done, this will define
#
#  LIBOPENBLAS_FOUND - system has openblas
#  LIBOPENBLAS_INCLUDE_DIRS - the openblas include directories
#  LIBOPENBLAS_LIBRARIES - link these to use libuv
#
# Set the LIBOPENBLAS_USE_STATIC variable to specify if static libraries should
# be preferred to shared ones.

if(NOT USE_BUNDLED_LIBOPENBLAS)
    find_package(PkgConfig)
    if (PKG_CONFIG_FOUND)
        pkg_check_modules(PC_LIBOPENBLAS QUIET libopenblas)
    endif()
else()
    set(PC_LIBOPENBLAS_INCLUDEDIR)
    set(PC_LIBOPENBLAS_INCLUDE_DIRS)
    set(PC_LIBOPENBLAS_LIBDIR)
    set(PC_LIBOPENBLAS_LIBRARY_DIRS)
    set(LIMIT_SEARCH NO_DEFAULT_PATH)
endif()

find_path(LIBOPENBLAS_INCLUDE_DIR cblas.h
        HINTS ${PC_LIBOPENBLAS_INCLUDEDIR} ${PC_LIBOPENBLAS_INCLUDE_DIRS}
        ${LIMIT_SEARCH})

# If we're asked to use static linkage, add libopenblas.a as a preferred library name.
if(LIBOPENBLAS_USE_STATIC)
    list(APPEND LIBOPENBLAS_NAMES
            "${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif(LIBOPENBLAS_USE_STATIC)

list(APPEND LIBOPENBLAS_NAMES openblas)

find_library(LIBOPENBLAS_LIBRARY NAMES ${LIBOPENBLAS_NAMES}
        HINTS ${PC_LIBOPENBLAS_LIBDIR} ${PC_LIBOPENBLAS_LIBRARY_DIRS}
        ${LIMIT_SEARCH})

mark_as_advanced(LIBOPENBLAS_INCLUDE_DIR LIBOPENBLAS_LIBRARY)

if(PC_LIBOPENBLAS_LIBRARIES)
    list(REMOVE_ITEM PC_LIBOPENBLAS_LIBRARIES openblas)
endif()

set(LIBOPENBLAS_LIBRARIES ${LIBOPENBLAS_LIBRARY} ${PC_LIBOPENBLAS_LIBRARIES})
set(LIBOPENBLAS_INCLUDE_DIRS ${LIBOPENBLAS_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)

# handle the QUIETLY and REQUIRED arguments and set LIBOPENBLAS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(LIBOpenBLAS DEFAULT_MSG
        LIBOPENBLAS_LIBRARY LIBOPENBLAS_INCLUDE_DIR)

mark_as_advanced(LIBOPENBLAS_INCLUDE_DIR LIBOPENBLAS_LIBRARY)