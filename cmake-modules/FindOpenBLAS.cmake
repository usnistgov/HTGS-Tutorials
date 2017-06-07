
# - Try to find openblas
# Once done, this will define
#
#  LIB_OpenBLAS_FOUND - system has openblas
#  LIB_OpenBLAS_INCLUDE_DIRS - the openblas include directories
#  LIB_OpenBLAS_LIBRARIES - link these to use libuv
#
# Set the LIB_OpenBLAS_USE_STATIC variable to specify if static libraries should
# be preferred to shared ones.

if(NOT USE_BUNDLED_LIB_OpenBLAS)
    find_package(PkgConfig)
    if (PKG_CONFIG_FOUND)
        pkg_check_modules(PC_LIB_OpenBLAS QUIET libopenblas)
    endif()
else()
    set(PC_LIB_OpenBLAS_INCLUDEDIR)
    set(PC_LIB_OpenBLAS_INCLUDE_DIRS)
    set(PC_LIB_OpenBLAS_LIBDIR)
    set(PC_LIB_OpenBLAS_LIBRARY_DIRS)
    set(LIMIT_SEARCH NO_DEFAULT_PATH)
endif()

find_path(LIB_OpenBLAS_INCLUDE_DIR cblas.h
        HINTS ${PC_LIB_OpenBLAS_INCLUDEDIR} ${PC_LIB_OpenBLAS_INCLUDE_DIRS}
        ${LIMIT_SEARCH})

# If we're asked to use static linkage, add libopenblas.a as a preferred library name.
if(LIB_OpenBLAS_USE_STATIC)
    list(APPEND LIB_OpenBLAS_NAMES
            "${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif(LIB_OpenBLAS_USE_STATIC)

list(APPEND LIB_OpenBLAS_NAMES openblas)

find_library(LIB_OpenBLAS_LIBRARY NAMES ${LIB_OpenBLAS_NAMES}
        HINTS ${PC_LIB_OpenBLAS_LIBDIR} ${PC_LIB_OpenBLAS_LIBRARY_DIRS}
        ${LIMIT_SEARCH})

mark_as_advanced(LIB_OpenBLAS_INCLUDE_DIR LIB_OpenBLAS_LIBRARY)

if(PC_LIB_OpenBLAS_LIBRARIES)
    list(REMOVE_ITEM PC_LIB_OpenBLAS_LIBRARIES openblas)
endif()

set(LIB_OpenBLAS_LIBRARIES ${LIB_OpenBLAS_LIBRARY} ${PC_LIB_OpenBLAS_LIBRARIES})
set(LIB_OpenBLAS_INCLUDE_DIRS ${LIB_OpenBLAS_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)

# handle the QUIETLY and REQUIRED arguments and set LIB_OpenBLAS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(LibOpenBLAS DEFAULT_MSG
        LIB_OpenBLAS_LIBRARY LIB_OpenBLAS_INCLUDE_DIR)

mark_as_advanced(LIB_OpenBLAS_INCLUDE_DIR LIB_OpenBLAS_LIBRARY)