# - Find Intel MKL
# Find the MKL libraries
#
# Options:
#
#
# This module defines the following variables:
#
#   MKL_FOUND            : True if MKL_INCLUDE_DIR are found
#   MKL_INCLUDE_DIR      : where to find mkl.h, etc.
#   MKL_INCLUDE_DIRS     : set when MKL_INCLUDE_DIR found
#   MKL_LIBRARIES        : the library to link against.


include(FindPackageHandleStandardArgs)

set(INTEL_ROOT "/opt/intel" CACHE PATH "Folder contains intel libs")
set(MKL_ROOT $ENV{MKLROOT} CACHE PATH "Folder contains MKL")

# Find include dir
find_path(MKL_INCLUDE_DIR mkl.h
    PATHS ${MKL_ROOT}/include)

# Find libraries
# Handle suffix
set(_MKL_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

if(MKL_STATAIC)
   set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
else()
   set(CMAKE_FIND_LIBRARY_SUFFIXES .so)
endif()

#find_library(MKL_LIBRARY mkl_rt
#        PATHS ${MKL_ROOT}/lib/intel64/)

find_library(MKL_INTEL_LIBRARY mkl_intel_lp64
        PATHS ${MKL_ROOT}/lib/intel64/)

find_library(MKL_THREAD_LIBRARY mkl_gnu_thread
        PATHS ${MKL_ROOT}/lib/intel64/)

find_library(MKL_CORE_LIBRARY mkl_core
        PATHS ${MKL_ROOT}/lib/intel64/)

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_MKL_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})

find_package_handle_standard_args(MKL DEFAULT_MSG
    MKL_INCLUDE_DIR MKL_INTEL_LIBRARY MKL_THREAD_LIBRARY MKL_CORE_LIBRARY)

if(MKL_FOUND)
    set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
    set(MKL_LIBRARIES ${MKL_INTEL_LIBRARY} ${MKL_THREAD_LIBRARY} ${MKL_CORE_LIBRARY})
endif()
