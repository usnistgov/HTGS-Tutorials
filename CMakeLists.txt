cmake_minimum_required(VERSION 2.7)
project(htgs-tutorials)

SET(CMAKE_VERBOSE_MAKEFILE on)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")

find_package (Threads REQUIRED)
find_package (HTGS REQUIRED)
find_package (OpenBLAS QUIET)


include_directories(${HTGS_INCLUDE_DIR})
include_directories(${OpenBLAS_INCLUDE_DIR})

link_directories(${OpenBLAS_LIBRARIES})
link_libraries(${CMAKE_THREAD_LIBS_INIT})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--no-as-needed ${CMAKE_THREAD_LIBS_INIT}")



add_subdirectory(tutorial1)
#add_subdirectory(tutorial2)
#add_subdirectory(tutorial2-a)
#add_subdirectory(tutorial3)
#add_subdirectory(tutorial3-openblas)
#add_subdirectory(tutorial3-mmap)
#add_subdirectory(tutorial3-mmap-openblas)
#add_subdirectory(tutorial3-mmap-cuda)
#add_subdirectory(tutorial3-mmap-openblas-alternate)
