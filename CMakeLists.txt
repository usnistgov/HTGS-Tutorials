
# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

cmake_minimum_required(VERSION 2.7)
project(htgs-tutorials)

SET(CMAKE_VERBOSE_MAKEFILE on)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter -Wno-reorder")

find_package(Threads REQUIRED)
find_package(HTGS REQUIRED)
find_package(OpenBLAS QUIET)
find_package(CUDA QUIET)
find_package(MAGMA QUIET)
find_package(PLASMA QUIET)
#find_package(MKL REQUIRED)

option(BUILD_STITCHING_TEST "Enables building the stitching test" OFF)


include_directories(${HTGS_INCLUDE_DIR})
link_libraries(${CMAKE_THREAD_LIBS_INIT})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--no-as-needed ${CMAKE_THREAD_LIBS_INIT}")

add_subdirectory(tutorial1)
add_subdirectory(tutorial2)
add_subdirectory(tutorial3)
add_subdirectory(tutorial4)
add_subdirectory(tutorial5)
add_subdirectory(tutorial6)
add_subdirectory(tutorial7)
add_subdirectory(tutorial8)


if (BUILD_STITCHING_TEST)
    add_subdirectory(image-stitching)
endif(BUILD_STITCHING_TEST)
