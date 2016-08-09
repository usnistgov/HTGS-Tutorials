# - Find the MAGMA library
#
# Usage:
#   FIND_PACKAGE(MAGMA [REQUIRED] [QUIET] )
#
# It sets the following variables:
#   MAGMA_FOUND               ... true if MAGMA is found on the system
#   MAGMA_LIBRARIES           ... full path to MAGMA library
#   MAGMA_INCLUDES            ... MAGMA include directory
#

SET(MAGMA_ROOT_DIR CACHE STRING
        "Root directory for MAGMA implementation")

IF(NOT MAGMA_ROOT_DIR)

    IF (ENV{MAGMADIR})
        SET(MAGMA_ROOT_DIR $ENV{MAGMADIR})
    ENDIF()

    IF (ENV{MAGMA_ROOT_DIR_DIR})
        SET(MAGMA_ROOT_DIR $ENV{MAGMA_ROOT_DIR})
    ENDIF()
ENDIF()

# Check if we can use PkgConfig
FIND_PACKAGE(PkgConfig)

#Determine from PKG
IF(PKG_CONFIG_FOUND AND NOT MAGMA_ROOT_DIR)
    PKG_CHECK_MODULES( PC_MAGMA QUIET "magma")
ENDIF()

IF(PC_MAGMA_FOUND)
    FOREACH(PC_LIB ${PC_MAGMA_LIBRARIES})
        FIND_LIBRARY(${PC_LIB}_LIBRARY NAMES ${PC_LIB} HINTS ${PC_MAGMA_LIBRARY_DIRS} )
        IF (NOT ${PC_LIB}_LIBRARY)
            MESSAGE(FATAL_ERROR "Something is wrong in your pkg-config file - lib ${PC_LIB} not found in ${PC_MAGMA_LIBRARY_DIRS}")
        ENDIF (NOT ${PC_LIB}_LIBRARY)
        LIST(APPEND MAGMA_LIB ${${PC_LIB}_LIBRARY})
    ENDFOREACH(PC_LIB)

    FIND_PATH(
            MAGMA_INCLUDES
            NAMES "magma.h"
            PATHS
            ${PC_MAGMA_INCLUDE_DIRS}
            ${INCLUDE_INSTALL_DIR}
            /usr/include
            /usr/local/include
            /sw/include
            /opt/local/include
            DOC "MAGMA Include Directory"
    )

    FIND_PACKAGE_HANDLE_STANDARD_ARGS(MAGMA DEFAULT_MSG MAGMA_LIB)
    MARK_AS_ADVANCED(MAGMA_INCLUDES MAGMA_LIB)

ELSE(PC_MAGMA_FOUND)

    IF(MAGMA_ROOT_DIR)
        #find libs
        FIND_LIBRARY(
                MAGMA_LIB
                NAMES "magma" "MAGMA"
                PATHS ${MAGMA_ROOT_DIR}
                PATH_SUFFIXES "lib" "lib64" "lib/ia32" "lib/intel64"
                DOC "MAGMA Library"
                NO_DEFAULT_PATH
        )

        FIND_PATH(
                MAGMA_INCLUDES
                NAMES "magma.h"
                PATHS ${MAGMA_ROOT_DIR}
                PATH_SUFFIXES "include"
                DOC "MAGMA Include Directory"
                NO_DEFAULT_PATH
        )
    ELSE()
        FIND_LIBRARY(
                MAGMA_LIB
                NAMES "magma"
                PATHS
                ${PC_MAGMA_LIBRARY_DIRS}
                ${LIB_INSTALL_DIR}
                /usr/lib64
                /usr/lib
                /usr/local/lib64
                /usr/local/lib
                /sw/lib
                /opt/local/lib
                DOC "MAGMA Library"
        )

        FIND_PATH(
                MAGMA_INCLUDES
                NAMES "magma.h"
                PATHS
                ${PC_MAGMA_INCLUDE_DIRS}
                ${INCLUDE_INSTALL_DIR}
                /usr/include
                /usr/local/include
                /sw/include
                /opt/local/include
                DOC "MAGMA Include Directory"
                PATH_SUFFIXES
                lapacke
        )
    ENDIF(MAGMA_ROOT_DIR)
ENDIF(PC_MAGMA_FOUND)

IF(PC_MAGMA_FOUND OR (MAGMA_LIB))
    SET(MAGMA_LIBRARIES ${MAGMA_LIB})
ENDIF()
IF(MAGMA_INCLUDES)
    SET(MAGMA_INCLUDE_DIR ${MAGMA_INCLUDES})
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MAGMA DEFAULT_MSG
        MAGMA_INCLUDE_DIR MAGMA_LIBRARIES)

MARK_AS_ADVANCED(
        MAGMA_ROOT_DIR
        MAGMA_INCLUDES
        MAGMA_LIBRARIES
        MAGMA_LIB
        MAGMA_INCLUDES
        MAGMA_LIB)
