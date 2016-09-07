# - Find the PLASMA library
#
# Usage:
#   FIND_PACKAGE(PLASMA [REQUIRED] [QUIET] )
#
# It sets the following variables:
#   PLASMA_FOUND               ... true if PLASMA is found on the system
#   PLASMA_LIBRARIES           ... full path to PLASMA library
#   PLASMA_INCLUDES            ... PLASMA include directory
#
INCLUDE(FindPackageHandleStandardArgs)

SET(PLASMA_ROOT_DIR CACHE STRING
        "Root directory for PLASMA implementation")


IF(NOT PLASMA_ROOT_DIR)

    IF (ENV{PLASMADIR})
        SET(PLASMA_ROOT_DIR $ENV{PLASMADIR})
    ENDIF()

    IF (ENV{PLASMA_ROOT_DIR_DIR})
        SET(PLASMA_ROOT_DIR $ENV{PLASMA_ROOT_DIR})
    ENDIF()
ENDIF()

# Check if we can use PkgConfig
FIND_PACKAGE(PkgConfig)
set( ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${PLASMA_ROOT_DIR}/lib/pkgconfig" )

#Determine from PKG
IF(PKG_CONFIG_FOUND AND NOT PLASMA_ROOT_DIR)
    SET(PC_PLASMA_FOUND FALSE)
#    PKG_CHECK_MODULES( PC_PLASMA QUIET "plasma")
ENDIF()

IF(PC_PLASMA_FOUND)
    FOREACH(PC_LIB ${PC_PLASMA_LIBRARIES})
        FIND_LIBRARY(${PC_LIB}_LIBRARY NAMES ${PC_LIB} HINTS ${PC_PLASMA_LIBRARY_DIRS} )
        IF (NOT ${PC_LIB}_LIBRARY)
            MESSAGE(FATAL_ERROR "Something is wrong in your pkg-config file - lib ${PC_LIB} not found in ${PC_PLASMA_LIBRARY_DIRS}")
        ENDIF (NOT ${PC_LIB}_LIBRARY)
        LIST(APPEND PLASMA_LIB ${${PC_LIB}_LIBRARY})
    ENDFOREACH(PC_LIB)

    FIND_PATH(
            PLASMA_INCLUDES
            NAMES "plasma.h"
            PATHS
            ${PC_PLASMA_INCLUDE_DIRS}
            ${INCLUDE_INSTALL_DIR}
            /usr/include
            /usr/local/include
            /sw/include
            /opt/local/include
            DOC "PLASMA Include Directory"
    )

    FIND_PACKAGE_HANDLE_STANDARD_ARGS(PLASMA DEFAULT_MSG PLASMA_LIB)
    MARK_AS_ADVANCED(PLASMA_INCLUDES PLASMA_LIB)

ELSE(PC_PLASMA_FOUND)

    IF(PLASMA_ROOT_DIR)
        #find libs
        FIND_LIBRARY(
                PLASMA_LIB
                NAMES "plasma"
                PATHS ${PLASMA_ROOT_DIR}
                PATH_SUFFIXES "lib" "lib64" "lib/ia32" "lib/intel64"
                DOC "PLASMA Library"
                NO_DEFAULT_PATH
        )

        FIND_LIBRARY(
                QUARK_LIB
                NAMES "quark"
                PATHS ${PLASMA_ROOT_DIR}
                PATH_SUFFIXES "lib" "lib64" "lib/ia32" "lib/intel64"
                DOC "PLASMA Library"
                NO_DEFAULT_PATH
        )

        FIND_LIBRARY(
                COREBLAS_LIB
                NAMES "coreblas"
                PATHS ${PLASMA_ROOT_DIR}
                PATH_SUFFIXES "lib" "lib64" "lib/ia32" "lib/intel64"
                DOC "PLASMA Library"
                NO_DEFAULT_PATH
        )

        FIND_LIBRARY(
                COREBLASQW_LIB
                NAMES "coreblasqw"
                PATHS ${PLASMA_ROOT_DIR}
                PATH_SUFFIXES "lib" "lib64" "lib/ia32" "lib/intel64"
                DOC "PLASMA Library"
                NO_DEFAULT_PATH
        )


        FIND_PATH(
                PLASMA_INCLUDES
                NAMES "plasma.h"
                PATHS ${PLASMA_ROOT_DIR}
                PATH_SUFFIXES "include"
                DOC "PLASMA Include Directory"
                NO_DEFAULT_PATH
        )
    ELSE()
        FIND_LIBRARY(
                PLASMA_LIB
                NAMES "plasma" "quark" "coreblasqw" "coreblas"
                PATHS
                ${PC_PLASMA_LIBRARY_DIRS}
                ${LIB_INSTALL_DIR}
                /usr/lib64
                /usr/lib
                /usr/local/lib64
                /usr/local/lib
                /sw/lib
                /opt/local/lib
                DOC "PLASMA Library"
        )

        FIND_LIBRARY(
                QUARK_LIB
                NAMES "quark"
                PATHS
                ${PC_PLASMA_LIBRARY_DIRS}
                ${LIB_INSTALL_DIR}
                /usr/lib64
                /usr/lib
                /usr/local/lib64
                /usr/local/lib
                /sw/lib
                /opt/local/lib
                DOC "PLASMA Library"
        )

        FIND_LIBRARY(
                COREBLAS_LIB
                NAMES "coreblas"
                PATHS
                ${PC_PLASMA_LIBRARY_DIRS}
                ${LIB_INSTALL_DIR}
                /usr/lib64
                /usr/lib
                /usr/local/lib64
                /usr/local/lib
                /sw/lib
                /opt/local/lib
                DOC "PLASMA Library"
        )

        FIND_LIBRARY(
                COREBLASQW_LIB
                NAMES "coreblasqw"
                PATHS
                ${PC_PLASMA_LIBRARY_DIRS}
                ${LIB_INSTALL_DIR}
                /usr/lib64
                /usr/lib
                /usr/local/lib64
                /usr/local/lib
                /sw/lib
                /opt/local/lib
                DOC "PLASMA Library"
        )

        FIND_PATH(
                PLASMA_INCLUDES
                NAMES "plasma.h"
                PATHS
                ${PC_PLASMA_INCLUDE_DIRS}
                ${INCLUDE_INSTALL_DIR}
                /usr/include
                /usr/local/include
                /sw/include
                /opt/local/include
                DOC "PLASMA Include Directory"
                PATH_SUFFIXES
                lapacke
        )
    ENDIF(PLASMA_ROOT_DIR)


ENDIF(PC_PLASMA_FOUND)

IF(PC_PLASMA_FOUND OR (PLASMA_LIB AND COREBLAS_LIB AND COREBLASQW_LIB AND QUARK_LIB))
    SET(PLASMA_LIBRARIES ${PLASMA_LIB} ${QUARK_LIB} ${COREBLASQW_LIB}  ${COREBLAS_LIB})
ENDIF()
IF(PLASMA_INCLUDES)
    SET(PLASMA_INCLUDE_DIR ${PLASMA_INCLUDES})
ENDIF()


FIND_PACKAGE_HANDLE_STANDARD_ARGS(PLASMA DEFAULT_MSG
        PLASMA_INCLUDE_DIR PLASMA_LIBRARIES)

MARK_AS_ADVANCED(
        PLASMA_ROOT_DIR
        PLASMA_INCLUDES
        PLASMA_LIBRARIES
        PLASMA_LIB
        PLASMA_INCLUDES
        PLASMA_LIB)
