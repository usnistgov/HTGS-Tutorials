# - Find HTGS includes
#
# This module defines
#  HTGS_INCLUDE_DIR
#  HTGS_FOUND
#

FIND_PATH(HTGS_INCLUDE_DIR htgs/api/TaskGraph.hpp
        /usr/include
        /usr/local/include
        )

SET(HTGS_FOUND ON)

#    Check include files
IF (NOT HTGS_INCLUDE_DIR)
    SET(HTGS_FOUND OFF)
    MESSAGE(STATUS "Could not find HTGS includes. Turning HTGS_FOUND off")
ENDIF ()

IF (HTGS_FOUND)
    IF (NOT HTGS_FIND_QUIETLY)
        MESSAGE(STATUS "Found HTGS include: ${HTGS_INCLUDE_DIR}")
    ENDIF (NOT HTGS_FIND_QUIETLY)
ELSE (HTGS_FOUND)
    IF (HTGS_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find HTGS")
    ENDIF (HTGS_FIND_REQUIRED)
ENDIF (HTGS_FOUND)

MARK_AS_ADVANCED(HTGS_INCLUDE_DIR)
