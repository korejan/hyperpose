INCLUDE(${CMAKE_SOURCE_DIR}/cmake/helpers.cmake)

FILE(GLOB_RECURSE USER_CODES ${CMAKE_SOURCE_DIR}/examples/user_codes/*.cpp)

foreach(USERCODE_FULL_PATH ${USER_CODES})
    GET_FILENAME_COMPONENT(USER_CODE_NAME ${USERCODE_FULL_PATH} NAME_WE)

    SET(USER_CODE_TAR user.${USER_CODE_NAME})

    MESSAGE(STATUS ">>> To build [USER CODES]: ${USERCODE_FULL_PATH} --> ${USER_CODE_TAR}")

    ADD_EXECUTABLE(${USER_CODE_TAR} ${USERCODE_FULL_PATH})
    TARGET_LINK_LIBRARIES(${USER_CODE_TAR} helpers hyperpose gflags)
    ADD_GLOBAL_DEPS(${USER_CODE_TAR})
endforeach()