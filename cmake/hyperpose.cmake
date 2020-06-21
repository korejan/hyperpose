# Library Name
set(POSE_LIB_NAME hyperpose)

SET(CPU_PARALLEL_LIB "NONE" CACHE STRING "Select a which (CPU) parallel library to use for parallelizing some hyperpose loops.")
SET_PROPERTY(CACHE CPU_PARALLEL_LIB PROPERTY STRINGS CPP17 TBB PPL NONE)
MESSAGE(STATUS "CPU_PARALLEL_LIB='${CPU_PARALLEL_LIB}'")

# Dependencies(OpenCV & CUDA)
INCLUDE(cmake/cuda.cmake)
FIND_PACKAGE(OpenCV)
IF(CPU_PARALLEL_LIB STREQUAL "TBB")
	FIND_PACKAGE(TBB REQUIRED tbb)
ENDIF()

ADD_LIBRARY(
        ${POSE_LIB_NAME} # SHARED
        src/logging.cpp
        src/tensorrt.cpp
        src/paf.cpp
        src/data.cpp
        src/stream.cpp
        src/thread_pool.cpp
        src/pose_proposal.cpp
        src/human.cpp)
		
IF(NOT(CPU_PARALLEL_LIB STREQUAL "NONE"))
	TARGET_COMPILE_DEFINITIONS(${POSE_LIB_NAME} PRIVATE "HYPERPOSE_USE_${CPU_PARALLEL_LIB}_PARALLEL_FOR")
ENDIF()

TARGET_LINK_LIBRARIES(
        ${POSE_LIB_NAME}
        cudnn
        cudart
        nvinfer
        nvparsers
        nvonnxparser
        ${OpenCV_LIBS}
		${TBB_IMPORTED_TARGETS})

TARGET_INCLUDE_DIRECTORIES(${POSE_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${CUDA_RT}/include
        ${CUDA_RT}/include/crt)

SET_TARGET_PROPERTIES(${POSE_LIB_NAME} PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION})

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/cmake/configuration.h.in ${CMAKE_BINARY_DIR}/configuration.h)
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

ADD_GLOBAL_DEPS(${POSE_LIB_NAME})
