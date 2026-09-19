cmake_minimum_required(VERSION 3.16)

project(llama-audio-jni
    VERSION     0.1.0
    DESCRIPTION "Java wrapper for llama.cpp audio processing"
    LANGUAGES   C CXX)

# True when this repository is the top of the build, rather than part of a larger one.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LLAMA_AUDIO_STANDALONE ON)
else()
    set(LLAMA_AUDIO_STANDALONE OFF)
endif()

option(LLAMA_AUDIO_BUILD_TESTS "llama-audio: build the native test programs" ${LLAMA_AUDIO_STANDALONE})
option(LLAMA_AUDIO_BUILD_JNI   "llama-audio: build the shared library the JVM loads" ON)


# The llama.cpp version this project is built against. Read the README before changing it.
set(LLAMA_AUDIO_LLAMA_CPP_TAG "b10670" CACHE STRING
    "llama-audio: llama.cpp git tag to build against")


# Path to a llama.cpp checkout that is already on disk. When set, nothing is downloaded, which
# is what to use on a machine without access to github.com.
set(LLAMA_AUDIO_LLAMA_CPP_SOURCE "" CACHE PATH
    "llama-audio: use this llama.cpp checkout instead of downloading one")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

# Everything ends up inside a shared library that the JVM loads, so all of it has to be
# position independent, including the static libraries from llama.cpp.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#################### llama.cpp ####################

# Static libraries, so the shared library the JVM loads has no siblings to ship with it.
set(BUILD_SHARED_LIBS OFF)

# Only llama.h and mtmd.h are used. `common` is llama.cpp's helper code for its own command
# line programs and carries no stability promise, so it is left out.
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD     ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)

if (LLAMA_AUDIO_LLAMA_CPP_SOURCE)
    message(STATUS "llama-audio: using the llama.cpp checkout at ${LLAMA_AUDIO_LLAMA_CPP_SOURCE}")
    if (NOT EXISTS "${LLAMA_AUDIO_LLAMA_CPP_SOURCE}/CMakeLists.txt")
        message(FATAL_ERROR
            "LLAMA_AUDIO_LLAMA_CPP_SOURCE is set to '${LLAMA_AUDIO_LLAMA_CPP_SOURCE}', "
            "which does not look like a llama.cpp checkout (no CMakeLists.txt there).")
    endif()
    add_subdirectory(${LLAMA_AUDIO_LLAMA_CPP_SOURCE} ${CMAKE_BINARY_DIR}/llama.cpp)
    set(LLAMA_AUDIO_LLAMA_CPP_DIR ${LLAMA_AUDIO_LLAMA_CPP_SOURCE})
else()
    message(STATUS "llama-audio: downloading llama.cpp ${LLAMA_AUDIO_LLAMA_CPP_TAG}")
    include(FetchContent)
    FetchContent_Declare(llama_cpp
        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
        GIT_TAG        ${LLAMA_AUDIO_LLAMA_CPP_TAG}
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(llama_cpp)
    set(LLAMA_AUDIO_LLAMA_CPP_DIR ${llama_cpp_SOURCE_DIR})
endif()

# mtmd ships its headers next to its source rather than in an include directory.
set(LLAMA_AUDIO_MTMD_INCLUDE_DIR ${LLAMA_AUDIO_LLAMA_CPP_DIR}/tools/mtmd)

#################### this project ####################

add_subdirectory(native)

if (LLAMA_AUDIO_BUILD_JNI)
    add_subdirectory(java)
endif()

if (LLAMA_AUDIO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(native/test)
endif()

message(STATUS "llama-audio: configured ${PROJECT_NAME} ${PROJECT_VERSION} (${CMAKE_BUILD_TYPE})")

