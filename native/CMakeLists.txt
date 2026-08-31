#
# llama-audio: the C interface that language bindings are written against.
#
# Depends on llama.cpp's published interface only: llama (llama.h) for the model and mtmd
# (mtmd.h) for reading audio. It does not link llama-common.
#
# Built as a static library because it is meant to end up inside another shared library, such
# as the JNI one, with nothing to ship alongside it.
#

add_library(llama-audio STATIC
    src/llama_audio.cpp
    include/llama_audio.h)

target_include_directories(llama-audio
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${LLAMA_AUDIO_MTMD_INCLUDE_DIR})

target_link_libraries(llama-audio
    PUBLIC  llama
    PRIVATE mtmd)

target_compile_features(llama-audio PRIVATE cxx_std_17)

set_target_properties(llama-audio PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (MSVC)
    target_compile_options(llama-audio PRIVATE /W4)
else()
    target_compile_options(llama-audio PRIVATE -Wall -Wextra -Wpedantic)
endif()
