#
# Test programs for the C interface. These need no JVM, which keeps most debugging away from
# JNI and from the JVM's own use of signals.
#

add_executable(llama-audio-smoke smoke.cpp)

target_link_libraries(llama-audio-smoke PRIVATE llama-audio)

target_compile_features(llama-audio-smoke PRIVATE cxx_std_17)

if (NOT MSVC)
    target_compile_options(llama-audio-smoke PRIVATE -Wall -Wextra)
endif()

# Registered with ctest only when a model to run it against is given, since it needs a real
# model file:
#   cmake -B build -DLLAMA_AUDIO_TEST_MODEL=/path/model.gguf \
#                  -DLLAMA_AUDIO_TEST_MMPROJ=/path/mmproj.gguf \
#                  -DLLAMA_AUDIO_TEST_AUDIO=/path/sample.wav
set(LLAMA_AUDIO_TEST_MODEL  "" CACHE FILEPATH "llama-audio: model file for the smoke test")
set(LLAMA_AUDIO_TEST_MMPROJ "" CACHE FILEPATH "llama-audio: projector file for the smoke test")
set(LLAMA_AUDIO_TEST_AUDIO  "" CACHE FILEPATH "llama-audio: audio file for the smoke test")

if (LLAMA_AUDIO_TEST_MODEL)
    set(smoke_args ${LLAMA_AUDIO_TEST_MODEL})
    if (LLAMA_AUDIO_TEST_MMPROJ)
        list(APPEND smoke_args --mmproj ${LLAMA_AUDIO_TEST_MMPROJ})
    endif()
    if (LLAMA_AUDIO_TEST_AUDIO)
        list(APPEND smoke_args --audio ${LLAMA_AUDIO_TEST_AUDIO})
    endif()

    add_test(NAME smoke COMMAND llama-audio-smoke ${smoke_args})
    set_tests_properties(smoke PROPERTIES TIMEOUT 600)
else()
    message(STATUS
        "llama-audio: LLAMA_AUDIO_TEST_MODEL is not set, so the smoke test is built but not "
        "registered with ctest")
endif()
