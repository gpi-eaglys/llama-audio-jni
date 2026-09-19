#
# llama-audio-jni: the shared library the JVM loads.
#
# This is the only piece that has to be a shared library. It carries the C interface and all of
# llama.cpp statically, so there is nothing to ship next to it.
#
# The library is written straight into the Maven resources tree, under the directory the loader
# in NativeLibrary.java looks in, so `mvn package` puts it in the jar without being told where
# to find it.
#

# Only the JNI headers are needed. The JVM provides the JNI functions themselves at run time,
# so nothing here links against libjvm.
set(LLAMA_AUDIO_JNI_INCLUDE_DIR "" CACHE PATH "llama-audio: the directory holding jni.h")

if (NOT LLAMA_AUDIO_JNI_INCLUDE_DIR)
    find_path(jni_h_dir NAMES jni.h HINTS ENV JAVA_HOME PATH_SUFFIXES include)
    if (NOT jni_h_dir)
        find_package(JNI QUIET)
        set(jni_h_dir "${JAVA_INCLUDE_PATH}")
    endif()
    set(LLAMA_AUDIO_JNI_INCLUDE_DIR "${jni_h_dir}" CACHE PATH "" FORCE)
endif()

if (NOT LLAMA_AUDIO_JNI_INCLUDE_DIR)
    message(FATAL_ERROR
        "llama-audio: could not find jni.h. Set JAVA_HOME to a JDK, or pass "
        "-DLLAMA_AUDIO_JNI_INCLUDE_DIR=/path/to/jdk/include.")
endif()

# jni_md.h sits in a directory named after the platform, next to jni.h.
file(GLOB_RECURSE jni_md_paths RELATIVE "${LLAMA_AUDIO_JNI_INCLUDE_DIR}"
     "${LLAMA_AUDIO_JNI_INCLUDE_DIR}/*/jni_md.h")
set(jni_include_dirs "${LLAMA_AUDIO_JNI_INCLUDE_DIR}")
foreach (jni_md_path IN LISTS jni_md_paths)
    get_filename_component(jni_md_dir "${jni_md_path}" DIRECTORY)
    list(APPEND jni_include_dirs "${LLAMA_AUDIO_JNI_INCLUDE_DIR}/${jni_md_dir}")
endforeach()

message(STATUS "llama-audio: JNI headers in ${LLAMA_AUDIO_JNI_INCLUDE_DIR}")

# The name of the directory the loader looks in, which has to agree with osName() and osArch()
# in NativeLibrary.java.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(jni_os linux)
elseif (APPLE)
    set(jni_os macos)
elseif (WIN32)
    set(jni_os windows)
else()
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" jni_os)
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^([Xx]86_64|AMD64|amd64)$")
    set(jni_arch x86_64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^([Aa]arch64|ARM64|arm64)$")
    set(jni_arch aarch64)
else()
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" jni_arch)
endif()

set(jni_output_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/main/resources/com/eaglys/llamaaudio/${jni_os}/${jni_arch})

add_library(llama-audio-jni SHARED src/main/native/llama_audio_jni.cpp)

target_link_libraries(llama-audio-jni PRIVATE llama-audio)

target_include_directories(llama-audio-jni PRIVATE ${jni_include_dirs})

target_compile_features(llama-audio-jni PRIVATE cxx_std_17)

# Everything except the JNI entry points stays inside this library. Without that, the llama.cpp
# symbols carried in it are exported too, and a JVM that also loads another llama.cpp binding
# would end up with two copies answering to the same names.
set_target_properties(llama-audio-jni PROPERTIES
    CXX_VISIBILITY_PRESET     hidden
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY  ${jni_output_dir}
    RUNTIME_OUTPUT_DIRECTORY  ${jni_output_dir})

# Visual Studio and Xcode put the build type in the path unless each one is set on its own.
foreach (config IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(llama-audio-jni PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${config} ${jni_output_dir}
        RUNTIME_OUTPUT_DIRECTORY_${config} ${jni_output_dir})
endforeach()

if (MSVC)
    target_compile_options(llama-audio-jni PRIVATE /W4)
else()
    target_compile_options(llama-audio-jni PRIVATE -Wall -Wextra -Wpedantic)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(llama-audio-jni PRIVATE -Wl,--exclude-libs,ALL)
endif()

message(STATUS "llama-audio: the JNI library is written to ${jni_output_dir}")
