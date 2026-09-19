/*
 * The JNI layer over the C interface in native/include/llama_audio.h.
 *
 * It converts types and nothing else: no model handling, no prompt building, no policy of its
 * own. Anything a caller has to reason about belongs on one side or the other of this file.
 *
 * Text crosses as UTF-8 bytes rather than through JNI's string calls, which speak modified
 * UTF-8. The two differ on any character outside the basic multilingual plane, and llama.cpp
 * reads and writes plain UTF-8.
 */

#include <jni.h>

#include "llama_audio.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr const char * EXCEPTION_CLASS = "com/eaglys/llamaaudio/LlamaAudioException";
constexpr const char * EXCEPTION_FACTORY_SIGNATURE =
        "(I[B)Lcom/eaglys/llamaaudio/LlamaAudioException;";

la_context * as_context(jlong handle) {
    return reinterpret_cast<la_context *>(static_cast<intptr_t>(handle));
}

/** Copies bytes into a new Java array. Returns null with an exception pending when out of memory. */
jbyteArray to_byte_array(JNIEnv * env, const char * data, size_t length) {
    jbyteArray array = env->NewByteArray(static_cast<jsize>(length));
    if (array == nullptr) {
        return nullptr;
    }
    if (length > 0) {
        env->SetByteArrayRegion(array, 0, static_cast<jsize>(length),
                                reinterpret_cast<const jbyte *>(data));
    }
    return array;
}

/** Copies a Java byte array into a string. Returns false when a JNI call left an exception pending. */
bool to_string(JNIEnv * env, jbyteArray bytes, std::string & out) {
    const jsize length = env->GetArrayLength(bytes);
    out.assign(static_cast<size_t>(length), '\0');
    if (length > 0) {
        env->GetByteArrayRegion(bytes, 0, length, reinterpret_cast<jbyte *>(&out[0]));
    }
    return env->ExceptionCheck() == JNI_FALSE;
}

/** Throws LlamaAudioException carrying the status and the message for the call that just failed. */
void throw_status(JNIEnv * env, la_status status) {
    if (env->ExceptionCheck() == JNI_TRUE) {
        // Something already went wrong on the Java side; that exception is the more useful one.
        return;
    }

    const char * message = la_last_error();
    jbyteArray message_bytes = to_byte_array(env, message, strlen(message));
    if (message_bytes == nullptr) {
        return;
    }

    jclass exception_class = env->FindClass(EXCEPTION_CLASS);
    if (exception_class == nullptr) {
        return;
    }
    jmethodID factory =
            env->GetStaticMethodID(exception_class, "of", EXCEPTION_FACTORY_SIGNATURE);
    if (factory == nullptr) {
        return;
    }
    jobject exception = env->CallStaticObjectMethod(exception_class, factory,
                                                    static_cast<jint>(status), message_bytes);
    if (exception == nullptr) {
        return;
    }
    env->Throw(static_cast<jthrowable>(exception));
}

la_generate_params generate_params(jfloat temperature, jint max_tokens, jint seed,
                                   jboolean add_special, jboolean reuse_prefix) {
    la_generate_params params = la_generate_params_default();
    params.temperature = temperature;
    params.max_tokens = max_tokens;
    params.seed = static_cast<uint32_t>(seed);
    params.add_special = add_special == JNI_TRUE;
    params.reuse_prefix = reuse_prefix == JNI_TRUE;
    return params;
}

/** Hands the generated text to Java and frees the C copy. */
jbyteArray take_text(JNIEnv * env, char * text) {
    jbyteArray result = to_byte_array(env, text, strlen(text));
    la_free_string(text);
    return result;
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_com_eaglys_llamaaudio_LlamaAudio_nativeOpen(
        JNIEnv * env, jclass, jbyteArray model_path, jbyteArray mmproj_path, jint context_size,
        jint gpu_layers, jint threads, jboolean mmproj_use_gpu) {
    std::string model;
    if (!to_string(env, model_path, model)) {
        return 0;
    }

    std::string mmproj;
    const bool has_mmproj = mmproj_path != nullptr;
    if (has_mmproj && !to_string(env, mmproj_path, mmproj)) {
        return 0;
    }

    la_model_params params = la_model_params_default();
    params.model_path     = model.c_str();
    params.mmproj_path    = has_mmproj ? mmproj.c_str() : nullptr;
    params.n_ctx          = static_cast<uint32_t>(context_size);
    params.n_gpu_layers   = gpu_layers;
    params.n_threads      = threads;
    params.mmproj_use_gpu = mmproj_use_gpu == JNI_TRUE;

    la_context * ctx = nullptr;
    const la_status status = la_open(&params, &ctx);
    if (status != LA_OK) {
        throw_status(env, status);
        return 0;
    }
    return static_cast<jlong>(reinterpret_cast<intptr_t>(ctx));
}

JNIEXPORT void JNICALL Java_com_eaglys_llamaaudio_LlamaAudio_nativeClose(
        JNIEnv *, jclass, jlong handle) {
    la_close(as_context(handle));
}

JNIEXPORT jboolean JNICALL Java_com_eaglys_llamaaudio_LlamaAudio_nativeSupportsAudio(
        JNIEnv *, jclass, jlong handle) {
    return la_supports_audio(as_context(handle)) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jint JNICALL Java_com_eaglys_llamaaudio_LlamaAudio_nativeAudioSampleRate(
        JNIEnv *, jclass, jlong handle) {
    return la_audio_sample_rate(as_context(handle));
}

JNIEXPORT jbyteArray JNICALL Java_com_eaglys_llamaaudio_LlamaAudio_nativeMediaMarker(
        JNIEnv * env, jclass) {
    const char * marker = la_media_marker();
    return to_byte_array(env, marker, strlen(marker));
}

JNIEXPORT jbyteArray JNICALL Java_com_eaglys_llamaaudio_LlamaAudio_nativeGenerate(
        JNIEnv * env, jclass, jlong handle, jbyteArray prompt_bytes, jfloat temperature,
        jint max_tokens, jint seed, jboolean add_special, jboolean reuse_prefix) {
    std::string prompt;
    if (!to_string(env, prompt_bytes, prompt)) {
        return nullptr;
    }

    const la_generate_params params =
            generate_params(temperature, max_tokens, seed, add_special, reuse_prefix);

    char * text = nullptr;
    const la_status status = la_generate(as_context(handle), &params, prompt.c_str(), &text);
    if (status != LA_OK) {
        throw_status(env, status);
        return nullptr;
    }
    return take_text(env, text);
}

JNIEXPORT jbyteArray JNICALL Java_com_eaglys_llamaaudio_LlamaAudio_nativeGenerateWithAudio(
        JNIEnv * env, jclass, jlong handle, jbyteArray prompt_bytes, jfloatArray sample_array,
        jfloat temperature, jint max_tokens, jint seed, jboolean add_special,
        jboolean reuse_prefix) {
    std::string prompt;
    if (!to_string(env, prompt_bytes, prompt)) {
        return nullptr;
    }

    // Copied rather than pinned: generating takes seconds, and a pinned array can hold up the
    // garbage collector for all of it.
    const jsize n_samples = env->GetArrayLength(sample_array);
    std::vector<float> samples(static_cast<size_t>(n_samples));
    if (n_samples > 0) {
        env->GetFloatArrayRegion(sample_array, 0, n_samples, samples.data());
    }
    if (env->ExceptionCheck() == JNI_TRUE) {
        return nullptr;
    }

    const la_generate_params params =
            generate_params(temperature, max_tokens, seed, add_special, reuse_prefix);

    char * text = nullptr;
    const la_status status =
            la_generate_with_audio(as_context(handle), &params, prompt.c_str(), samples.data(),
                                   samples.size(), &text);
    if (status != LA_OK) {
        throw_status(env, status);
        return nullptr;
    }
    return take_text(env, text);
}

} // extern "C"
