#ifndef LLAMA_AUDIO_H
#define LLAMA_AUDIO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Generates text from a prompt, on its own or together with audio, using a local model.
 *
 * This is a plain C interface so that bindings in other languages stay small. It is built on
 * two parts of llama.cpp:
 *
 *   - llama.h, the published C interface: models, tokens, sampling, key-value cache.
 *   - mtmd.h, the multimodal library: turns audio into something the model can read.
 *
 * It does not use llama.cpp's `common` directory. That code is written for llama.cpp's own
 * command line programs and changes without notice.
 *
 * One context holds one model in native memory. Call into a single context from one thread at
 * a time.
 */

/** Holds a loaded model and the state that goes with it. */
typedef struct la_context la_context;

/** Tells how a call ended. */
typedef enum {
    LA_OK                   = 0,
    LA_ERR_INVALID_ARGUMENT = 1,
    LA_ERR_LOAD_FAILED      = 2,
    /** The model was opened without a projector file, so it cannot read audio. */
    LA_ERR_NO_AUDIO         = 3,
    LA_ERR_TOKENIZE_FAILED  = 4,
    LA_ERR_DECODE_FAILED    = 5,
    /** The prompt does not fit in the context window. */
    LA_ERR_CONTEXT_FULL     = 6,
    LA_ERR_OUT_OF_MEMORY    = 7,
} la_status;

/** Returns a short name for a status, for logs and error messages. */
const char * la_status_name(la_status status);

/**
 * Returns the message for the last failed call on this thread. Returns an empty string when
 * the last call succeeded. The text stays valid until the next call on the same thread.
 */
const char * la_last_error(void);

/** What to open, and how. */
typedef struct {
    /** Path to the model file, in GGUF format. Required. */
    const char * model_path;
    /** Path to the multimodal projector file. Leave as NULL when audio is not needed. */
    const char * mmproj_path;
    /** Context window size, in tokens. 0 takes the size the model was trained with. */
    uint32_t n_ctx;
    /** How many model layers to run on the GPU. 0 runs everything on the CPU. */
    int32_t n_gpu_layers;
    /** How many threads to generate with. 0 or less picks a number from the hardware. */
    int32_t n_threads;
    /** Runs the audio and image encoder on the GPU. */
    bool mmproj_use_gpu;
} la_model_params;

/** Returns parameters with every field set to its default. */
la_model_params la_model_params_default(void);

/** How to generate. */
typedef struct {
    /** Higher values give more varied output. 0 or less always picks the most likely token. */
    float temperature;
    /** Upper limit on how many tokens to generate. */
    int32_t max_tokens;
    /** Seed for the random choice between tokens. Ignored when temperature is 0 or less. */
    uint32_t seed;
    /** Adds the model's begin-of-text token in front of the prompt. */
    bool add_special;
    /**
     * Skips work for the part of the prompt that repeats the previous prompt on this context.
     * This is what makes a growing prompt, such as a conversation being transcribed as it
     * happens, cheap to send again. Text prompts only.
     */
    bool reuse_prefix;
} la_generate_params;

/** Returns parameters with every field set to its default. */
la_generate_params la_generate_params_default(void);

/**
 * Opens a model. On success writes a context to out_ctx, which must be closed with la_close.
 * On failure out_ctx is left alone and la_last_error explains why.
 */
la_status la_open(const la_model_params * params, la_context ** out_ctx);

/** Closes a context and frees the model. Does nothing when ctx is NULL. */
void la_close(la_context * ctx);

/** Reports whether this context can read audio. */
bool la_supports_audio(const la_context * ctx);

/**
 * Returns the sample rate the model expects, in Hz. Returns -1 when the context cannot read
 * audio.
 */
int32_t la_audio_sample_rate(const la_context * ctx);

/**
 * Returns the text that marks where audio belongs in a prompt. Put it in the prompt passed to
 * la_generate_with_audio to choose where the audio goes. When the prompt does not contain it,
 * the audio is placed in front of the prompt.
 */
const char * la_media_marker(void);

/**
 * Generates text for a prompt. On success writes the generated text to out_text, which must be
 * freed with la_free_string.
 *
 * The prompt is used as it stands. Adding whatever turn or role markers a model expects is the
 * caller's job.
 */
la_status la_generate(la_context * ctx, const la_generate_params * params, const char * prompt,
                      char ** out_text);

/**
 * Generates text for a prompt together with a piece of audio. On success writes the generated
 * text to out_text, which must be freed with la_free_string.
 *
 * The audio is one channel of floating point samples between -1 and 1, at the rate reported by
 * la_audio_sample_rate. Models put an upper limit on how long a single piece of audio can be;
 * for Gemma 4 it is 30 seconds, and longer audio is split up and read in parts.
 *
 * Each call starts with an empty key-value cache, so reuse_prefix has no effect here.
 */
la_status la_generate_with_audio(la_context * ctx, const la_generate_params * params,
                                 const char * prompt, const float * samples, size_t n_samples,
                                 char ** out_text);

/** Frees text returned by a generate call. Does nothing when text is NULL. */
void la_free_string(char * text);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_AUDIO_H
