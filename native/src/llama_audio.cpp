#include "llama_audio.h"

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

/** The message for the last failed call, per thread. */
thread_local std::string g_last_error;

void set_error(const char * fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    g_last_error = buf;
}

void clear_error() {
    g_last_error.clear();
}

/**
 * Starts the llama.cpp backend once per process. There is no matching shutdown: a context can
 * be opened at any time, so there is no point at which the backend is known to be unused.
 */
void backend_init_once() {
    static std::once_flag once;
    std::call_once(once, []() { llama_backend_init(); });
}

int32_t pick_thread_count(int32_t requested) {
    if (requested > 0) {
        return requested;
    }
    const unsigned int hardware = std::thread::hardware_concurrency();
    return hardware > 0 ? (int32_t) hardware : 4;
}

/** Turns text into tokens. */
la_status tokenize(const llama_vocab * vocab, const std::string & text, bool add_special,
                   std::vector<llama_token> & out) {
    // A negative return is the token count needed, so the first call sizes the buffer.
    const int32_t needed = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0,
                                            add_special, true);
    if (needed < 0) {
        set_error("could not work out how many tokens the prompt needs");
        return LA_ERR_TOKENIZE_FAILED;
    }
    out.resize((size_t) needed);
    const int32_t written = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(),
                                            needed, add_special, true);
    if (written < 0) {
        set_error("could not turn the prompt into tokens");
        return LA_ERR_TOKENIZE_FAILED;
    }
    out.resize((size_t) written);
    return LA_OK;
}

/** Appends the text for one token, leaving out tokens that carry no text of their own. */
void append_token_text(std::string & out, const llama_vocab * vocab, llama_token token) {
    char buf[256];
    int32_t n = llama_token_to_piece(vocab, token, buf, (int32_t) sizeof(buf), 0, false);
    if (n >= 0) {
        out.append(buf, (size_t) n);
        return;
    }
    std::vector<char> larger((size_t) -n);
    n = llama_token_to_piece(vocab, token, larger.data(), (int32_t) larger.size(), 0, false);
    if (n > 0) {
        out.append(larger.data(), (size_t) n);
    }
}

char * copy_to_c_string(const std::string & text) {
    char * out = (char *) malloc(text.size() + 1);
    if (out == nullptr) {
        return nullptr;
    }
    memcpy(out, text.c_str(), text.size() + 1);
    return out;
}

} // namespace

struct la_context {
    llama_model * model = nullptr;
    llama_context * lctx = nullptr;
    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int32_t n_threads = 4;

    /** The tokens currently held in the key-value cache, used to reuse a repeated prefix. */
    std::vector<llama_token> cached;

    /** Feeds tokens to the model, starting at position n_past. */
    la_status decode_tokens(const std::vector<llama_token> & tokens, size_t from,
                            llama_pos n_past, llama_pos * out_n_past) {
        const int32_t n_batch = (int32_t) llama_n_batch(lctx);
        llama_batch batch = llama_batch_init(n_batch, 0, 1);

        la_status status = LA_OK;
        for (size_t i = from; i < tokens.size(); i += (size_t) n_batch) {
            const size_t count = std::min((size_t) n_batch, tokens.size() - i);
            const bool is_last_batch = (i + count) >= tokens.size();

            batch.n_tokens = (int32_t) count;
            for (size_t j = 0; j < count; j++) {
                batch.token[j] = tokens[i + j];
                batch.pos[j] = n_past + (llama_pos) j;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                // Only the very last token needs to produce logits to generate from.
                batch.logits[j] = is_last_batch && (j + 1 == count);
            }

            if (llama_decode(lctx, batch) != 0) {
                set_error("the model could not read the prompt");
                status = LA_ERR_DECODE_FAILED;
                break;
            }
            n_past += (llama_pos) count;
        }

        llama_batch_free(batch);
        *out_n_past = n_past;
        return status;
    }

    /** Generates tokens until the model stops or the limit is reached. */
    la_status generate_tokens(const la_generate_params & params, llama_pos n_past,
                              std::string & out_text) {
        llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (params.temperature <= 0.0f) {
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed));
        }

        llama_batch batch = llama_batch_init(1, 0, 1);
        la_status status = LA_OK;
        const llama_pos n_ctx = (llama_pos) llama_n_ctx(lctx);

        for (int32_t produced = 0; produced < params.max_tokens; produced++) {
            const llama_token token = llama_sampler_sample(sampler, lctx, -1);
            llama_sampler_accept(sampler, token);

            if (llama_vocab_is_eog(vocab, token)) {
                break;
            }
            append_token_text(out_text, vocab, token);
            cached.push_back(token);

            if (n_past >= n_ctx) {
                set_error("ran out of context window after %d generated tokens", produced + 1);
                status = LA_ERR_CONTEXT_FULL;
                break;
            }

            batch.n_tokens = 1;
            batch.token[0] = token;
            batch.pos[0] = n_past++;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;
            if (llama_decode(lctx, batch) != 0) {
                set_error("the model could not read back a generated token");
                status = LA_ERR_DECODE_FAILED;
                break;
            }
        }

        llama_batch_free(batch);
        llama_sampler_free(sampler);
        return status;
    }
};

const char * la_status_name(la_status status) {
    switch (status) {
        case LA_OK:                   return "ok";
        case LA_ERR_INVALID_ARGUMENT: return "invalid argument";
        case LA_ERR_LOAD_FAILED:      return "load failed";
        case LA_ERR_NO_AUDIO:         return "no audio support";
        case LA_ERR_TOKENIZE_FAILED:  return "tokenize failed";
        case LA_ERR_DECODE_FAILED:    return "decode failed";
        case LA_ERR_CONTEXT_FULL:     return "context full";
        case LA_ERR_OUT_OF_MEMORY:    return "out of memory";
    }
    return "unknown";
}

const char * la_last_error(void) {
    return g_last_error.c_str();
}

la_model_params la_model_params_default(void) {
    la_model_params params = {};
    params.model_path = nullptr;
    params.mmproj_path = nullptr;
    params.n_ctx = 4096;
    params.n_gpu_layers = 0;
    params.n_threads = 0;
    params.mmproj_use_gpu = false;
    return params;
}

la_generate_params la_generate_params_default(void) {
    la_generate_params params = {};
    params.temperature = 0.2f;
    params.max_tokens = 256;
    params.seed = 0;
    params.add_special = true;
    params.reuse_prefix = true;
    return params;
}

la_status la_open(const la_model_params * params, la_context ** out_ctx) {
    clear_error();

    if (params == nullptr || out_ctx == nullptr || params->model_path == nullptr) {
        set_error("model_path and out_ctx are required");
        return LA_ERR_INVALID_ARGUMENT;
    }

    backend_init_once();

    la_context * ctx = new la_context();
    ctx->n_threads = pick_thread_count(params->n_threads);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params->n_gpu_layers;

    ctx->model = llama_model_load_from_file(params->model_path, model_params);
    if (ctx->model == nullptr) {
        set_error("could not load the model from '%s'", params->model_path);
        la_close(ctx);
        return LA_ERR_LOAD_FAILED;
    }
    ctx->vocab = llama_model_get_vocab(ctx->model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params->n_ctx;
    ctx_params.n_batch = 2048;
    ctx_params.n_seq_max = 1;
    ctx_params.n_threads = ctx->n_threads;
    ctx_params.n_threads_batch = ctx->n_threads;

    ctx->lctx = llama_init_from_model(ctx->model, ctx_params);
    if (ctx->lctx == nullptr) {
        set_error("could not create a context for the model");
        la_close(ctx);
        return LA_ERR_LOAD_FAILED;
    }

    if (params->mmproj_path != nullptr && params->mmproj_path[0] != '\0') {
        mtmd_context_params mtmd_params = mtmd_context_params_default();
        mtmd_params.use_gpu = params->mmproj_use_gpu;
        mtmd_params.print_timings = false;
        mtmd_params.n_threads = ctx->n_threads;

        ctx->mctx = mtmd_init_from_file(params->mmproj_path, ctx->model, mtmd_params);
        if (ctx->mctx == nullptr) {
            set_error("could not load the projector from '%s'", params->mmproj_path);
            la_close(ctx);
            return LA_ERR_LOAD_FAILED;
        }
    }

    *out_ctx = ctx;
    return LA_OK;
}

void la_close(la_context * ctx) {
    if (ctx == nullptr) {
        return;
    }
    if (ctx->mctx != nullptr) {
        mtmd_free(ctx->mctx);
    }
    if (ctx->lctx != nullptr) {
        llama_free(ctx->lctx);
    }
    if (ctx->model != nullptr) {
        llama_model_free(ctx->model);
    }
    delete ctx;
}

bool la_supports_audio(const la_context * ctx) {
    return ctx != nullptr && ctx->mctx != nullptr && mtmd_support_audio(ctx->mctx);
}

int32_t la_audio_sample_rate(const la_context * ctx) {
    if (!la_supports_audio(ctx)) {
        return -1;
    }
    return mtmd_get_audio_sample_rate(ctx->mctx);
}

const char * la_media_marker(void) {
    return mtmd_default_marker();
}

la_status la_generate(la_context * ctx, const la_generate_params * params, const char * prompt,
                      char ** out_text) {
    clear_error();

    if (ctx == nullptr || params == nullptr || prompt == nullptr || out_text == nullptr) {
        set_error("ctx, params, prompt and out_text are all required");
        return LA_ERR_INVALID_ARGUMENT;
    }

    std::vector<llama_token> tokens;
    const la_status tokenized = tokenize(ctx->vocab, prompt, params->add_special, tokens);
    if (tokenized != LA_OK) {
        return tokenized;
    }
    if (tokens.empty()) {
        set_error("the prompt produced no tokens");
        return LA_ERR_INVALID_ARGUMENT;
    }
    if (tokens.size() >= (size_t) llama_n_ctx(ctx->lctx)) {
        set_error("the prompt is %zu tokens, which does not fit in a context window of %u",
                  tokens.size(), llama_n_ctx(ctx->lctx));
        return LA_ERR_CONTEXT_FULL;
    }

    // Work out how much of this prompt the previous one already put in the cache.
    size_t reuse = 0;
    if (params->reuse_prefix) {
        while (reuse < ctx->cached.size() && reuse < tokens.size() &&
               ctx->cached[reuse] == tokens[reuse]) {
            reuse++;
        }
        // At least one token has to be read now, otherwise there are no logits to generate from.
        if (reuse == tokens.size() && reuse > 0) {
            reuse--;
        }
    }

    // Drop whatever the cache holds past the part being reused.
    llama_memory_seq_rm(llama_get_memory(ctx->lctx), 0, (llama_pos) reuse, -1);
    ctx->cached.assign(tokens.begin(), tokens.end());

    llama_pos n_past = (llama_pos) reuse;
    const la_status decoded = ctx->decode_tokens(tokens, reuse, n_past, &n_past);
    if (decoded != LA_OK) {
        return decoded;
    }

    std::string text;
    const la_status generated = ctx->generate_tokens(*params, n_past, text);
    if (generated != LA_OK && text.empty()) {
        return generated;
    }

    *out_text = copy_to_c_string(text);
    if (*out_text == nullptr) {
        set_error("could not allocate the result text");
        return LA_ERR_OUT_OF_MEMORY;
    }
    return LA_OK;
}

la_status la_generate_with_audio(la_context * ctx, const la_generate_params * params,
                                 const char * prompt, const float * samples, size_t n_samples,
                                 char ** out_text) {
    clear_error();

    if (ctx == nullptr || params == nullptr || prompt == nullptr || out_text == nullptr) {
        set_error("ctx, params, prompt and out_text are all required");
        return LA_ERR_INVALID_ARGUMENT;
    }
    if (samples == nullptr || n_samples == 0) {
        set_error("samples is empty");
        return LA_ERR_INVALID_ARGUMENT;
    }
    if (!la_supports_audio(ctx)) {
        set_error("this context cannot read audio; open it with a projector file");
        return LA_ERR_NO_AUDIO;
    }

    // Audio turns into embeddings rather than tokens, so there is no token prefix to match
    // against a previous call. Every call starts from an empty cache.
    llama_memory_clear(llama_get_memory(ctx->lctx), true);
    ctx->cached.clear();

    mtmd_bitmap * bitmap = mtmd_bitmap_init_from_audio(n_samples, samples);
    if (bitmap == nullptr) {
        set_error("could not wrap the samples as audio");
        return LA_ERR_INVALID_ARGUMENT;
    }

    std::string full_prompt = prompt;
    const char * marker = mtmd_default_marker();
    if (full_prompt.find(marker) == std::string::npos) {
        full_prompt = std::string(marker) + "\n" + full_prompt;
    }

    mtmd_input_text text = {};
    text.text = full_prompt.c_str();
    text.text_len = full_prompt.size();
    text.add_special = params->add_special;
    text.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps[1] = {bitmap};

    const int32_t tokenized = mtmd_tokenize(ctx->mctx, chunks, &text, bitmaps, 1);
    mtmd_bitmap_free(bitmap);
    if (tokenized != 0) {
        mtmd_input_chunks_free(chunks);
        set_error("could not read the prompt together with the audio (code %d)", tokenized);
        return LA_ERR_TOKENIZE_FAILED;
    }

    llama_pos n_past = 0;
    const int32_t evaluated =
            mtmd_helper_eval_chunks(ctx->mctx, ctx->lctx, chunks, /*n_past=*/0, /*seq_id=*/0,
                                    (int32_t) llama_n_batch(ctx->lctx), /*logits_last=*/true,
                                    &n_past);
    mtmd_input_chunks_free(chunks);
    if (evaluated != 0) {
        set_error("the model could not read the audio (code %d)", evaluated);
        return LA_ERR_DECODE_FAILED;
    }

    std::string result;
    const la_status generated = ctx->generate_tokens(*params, n_past, result);
    if (generated != LA_OK && result.empty()) {
        return generated;
    }

    *out_text = copy_to_c_string(result);
    if (*out_text == nullptr) {
        set_error("could not allocate the result text");
        return LA_ERR_OUT_OF_MEMORY;
    }
    return LA_OK;
}

void la_free_string(char * text) {
    free(text);
}
