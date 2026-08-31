/*
 * Checks that a model loads and generates, from text and from audio, without a JVM in the way.
 *
 * Usage:
 *   llama-audio-smoke <model.gguf> [--mmproj <mmproj.gguf>] [--audio <file.wav>]
 *                     [--prompt <text>]
 *
 * Without --mmproj it checks the text path only. With --mmproj and --audio it also reads the
 * audio file and checks the audio path.
 */

#include "llama_audio.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct options {
    std::string model_path;
    std::string mmproj_path;
    std::string audio_path;
    std::string prompt = "Say hello in five words.";
    /** Sends the prompt exactly as given, instead of putting it in a Gemma user turn. */
    bool raw_prompt = false;
};

/*
 * The library sends prompts as they are given, because which turn markers a model wants is the
 * model's business, not the library's. These tests run against Gemma, so they add Gemma's
 * markers here. Without them an instruction trained model answers a bare prompt with an
 * end-of-turn token and generates nothing at all.
 */
std::string gemma_user_turn(const std::string & text) {
    return "<start_of_turn>user\n" + text + "<end_of_turn>\n<start_of_turn>model\n";
}

std::string gemma_next_user_turn(const std::string & conversation_so_far,
                                 const std::string & model_reply, const std::string & text) {
    return conversation_so_far + model_reply + "<end_of_turn>\n<start_of_turn>user\n" + text +
           "<end_of_turn>\n<start_of_turn>model\n";
}

bool parse_options(int argc, char ** argv, options & out) {
    if (argc < 2) {
        return false;
    }
    out.model_path = argv[1];
    for (int i = 2; i < argc; i++) {
        const bool has_value = (i + 1) < argc;
        if (strcmp(argv[i], "--mmproj") == 0 && has_value) {
            out.mmproj_path = argv[++i];
        } else if (strcmp(argv[i], "--audio") == 0 && has_value) {
            out.audio_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && has_value) {
            out.prompt = argv[++i];
        } else if (strcmp(argv[i], "--raw") == 0) {
            out.raw_prompt = true;
        } else {
            fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

uint32_t read_u32(const unsigned char * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) |
           ((uint32_t) p[3] << 24);
}

uint16_t read_u16(const unsigned char * p) {
    return (uint16_t) ((uint16_t) p[0] | ((uint16_t) p[1] << 8));
}

/**
 * Reads 16 bit PCM samples out of a WAV file and scales them to between -1 and 1. Keeps the
 * first channel when the file has more than one. Only enough of the format is handled to read
 * test files.
 */
bool read_wav(const std::string & path, std::vector<float> & out, uint32_t & out_sample_rate) {
    FILE * file = fopen(path.c_str(), "rb");
    if (file == nullptr) {
        fprintf(stderr, "could not open '%s'\n", path.c_str());
        return false;
    }

    std::vector<unsigned char> bytes;
    unsigned char buffer[65536];
    size_t read = 0;
    while ((read = fread(buffer, 1, sizeof(buffer), file)) > 0) {
        bytes.insert(bytes.end(), buffer, buffer + read);
    }
    fclose(file);

    if (bytes.size() < 44 || memcmp(bytes.data(), "RIFF", 4) != 0 ||
        memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "'%s' is not a WAV file\n", path.c_str());
        return false;
    }

    uint16_t channels = 1;
    uint16_t bits_per_sample = 16;
    out_sample_rate = 0;

    // Walk the chunks, picking up the format and then the samples.
    size_t at = 12;
    while (at + 8 <= bytes.size()) {
        const char * chunk_id = (const char *) (bytes.data() + at);
        const uint32_t chunk_size = read_u32(bytes.data() + at + 4);
        const size_t body = at + 8;

        if (memcmp(chunk_id, "fmt ", 4) == 0 && body + 16 <= bytes.size()) {
            channels        = read_u16(bytes.data() + body + 2);
            out_sample_rate = read_u32(bytes.data() + body + 4);
            bits_per_sample = read_u16(bytes.data() + body + 14);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            if (bits_per_sample != 16) {
                fprintf(stderr, "'%s' holds %u bit samples; only 16 bit is handled\n",
                        path.c_str(), bits_per_sample);
                return false;
            }
            if (channels == 0) {
                fprintf(stderr, "'%s' reports no channels\n", path.c_str());
                return false;
            }
            const size_t available = std::min((size_t) chunk_size, bytes.size() - body);
            const size_t frames = available / (2u * channels);
            out.resize(frames);
            for (size_t frame = 0; frame < frames; frame++) {
                const size_t offset = body + frame * 2u * channels;
                const int16_t sample =
                        (int16_t) ((uint16_t) bytes[offset] | ((uint16_t) bytes[offset + 1] << 8));
                out[frame] = (float) sample / 32768.0f;
            }
            return true;
        }

        // Chunks are padded to an even number of bytes.
        at = body + chunk_size + (chunk_size & 1u);
    }

    fprintf(stderr, "'%s' has no data chunk\n", path.c_str());
    return false;
}

bool check(la_status status, const char * what) {
    if (status == LA_OK) {
        return true;
    }
    fprintf(stderr, "FAIL %s: %s (%s)\n", what, la_status_name(status), la_last_error());
    return false;
}

/**
 * Checks a generate call: it has to report success and it has to produce text. A call that
 * succeeds and returns nothing is a failure, because a model that immediately stops is the way
 * a wrongly built prompt shows up.
 */
bool check_generated(la_status status, const char * what, const char * text,
                     std::string & out_text) {
    if (!check(status, what)) {
        return false;
    }
    if (text == nullptr || text[0] == '\0') {
        fprintf(stderr, "FAIL %s: the call succeeded but generated no text\n", what);
        return false;
    }
    out_text = text;
    printf("PASS  %s, %zu characters: %s\n", what, out_text.size(), out_text.c_str());
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    options opts;
    if (!parse_options(argc, argv, opts)) {
        fprintf(stderr,
                "usage: %s <model.gguf> [--mmproj <mmproj.gguf>] [--audio <file.wav>] "
                "[--prompt <text>] [--raw]\n",
                argv[0]);
        return 2;
    }

    la_model_params model_params = la_model_params_default();
    model_params.model_path = opts.model_path.c_str();
    if (!opts.mmproj_path.empty()) {
        model_params.mmproj_path = opts.mmproj_path.c_str();
    }

    la_context * ctx = nullptr;
    if (!check(la_open(&model_params, &ctx), "open the model")) {
        return 1;
    }
    printf("PASS  opened '%s'\n", opts.model_path.c_str());
    printf("      audio supported: %s\n", la_supports_audio(ctx) ? "yes" : "no");
    if (la_supports_audio(ctx)) {
        printf("      audio sample rate: %d Hz\n", la_audio_sample_rate(ctx));
    }

    int failures = 0;

    const std::string first_prompt =
            opts.raw_prompt ? opts.prompt : gemma_user_turn(opts.prompt);
    std::string first_reply;

    // Text.
    {
        la_generate_params params = la_generate_params_default();
        params.max_tokens = 24;
        char * text = nullptr;
        const la_status status = la_generate(ctx, &params, first_prompt.c_str(), &text);
        if (check_generated(status, "generated from text", text, first_reply)) {
            la_free_string(text);
        } else {
            failures++;
            la_free_string(text);
        }
    }

    // A second turn on the same context. The prompt starts with everything sent the first time,
    // so the work already in the cache is reused instead of being repeated. This is the shape of
    // a conversation being read as it happens.
    if (!opts.raw_prompt && !first_reply.empty()) {
        la_generate_params params = la_generate_params_default();
        params.max_tokens = 16;
        const std::string second_prompt =
                gemma_next_user_turn(first_prompt, first_reply, "Now count to three.");
        char * text = nullptr;
        std::string reply;
        const la_status status = la_generate(ctx, &params, second_prompt.c_str(), &text);
        if (check_generated(status, "generated from a second turn, reusing the prefix", text, reply)) {
            la_free_string(text);
        } else {
            failures++;
            la_free_string(text);
        }
    }

    // Audio.
    if (!opts.audio_path.empty()) {
        if (!la_supports_audio(ctx)) {
            fprintf(stderr, "FAIL audio requested but this model cannot read audio\n");
            failures++;
        } else {
            std::vector<float> samples;
            uint32_t sample_rate = 0;
            if (!read_wav(opts.audio_path, samples, sample_rate)) {
                failures++;
            } else {
                printf("      read %zu samples at %u Hz from '%s'\n", samples.size(), sample_rate,
                       opts.audio_path.c_str());
                const int32_t expected = la_audio_sample_rate(ctx);
                if (expected > 0 && sample_rate != (uint32_t) expected) {
                    printf("      note: the file is %u Hz but the model expects %d Hz\n",
                           sample_rate, expected);
                }

                // The media marker goes inside the user turn, so the audio is read as part of
                // what the user said rather than in front of the whole conversation.
                const std::string audio_prompt =
                        opts.raw_prompt
                                ? opts.prompt
                                : gemma_user_turn(std::string(la_media_marker()) + "\n" +
                                                  opts.prompt);

                la_generate_params params = la_generate_params_default();
                params.max_tokens = 64;
                char * text = nullptr;
                std::string reply;
                const la_status status =
                        la_generate_with_audio(ctx, &params, audio_prompt.c_str(), samples.data(),
                                               samples.size(), &text);
                if (check_generated(status, "generated from audio", text, reply)) {
                    la_free_string(text);
                } else {
                    failures++;
                    la_free_string(text);
                }
            }
        }
    }

    la_close(ctx);

    if (failures > 0) {
        printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    printf("\nall checks passed\n");
    return 0;
}
