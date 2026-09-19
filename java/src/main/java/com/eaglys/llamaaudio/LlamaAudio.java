package com.eaglys.llamaaudio;

import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * A loaded model, and the state that goes with it.
 *
 * <p>Generates text from a prompt, on its own or together with audio, with the model running in
 * this process. Open one with {@link #open(ModelParams)} and close it when done; it holds a model
 * in native memory, which the garbage collector knows nothing about.
 *
 * <pre>{@code
 * ModelParams params = ModelParams.of("gemma4.gguf").withMmproj("mmproj.gguf");
 * try (LlamaAudio model = LlamaAudio.open(params)) {
 *     String text = model.generateWithAudio(prompt, samples);
 * }
 * }</pre>
 *
 * <p>One instance is for one thread at a time. Calling into the same instance from two threads at
 * once is not checked for and corrupts the model's state.
 *
 * <p>Prompts are sent as they are given. Whichever turn or role markers a model expects are the
 * caller's to add, because which ones they are belongs to the model rather than to this wrapper.
 * An instruction trained model given a bare prompt replies with an end-of-turn token and generates
 * nothing at all.
 */
public final class LlamaAudio implements AutoCloseable {

    static {
        NativeLibrary.load();
    }

    /** The {@code la_context *} this instance owns, or 0 once it is closed. */
    private long handle;

    private LlamaAudio(long handle) {
        this.handle = handle;
    }

    /**
     * Loads a model.
     *
     * @throws LlamaAudioException when the model or the projector cannot be loaded
     */
    public static LlamaAudio open(ModelParams params) {
        Objects.requireNonNull(params, "params");
        byte[] mmproj = params.mmprojPath() == null ? null : utf8(params.mmprojPath().toString());
        long handle = nativeOpen(utf8(params.modelPath().toString()), mmproj, params.contextSize(),
                params.gpuLayers(), params.threads(), params.mmprojUseGpu());
        return new LlamaAudio(handle);
    }

    /** Reports whether this model can read audio, which it can when it was opened with a projector. */
    public boolean supportsAudio() {
        return nativeSupportsAudio(handle());
    }

    /**
     * The sample rate this model expects, in Hz. 16000 for Gemma 4.
     *
     * @return -1 when the model cannot read audio
     */
    public int audioSampleRate() {
        return nativeAudioSampleRate(handle());
    }

    /**
     * The text that marks where audio belongs in a prompt. Put it in the prompt passed to
     * {@link #generateWithAudio} to choose where the audio goes; without it the audio is placed in
     * front of the prompt.
     */
    public static String mediaMarker() {
        return decode(nativeMediaMarker());
    }

    /** Generates text for a prompt, with the default parameters. */
    public String generate(String prompt) {
        return generate(prompt, GenerateParams.defaults());
    }

    /**
     * Generates text for a prompt.
     *
     * <p>The part of the prompt that repeats the previous prompt on this instance is not read
     * again, so a conversation that grows as it is transcribed costs only the new text. Turn that
     * off with {@link GenerateParams#withReusePrefix}.
     *
     * @throws LlamaAudioException when the prompt cannot be read or generation fails
     */
    public String generate(String prompt, GenerateParams params) {
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(params, "params");
        return decode(nativeGenerate(handle(), utf8(prompt), params.temperature(),
                params.maxTokens(), params.seed(), params.addSpecial(), params.reusePrefix()));
    }

    /** Generates text for a prompt together with a piece of audio, with the default parameters. */
    public String generateWithAudio(String prompt, float[] samples) {
        return generateWithAudio(prompt, samples, GenerateParams.defaults());
    }

    /**
     * Generates text for a prompt together with a piece of audio.
     *
     * <p>The audio is one channel of samples between -1 and 1, at the rate reported by
     * {@link #audioSampleRate}. Models put an upper limit on how long a single piece of audio can
     * be; for Gemma 4 it is 30 seconds, and longer audio is split up and read in parts.
     *
     * <p>Each call starts with an empty key-value cache, because audio becomes embeddings rather
     * than tokens and there is no token prefix to compare against. {@code reusePrefix} therefore
     * has no effect here.
     *
     * @throws LlamaAudioException when this model cannot read audio, or generation fails
     */
    public String generateWithAudio(String prompt, float[] samples, GenerateParams params) {
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(samples, "samples");
        Objects.requireNonNull(params, "params");
        if (samples.length == 0) {
            throw new IllegalArgumentException("samples is empty");
        }
        return decode(nativeGenerateWithAudio(handle(), utf8(prompt), samples, params.temperature(),
                params.maxTokens(), params.seed(), params.addSpecial(), params.reusePrefix()));
    }

    /** Frees the model. Calling this more than once does nothing the second time. */
    @Override
    public void close() {
        if (handle != 0) {
            long closing = handle;
            handle = 0;
            nativeClose(closing);
        }
    }

    private long handle() {
        if (handle == 0) {
            throw new IllegalStateException("this LlamaAudio is closed");
        }
        return handle;
    }

    /*
     * Text crosses the boundary as UTF-8 bytes rather than as a Java string. JNI's own string
     * calls speak modified UTF-8, which is not the UTF-8 llama.cpp reads and writes: they differ
     * on any character outside the basic multilingual plane, so a prompt or a reply holding one
     * would be mangled.
     */

    private static byte[] utf8(String text) {
        return text.getBytes(StandardCharsets.UTF_8);
    }

    private static String decode(byte[] utf8) {
        return new String(utf8, StandardCharsets.UTF_8);
    }

    private static native long nativeOpen(byte[] modelPath, byte[] mmprojPath, int contextSize,
            int gpuLayers, int threads, boolean mmprojUseGpu);

    private static native void nativeClose(long handle);

    private static native boolean nativeSupportsAudio(long handle);

    private static native int nativeAudioSampleRate(long handle);

    private static native byte[] nativeMediaMarker();

    private static native byte[] nativeGenerate(long handle, byte[] prompt, float temperature,
            int maxTokens, int seed, boolean addSpecial, boolean reusePrefix);

    private static native byte[] nativeGenerateWithAudio(long handle, byte[] prompt, float[] samples,
            float temperature, int maxTokens, int seed, boolean addSpecial, boolean reusePrefix);
}
