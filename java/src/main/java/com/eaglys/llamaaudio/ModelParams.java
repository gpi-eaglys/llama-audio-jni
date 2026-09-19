package com.eaglys.llamaaudio;

import java.nio.file.Path;
import java.util.Objects;

/**
 * What to open, and how.
 *
 * <p>The defaults match {@code la_model_params_default} in
 * {@code native/src/llama_audio.cpp} and have to be changed together with it.
 *
 * @param modelPath     the model file, in GGUF format
 * @param mmprojPath    the multimodal projector file, or null when audio is not needed
 * @param contextSize   context window size in tokens; 0 takes the size the model was trained with
 * @param gpuLayers     how many model layers to run on the GPU; 0 runs everything on the CPU
 * @param threads       how many threads to generate with; 0 or less picks a number from the hardware
 * @param mmprojUseGpu  runs the audio and image encoder on the GPU
 */
public record ModelParams(
        Path modelPath,
        Path mmprojPath,
        int contextSize,
        int gpuLayers,
        int threads,
        boolean mmprojUseGpu) {

    public ModelParams {
        Objects.requireNonNull(modelPath, "modelPath");
        if (contextSize < 0) {
            throw new IllegalArgumentException("contextSize cannot be negative: " + contextSize);
        }
        if (gpuLayers < 0) {
            throw new IllegalArgumentException("gpuLayers cannot be negative: " + gpuLayers);
        }
    }

    /** Parameters for a model, with every other field at its default. */
    public static ModelParams of(Path modelPath) {
        return new ModelParams(modelPath, null, 4096, 0, 0, false);
    }

    /** Parameters for a model, with every other field at its default. */
    public static ModelParams of(String modelPath) {
        return of(Path.of(modelPath));
    }

    /** The projector file that lets this model read audio. */
    public ModelParams withMmproj(Path mmprojPath) {
        return new ModelParams(modelPath, Objects.requireNonNull(mmprojPath, "mmprojPath"),
                contextSize, gpuLayers, threads, mmprojUseGpu);
    }

    /** The projector file that lets this model read audio. */
    public ModelParams withMmproj(String mmprojPath) {
        return withMmproj(Path.of(mmprojPath));
    }

    public ModelParams withContextSize(int contextSize) {
        return new ModelParams(modelPath, mmprojPath, contextSize, gpuLayers, threads, mmprojUseGpu);
    }

    public ModelParams withGpuLayers(int gpuLayers) {
        return new ModelParams(modelPath, mmprojPath, contextSize, gpuLayers, threads, mmprojUseGpu);
    }

    public ModelParams withThreads(int threads) {
        return new ModelParams(modelPath, mmprojPath, contextSize, gpuLayers, threads, mmprojUseGpu);
    }

    public ModelParams withMmprojUseGpu(boolean mmprojUseGpu) {
        return new ModelParams(modelPath, mmprojPath, contextSize, gpuLayers, threads, mmprojUseGpu);
    }
}
