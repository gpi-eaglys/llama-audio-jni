package com.eaglys.llamaaudio;

/**
 * How to generate.
 *
 * <p>The defaults match {@code la_generate_params_default} in
 * {@code native/src/llama_audio.cpp} and have to be changed together with it.
 *
 * @param temperature  higher values give more varied output; 0 or less always picks the most
 *                     likely token
 * @param maxTokens    upper limit on how many tokens to generate
 * @param seed         seed for the random choice between tokens, ignored when temperature is 0
 *                     or less
 * @param addSpecial   adds the model's begin-of-text token in front of the prompt
 * @param reusePrefix  skips work for the part of the prompt that repeats the previous prompt on
 *                     this context; text prompts only
 */
public record GenerateParams(
        float temperature,
        int maxTokens,
        int seed,
        boolean addSpecial,
        boolean reusePrefix) {

    public GenerateParams {
        if (maxTokens <= 0) {
            throw new IllegalArgumentException("maxTokens has to be positive: " + maxTokens);
        }
    }

    /** Parameters with every field at its default. */
    public static GenerateParams defaults() {
        return new GenerateParams(0.2f, 256, 0, true, true);
    }

    public GenerateParams withTemperature(float temperature) {
        return new GenerateParams(temperature, maxTokens, seed, addSpecial, reusePrefix);
    }

    public GenerateParams withMaxTokens(int maxTokens) {
        return new GenerateParams(temperature, maxTokens, seed, addSpecial, reusePrefix);
    }

    public GenerateParams withSeed(int seed) {
        return new GenerateParams(temperature, maxTokens, seed, addSpecial, reusePrefix);
    }

    public GenerateParams withAddSpecial(boolean addSpecial) {
        return new GenerateParams(temperature, maxTokens, seed, addSpecial, reusePrefix);
    }

    public GenerateParams withReusePrefix(boolean reusePrefix) {
        return new GenerateParams(temperature, maxTokens, seed, addSpecial, reusePrefix);
    }
}
