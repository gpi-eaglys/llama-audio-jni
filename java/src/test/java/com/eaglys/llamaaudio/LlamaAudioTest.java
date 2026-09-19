package com.eaglys.llamaaudio;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Checks the binding against a real model, the same way {@code native/test/smoke.cpp} checks the
 * C interface.
 *
 * <p>Most of this needs model files, which are too large to keep in the repository, so those
 * tests are skipped unless the paths are given:
 *
 * <pre>
 * mvn test -Dllama.audio.test.model=model.gguf \
 *          -Dllama.audio.test.mmproj=mmproj.gguf \
 *          -Dllama.audio.test.audio=sample.wav
 * </pre>
 *
 * <p>The checks that only need the shared library run either way.
 */
class LlamaAudioTest {

    private static final Path MODEL = pathProperty("llama.audio.test.model");
    private static final Path MMPROJ = pathProperty("llama.audio.test.mmproj");
    private static final Path AUDIO = pathProperty("llama.audio.test.audio");

    @BeforeAll
    static void libraryLoads() {
        try {
            LlamaAudio.mediaMarker();
        } catch (UnsatisfiedLinkError | NoClassDefFoundError missing) {
            assumeTrue(false, "the JNI library is not built; run CMake first (" + missing + ")");
        }
    }

    @Test
    void mediaMarkerIsGiven() {
        assertFalse(LlamaAudio.mediaMarker().isEmpty());
    }

    @Test
    void openingAMissingModelFails() {
        Path missing = Path.of("no-such-model-2b41f0.gguf");

        LlamaAudioException failure =
                assertThrows(LlamaAudioException.class, () -> LlamaAudio.open(ModelParams.of(missing)));

        assertEquals(Status.LOAD_FAILED, failure.status());
        // The message comes from the C side, so this also checks that it survives the crossing.
        assertTrue(failure.getMessage().contains(missing.toString()), failure.getMessage());
    }

    @Test
    void generatesFromText() {
        try (LlamaAudio model = openModel()) {
            String reply = model.generate(gemmaUserTurn("Say hello in five words."));
            assertFalse(reply.isBlank(), "the call succeeded but generated no text");
        }
    }

    @Test
    void generatesFromASecondTurnReusingThePrefix() {
        try (LlamaAudio model = openModel()) {
            String firstPrompt = gemmaUserTurn("Say hello in five words.");
            String firstReply = model.generate(firstPrompt);

            String secondPrompt = firstPrompt + firstReply
                    + "<end_of_turn>\n<start_of_turn>user\nNow say goodbye.<end_of_turn>\n"
                    + "<start_of_turn>model\n";
            String secondReply = model.generate(secondPrompt);

            assertFalse(secondReply.isBlank(), "the call succeeded but generated no text");
            assertNotEquals(firstReply, secondReply);
        }
    }

    @Test
    void generatesFromAudio() throws IOException, UnsupportedAudioFileException {
        assumeTrue(MMPROJ != null, "no llama.audio.test.mmproj given");
        assumeTrue(AUDIO != null, "no llama.audio.test.audio given");

        try (LlamaAudio model = openModel()) {
            assertTrue(model.supportsAudio(), "the model was opened with a projector");

            float[] samples = readWav(AUDIO, model.audioSampleRate());
            String prompt = gemmaUserTurn(LlamaAudio.mediaMarker() + "\nTranscribe the audio.");

            String transcript = model.generateWithAudio(prompt, samples);
            assertFalse(transcript.isBlank(), "the call succeeded but generated no text");
        }
    }

    @Test
    void audioWithoutAProjectorIsRefused() {
        assumeTrue(MODEL != null, "no llama.audio.test.model given");

        // Opened without a projector on purpose, whatever else the run was given.
        try (LlamaAudio model = LlamaAudio.open(ModelParams.of(MODEL))) {
            assertFalse(model.supportsAudio());
            assertEquals(-1, model.audioSampleRate());

            LlamaAudioException failure = assertThrows(LlamaAudioException.class,
                    () -> model.generateWithAudio("anything", new float[] {0.0f}));
            assertEquals(Status.NO_AUDIO, failure.status());
        }
    }

    @Test
    void emptyAudioIsRefusedBeforeReachingTheModel() {
        try (LlamaAudio model = openModel()) {
            assertThrows(IllegalArgumentException.class,
                    () -> model.generateWithAudio("anything", new float[0]));
        }
    }

    @Test
    void usingAClosedModelThrows() {
        LlamaAudio model = openModel();
        model.close();
        model.close(); // closing twice is allowed

        assertThrows(IllegalStateException.class, () -> model.generate("anything"));
    }

    private static LlamaAudio openModel() {
        assumeTrue(MODEL != null, "no llama.audio.test.model given");
        ModelParams params = ModelParams.of(MODEL);
        if (MMPROJ != null) {
            params = params.withMmproj(MMPROJ);
        }
        return LlamaAudio.open(params);
    }

    /**
     * Wraps text in the turn markers Gemma expects. The library sends prompts as they are given,
     * so this belongs in the test rather than in the binding.
     */
    private static String gemmaUserTurn(String text) {
        return "<start_of_turn>user\n" + text + "<end_of_turn>\n<start_of_turn>model\n";
    }

    /** Reads a WAV file as one channel of samples between -1 and 1. */
    private static float[] readWav(Path path, int expectedSampleRate)
            throws IOException, UnsupportedAudioFileException {
        try (AudioInputStream stream = AudioSystem.getAudioInputStream(path.toFile())) {
            AudioFormat format = stream.getFormat();
            assertEquals(expectedSampleRate, (int) format.getSampleRate(),
                    "the model wants a different sample rate than the file holds");
            assertEquals(16, format.getSampleSizeInBits(), "only 16 bit samples are handled here");

            byte[] bytes = stream.readAllBytes();
            int channels = format.getChannels();
            int frames = bytes.length / (2 * channels);

            float[] samples = new float[frames];
            for (int frame = 0; frame < frames; frame++) {
                // Keep the first channel, which is all the model reads.
                int at = frame * 2 * channels;
                int sample = format.isBigEndian()
                        ? (bytes[at] << 8) | (bytes[at + 1] & 0xff)
                        : (bytes[at + 1] << 8) | (bytes[at] & 0xff);
                samples[frame] = (short) sample / 32768.0f;
            }
            return samples;
        }
    }

    private static Path pathProperty(String name) {
        String value = System.getProperty(name);
        if (value == null || value.isBlank()) {
            return null;
        }
        Path path = Path.of(value);
        assertTrue(Files.isReadable(path), name + " points at " + path + ", which cannot be read");
        return path;
    }
}
