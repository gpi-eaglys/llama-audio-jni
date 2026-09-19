# Java Bindings for Llama Audio Processing

Runs a local language model from Java, with audio as an input, using llama.cpp through JNI. No
server, no HTTP calls: the model runs inside the JVM process.

Built for models that read audio directly, such as Gemma 4, so speech can be sent to the model
as samples instead of being turned into text first by a separate speech recognizer.

## Project Structure

| Part | Directory | What it is |
| --- | --- | --- |
| Native | `native/` | A C interface over llama.cpp. No JNI, no Java. Has its own test program. |
| Java | `java/` | A thin JNI layer plus the Java classes. Converts types and calls the C interface. |


## Dependencies 

- **`llama.h`** — llama.cpp's published C interface. Models, tokens, sampling, key-value cache.
- **`mtmd.h`** — llama.cpp's multimodal library, which turns audio into something the model reads.


## Requirements

- CMake 3.16 or newer
- A C++17 compiler
- JDK 25 and Maven, for the Java part. `JAVA_HOME` has to point at the JDK 25, since both
  the Maven build and the CMake search for `jni.h` follow it.
- Access to `github.com`, unless you point the build at a llama.cpp checkout you already have

## Building the native part

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

This downloads llama.cpp at the pinned release, builds it, and builds this project on top.

### On a machine without access to github.com

Copy a llama.cpp checkout to the machine yourself and point the build at it. Nothing is
downloaded in that case:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_AUDIO_LLAMA_CPP_SOURCE=/path/to/llama.cpp
cmake --build build -j"$(nproc)"
```

### The pinned llama.cpp release

**`b10670`**, set in `LLAMA_AUDIO_LLAMA_CPP_TAG` in the top level `CMakeLists.txt`.

Keep this written down and move it deliberately. Anything older than roughly April 2026 does not
know the `gemma4` architecture and cannot load Gemma 4 at all.

## Building the Java part

Build the native part first, then:

```sh
export JAVA_HOME=/path/to/jdk-25   # on Debian and Ubuntu, /usr/lib/jvm/java-25-openjdk-amd64
mvn -f java/pom.xml package
```

The CMake build writes `libllama-audio-jni.so` straight into
`java/src/main/resources/com/eaglys/llamaaudio/<os>/<arch>/`, so Maven picks it up as an ordinary
resource and puts it in the jar. At run time `NativeLibrary` looks in three places, in this
order: the file named by the `llama.audio.library.path` system property, `java.library.path`, and
the copy in the jar, which it unpacks to a temporary file.

The library carries the C interface and all of llama.cpp statically and exports only its seven
JNI entry points, so a JVM that also loads another llama.cpp binding does not end up with two
sets of the same symbols.

## Running on a GPU

Build llama.cpp's CUDA backend and tell the model how much to put on the card:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j"$(nproc)"
```

```java
ModelParams params = ModelParams.of("gemma-4-E2B_q4_0-it.gguf")
        .withMmproj("gemma-4-E2B-it-mmproj.gguf")
        .withGpuLayers(99)        // 0, the default, runs everything on the CPU
        .withMmprojUseGpu(true);  // runs the audio encoder on the card as well
```

`withGpuLayers` and `withMmprojUseGpu` are separate because the audio encoder and the model can
be placed independently.

## Using it from Java

```java
ModelParams params = ModelParams.of("gemma-4-E2B_q4_0-it.gguf")
        .withMmproj("gemma-4-E2B-it-mmproj.gguf");

try (LlamaAudio model = LlamaAudio.open(params)) {
    String prompt = "<start_of_turn>user\n"
            + LlamaAudio.mediaMarker() + "\nTranscribe the audio.<end_of_turn>\n"
            + "<start_of_turn>model\n";

    String transcript = model.generateWithAudio(prompt, samples);
}
```

`samples` is a `float[]` of one channel between -1 and 1, at `model.audioSampleRate()`. A failed
call throws `LlamaAudioException`, which carries the `Status` the C interface returned.

One instance is for one thread at a time, and it holds a model in native memory that the garbage
collector knows nothing about, so close it.

## Checking it works

The native test program loads a model and generates from text and from audio:

```sh
./build/native/test/llama-audio-smoke model.gguf \
    --mmproj mmproj.gguf \
    --audio sample.wav \
    --prompt "Transcribe the audio."
```

It prints `PASS` or `FAIL` per check and exits non-zero when anything failed. A call that reports
success but generates no text counts as a failure, because that is how a wrongly built prompt
shows up.

To run it through `ctest`, give the file paths at configure time:

```sh
cmake -B build -DLLAMA_AUDIO_TEST_MODEL=model.gguf \
               -DLLAMA_AUDIO_TEST_MMPROJ=mmproj.gguf \
               -DLLAMA_AUDIO_TEST_AUDIO=sample.wav
ctest --test-dir build --output-on-failure
```

The Java tests check the same things through the binding. They need the same files, and are
skipped when they are not given:

```sh
mvn -f java/pom.xml test \
    -Dllama.audio.test.model=model.gguf \
    -Dllama.audio.test.mmproj=mmproj.gguf \
    -Dllama.audio.test.audio=sample.wav
```

The audio file has to be 16 bit PCM at the rate the model reports.

## Prompts are sent as they are given

The library does not add turn or role markers to a prompt. Which markers a model wants belongs to
the model, not to a wrapper, so putting them in is the caller's job.

This matters more than it sounds. An instruction trained model given a bare prompt with no turn
markers replies with an end-of-turn token and generates nothing at all — a call that succeeds and
returns an empty string. For Gemma, wrap a prompt like this:

```
<start_of_turn>user
Transcribe the audio.<end_of_turn>
<start_of_turn>model
```

`native/test/smoke.cpp` shows this, including where the audio marker goes inside the user turn.

## Audio format

One channel, floating point samples between -1 and 1, at the rate the model reports through
`la_audio_sample_rate` (16000 Hz for Gemma 4).

Models limit how much audio they read at once. Gemma 4 handles 30 seconds per piece, at 25 tokens
per second of audio; longer audio is split and read in parts.

## Reusing work across calls

Text calls on the same context skip the part of the prompt that repeats the previous prompt. A
conversation that grows as it is transcribed costs only the new text, not the whole transcript
again. Turn it off with `reuse_prefix`.

Audio calls always start with an empty cache, because audio becomes embeddings rather than tokens,
so there is no token prefix to compare against.

## License

MIT. llama.cpp is MIT as well.

