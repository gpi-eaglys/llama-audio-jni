# Development notes

Background that does not belong in the README: why things are the way they are, and what is
still open. The README covers how to build and use the project.

## Where it stands, 2026-09-20

| Part | State |
| --- | --- |
| `native/` | Done. The C interface and its standalone smoke test. |
| `java/` | Done. JNI layer, Java classes, JUnit tests. |

The Java tests were run against Gemma 4 E2B (`q4_0`) on the CPU: 8 tests, all passing, about two
minutes, audio included. Nothing has been run on a GPU yet.

## Decisions worth keeping

**Text crosses the JNI boundary as `byte[]`, not `String`.** JNI's own string calls speak
modified UTF-8, which is not the UTF-8 llama.cpp reads and writes. The two differ on every
character outside the basic multilingual plane, so `NewStringUTF` on a reply holding an emoji
produces something invalid. Encoding and decoding both happen in Java, in `LlamaAudio`.

**The shared library exports only its seven JNI entry points.** Hidden visibility plus
`-Wl,--exclude-libs,ALL`. All of llama.cpp is linked in statically, and `stereo-rag` also pulls
in `de.kherud:llama`, which carries its own copy. Without the symbols hidden, one JVM loading
both bindings gets two sets of the same names and picks whichever loaded first. Check it after
touching the link options:

```sh
nm -D --defined-only libllama-audio-jni.so | grep ' T ' | grep -v ' T Java_'
```

That has to print nothing.

**Audio samples are copied out of the `float[]`, not pinned.** Generating takes seconds, and a
pinned array stops the garbage collector moving anything for all of it. 30 seconds at 16 kHz is
under 2 MB, so the copy costs nothing worth measuring.

**The defaults in `ModelParams` and `GenerateParams` are written out in Java.** They duplicate
`la_model_params_default` and `la_generate_params_default`, and have to be changed together with
them. The JNI layer cannot fill them in instead, because Java has to send a value for every
field either way.

**Prompts are passed through without turn markers**, the same as the C interface. The Gemma
markers live in the tests, not in the binding. See the README for why.

## Not done yet

- **Never run on a GPU.** `withGpuLayers` and `withMmprojUseGpu` reach the C interface, and the C
  interface passes them to llama.cpp, but no run has gone through CUDA.
- **Only `linux/x86_64` has been built.** The loader and the CMake output path already handle
  `macos` and `windows`, and `aarch64`; none of it has been tried.
- **Generation cannot be watched or stopped.** A call runs to `maxTokens` or to an end-of-turn
  token and returns everything at once. A token callback would need the C interface to grow one
  first, and the callback would have to attach the calling thread to the JVM.
- **Nothing stops two threads using one instance.** The C interface asks for one thread at a
  time and the Java class says so, but a second caller is not turned away, and what happens then
  is a corrupted key-value cache rather than an exception.
- **Audio longer than the model reads at once** is handed to mtmd, which splits it. The only clip
  tested here was 10 seconds, well inside Gemma's 30.
- **The jar holds no library for any platform it was not built on.** Building a jar that runs
  anywhere means building the shared library on each platform and collecting them before
  packaging.

## Running the tests without network access

The build needs llama.cpp `b10670` on disk, and the tests need a model:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_AUDIO_LLAMA_CPP_SOURCE=/path/to/llama.cpp
```

The audio file has to be 16 bit PCM at the model's rate, 16 kHz for Gemma 4. `javax.sound`
reads it in the test, so a plain WAV is enough; anything else has to be converted first.

## Next

1. Run the Java tests on the GPU machine with `-DGGML_CUDA=ON`, `withGpuLayers(99)` and
   `withMmprojUseGpu(true)`, and see whether the audio encoder on the card changes the output.
2. Decide whether `stereo-rag` moves off `de.kherud:llama` and onto this, which is what it was
   written for.
