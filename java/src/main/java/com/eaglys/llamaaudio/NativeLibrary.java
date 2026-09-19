package com.eaglys.llamaaudio;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

/**
 * Finds and loads the shared library holding the JNI code.
 *
 * <p>Three places are tried, in this order: the path in the {@code llama.audio.library.path}
 * system property, the usual library search path, and finally a copy carried inside the jar,
 * which is unpacked to a temporary file. The first of these is what to use when the library was
 * built somewhere the search path does not reach, which is the normal case straight after a
 * CMake build.
 */
final class NativeLibrary {

    /** Loads a shared library from this exact file, skipping the search path and the jar. */
    static final String PATH_PROPERTY = "llama.audio.library.path";

    private static final String LIBRARY_NAME = "llama-audio-jni";

    private static boolean loaded;

    private NativeLibrary() {
    }

    static synchronized void load() {
        if (loaded) {
            return;
        }

        String explicit = System.getProperty(PATH_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            System.load(Path.of(explicit).toAbsolutePath().toString());
            loaded = true;
            return;
        }

        UnsatisfiedLinkError fromSearchPath;
        try {
            System.loadLibrary(LIBRARY_NAME);
            loaded = true;
            return;
        } catch (UnsatisfiedLinkError error) {
            fromSearchPath = error;
        }

        try {
            loadFromJar();
            loaded = true;
        } catch (IOException | UnsatisfiedLinkError error) {
            UnsatisfiedLinkError failure = new UnsatisfiedLinkError(
                    "could not load " + System.mapLibraryName(LIBRARY_NAME) + " for " + platform()
                            + ". Build it with CMake and point -D" + PATH_PROPERTY
                            + " at the file, or put it on java.library.path.");
            failure.addSuppressed(fromSearchPath);
            failure.addSuppressed(error);
            throw failure;
        }
    }

    /**
     * Unpacks the library from the jar and loads it. The file is left behind on exit on platforms
     * that hold shared libraries open, so it is asked to be deleted rather than deleted outright.
     */
    private static void loadFromJar() throws IOException {
        String fileName = System.mapLibraryName(LIBRARY_NAME);
        String resource = "/com/eaglys/llamaaudio/" + platform() + "/" + fileName;

        Path unpacked;
        try (InputStream source = NativeLibrary.class.getResourceAsStream(resource)) {
            if (source == null) {
                throw new IOException("the jar carries no " + resource);
            }
            int dot = fileName.lastIndexOf('.');
            unpacked = Files.createTempFile(LIBRARY_NAME, dot < 0 ? "" : fileName.substring(dot));
            Files.copy(source, unpacked, StandardCopyOption.REPLACE_EXISTING);
        }
        unpacked.toFile().deleteOnExit();

        System.load(unpacked.toAbsolutePath().toString());
    }

    /** The directory name the build writes the library under, such as {@code linux/x86_64}. */
    private static String platform() {
        return osName() + "/" + osArch();
    }

    private static String osName() {
        String name = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (name.contains("linux")) {
            return "linux";
        }
        if (name.contains("mac") || name.contains("darwin")) {
            return "macos";
        }
        if (name.contains("windows")) {
            return "windows";
        }
        return name.replaceAll("[^a-z0-9]+", "");
    }

    private static String osArch() {
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        return switch (arch) {
            case "amd64", "x86_64" -> "x86_64";
            case "aarch64", "arm64" -> "aarch64";
            default -> arch.replaceAll("[^a-z0-9]+", "");
        };
    }
}
