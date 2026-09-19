package com.eaglys.llamaaudio;

import java.io.Serial;
import java.nio.charset.StandardCharsets;

/** Thrown when a call into the native library fails. */
public class LlamaAudioException extends RuntimeException {

    @Serial
    private static final long serialVersionUID = 1L;

    private final Status status;

    LlamaAudioException(Status status, String message) {
        super(message == null || message.isEmpty() ? status.name() : message);
        this.status = status;
    }

    /**
     * Builds the exception the native code throws. The message arrives as UTF-8 bytes rather
     * than as a string because it can hold a file path, and every string crossing this boundary
     * is decoded in one place.
     */
    static LlamaAudioException of(int code, byte[] utf8Message) {
        String message = utf8Message == null ? "" : new String(utf8Message, StandardCharsets.UTF_8);
        return new LlamaAudioException(Status.fromCode(code), message);
    }

    /** How the call ended. */
    public Status status() {
        return status;
    }
}
