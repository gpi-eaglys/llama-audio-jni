package com.eaglys.llamaaudio;

/**
 * How a call into the native library ended.
 *
 * <p>Mirrors {@code la_status} in {@code native/include/llama_audio.h}. The codes are part of
 * the C interface, so they are written out here rather than taken from the ordinal.
 */
public enum Status {
    OK(0),
    INVALID_ARGUMENT(1),
    LOAD_FAILED(2),
    /** The model was opened without a projector file, so it cannot read audio. */
    NO_AUDIO(3),
    TOKENIZE_FAILED(4),
    DECODE_FAILED(5),
    /** The prompt does not fit in the context window. */
    CONTEXT_FULL(6),
    OUT_OF_MEMORY(7),
    /** A code this binding does not know, which means the C interface has moved on. */
    UNKNOWN(-1);

    private final int code;

    Status(int code) {
        this.code = code;
    }

    /** The value this status has in the C interface. */
    public int code() {
        return code;
    }

    static Status fromCode(int code) {
        for (Status status : values()) {
            if (status.code == code) {
                return status;
            }
        }
        return UNKNOWN;
    }
}
