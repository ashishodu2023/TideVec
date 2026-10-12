package io.tidevec;

import java.io.IOException;

/** A non-2xx response from the TideVec server. */
public class TideVecException extends IOException {
    private final int status;
    private final String leader;

    public TideVecException(int status, String message, String leader) {
        super("TideVec error " + status + ": " + message);
        this.status = status;
        this.leader = leader;
    }

    /** HTTP status code. */
    public int status() { return status; }

    /** Leader URL on a 421 (cluster mode), otherwise null. */
    public String leader() { return leader; }

    /** True for 404: unknown collection or endpoint. */
    public boolean isNotFound() { return status == 404; }
}
