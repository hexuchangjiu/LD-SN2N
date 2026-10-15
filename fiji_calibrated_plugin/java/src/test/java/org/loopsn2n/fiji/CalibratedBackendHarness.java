package org.loopsn2n.fiji;

import com.google.gson.JsonObject;
import java.nio.file.Paths;

/** Uses exactly the Java command's Python launch, log, exit and response-validation path. */
public final class CalibratedBackendHarness {
    private CalibratedBackendHarness() { }
    public static void main(String[] args) throws Exception {
        if (args.length != 3)
            throw new IllegalArgumentException("usage: CalibratedBackendHarness <request.json> <python.exe> <project-directory>");
        JsonObject result = LoopSN2NCalibratedCommand.runBackendAndValidate(
            Paths.get(args[0]), Paths.get(args[1]), Paths.get(args[2]));
        System.out.println("FIJI_CALIBRATED_BACKEND_TERMINAL " + result.toString());
        if ("FAILED".equals(result.get("status").getAsString())) System.exit(1);
    }
}
