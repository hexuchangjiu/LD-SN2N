package org.loopsn2n.fiji;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

/** Verify the real installer's configuration without GUI or checkpoint execution. */
public final class CalibratedInstalledRuntimeHarness {
    private CalibratedInstalledRuntimeHarness() { }
    public static void main(String[] args) throws Exception {
        if (args.length != 2)
            throw new IllegalArgumentException("usage: CalibratedInstalledRuntimeHarness <Fiji/plugins/LoopSN2N> <expected-python>");
        Path plugin = Paths.get(args[0]).toAbsolutePath().normalize();
        RuntimeResolver.Settings settings = RuntimeResolver.resolveAt(plugin,
            Arrays.asList(plugin.resolve("runtime_config.json")), null, null);
        if (!settings.backend.equals(plugin.resolve("runtime")))
            throw new AssertionError("Installed backend root does not match the plugin directory");
        if (!settings.python.equals(Paths.get(args[1]).toAbsolutePath().normalize()))
            throw new AssertionError("Installed Python selection changed");
        if (!Files.isRegularFile(settings.backend.resolve("scripts/check_runtime.py")))
            throw new AssertionError("Installed runtime preflight missing");
        System.out.println("FIJI_INSTALLED_RUNTIME_OK backend=" + settings.backend + " python=" + settings.python);
    }
}
