package org.loopsn2n.fiji;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;

/** No Python execution: portable paths, Unicode/spaces, overrides and fail-closed checks. */
public final class CalibratedRuntimeSmokeTest {
    public static void main(String[] args) throws Exception {
        for (String name : new String[] {"staticRepeats", "trustCheckpoint", "projectDirectory", "pythonExecutable"}) {
            org.scijava.plugin.Parameter parameter = LoopSN2NCalibratedCommand.class.getDeclaredField(name).getAnnotation(org.scijava.plugin.Parameter.class);
            require(parameter != null && !parameter.persist(), "Stale machine/domain input may persist: " + name);
        }
        java.lang.reflect.Field trust = LoopSN2NCalibratedCommand.class.getDeclaredField("trustCheckpoint");
        trust.setAccessible(true);
        require(!trust.getBoolean(new LoopSN2NCalibratedCommand()), "Legacy checkpoint trust must default to false");
        Path root = Files.createDirectories(Paths.get(args[0]).resolve("portable package 空格"));
        Path plugin = Files.createDirectories(root.resolve("plugin"));
        Path backend = Files.createDirectories(root.resolve("runtime/fiji_calibrated_plugin/backend"));
        Files.write(backend.resolve("bridge.py"), new byte[] {0});
        Path python = root.resolve("Python 空格.exe");
        Files.write(python, new byte[] {0});
        Path config = plugin.resolve("runtime_config.json");
        Files.write(config, "{\"backend_root\":\"../runtime\",\"python_executable\":\"../Python 空格.exe\"}".getBytes(StandardCharsets.UTF_8));
        RuntimeResolver.Settings configured = RuntimeResolver.resolveAt(plugin, Arrays.asList(config), null, null);
        require(configured.backend.equals(root.resolve("runtime").toAbsolutePath()), "Relative backend failed");
        require(configured.python.equals(python.toAbsolutePath()), "Relative Python failed");
        RuntimeResolver.Settings adjacent = RuntimeResolver.resolveAt(plugin, Collections.<Path>emptyList(), null, python.toFile());
        require(adjacent.backend.equals(configured.backend), "Adjacent runtime detection failed");
        RuntimeResolver.Settings explicit = RuntimeResolver.resolveAt(plugin, Arrays.asList(config), root.resolve("runtime").toFile(), python.toFile());
        require(explicit.python.equals(configured.python), "Explicit override failed");
        try { RuntimeResolver.resolveAt(plugin, Arrays.asList(config), root.resolve("missing").toFile(), python.toFile()); throw new AssertionError("Missing explicit backend accepted"); }
        catch (java.io.IOException expected) { }
        Path moved = root.resolveSibling("renamed release 中文 空格");
        Files.move(root, moved);
        RuntimeResolver.Settings renamed = RuntimeResolver.resolveAt(moved.resolve("plugin"), Arrays.asList(moved.resolve("plugin/runtime_config.json")), null, null);
        require(renamed.backend.equals(moved.resolve("runtime").toAbsolutePath()), "Renamed release backend failed");
        require(renamed.python.equals(moved.resolve("Python 空格.exe").toAbsolutePath()), "Renamed release Python failed");
        System.out.println("FIJI_RUNTIME_SMOKE_OK relative=PASS adjacent=PASS overrides=PASS missing_rejected=PASS renamed_unicode_spaces=PASS stale_preferences_disabled=PASS");
    }
    private static void require(boolean value, String message) { if (!value) throw new AssertionError(message); }
}
