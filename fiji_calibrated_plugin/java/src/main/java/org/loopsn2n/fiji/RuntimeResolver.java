package org.loopsn2n.fiji;

import com.google.gson.JsonObject;
import ij.IJ;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/** Locate a relocatable release or an explicitly configured installation. No shell is used. */
public final class RuntimeResolver {
    private RuntimeResolver() { }
    public static final class Settings {
        public final Path backend;
        public final Path python;
        Settings(Path backend, Path python) { this.backend = backend; this.python = python; }
    }

    public static Settings resolve(File backendOverride, File pythonOverride) throws Exception {
        Path own = Paths.get(LoopSN2NCalibratedCommand.class.getProtectionDomain().getCodeSource().getLocation().toURI());
        Path anchor = Files.isDirectory(own) ? own : own.getParent();
        List<Path> configs = new ArrayList<Path>();
        configs.add(anchor.resolve("runtime_config.json"));
        String fiji = IJ.getDirectory("imagej");
        if (fiji != null) configs.add(Paths.get(fiji).resolve("plugins/LoopSN2N/runtime_config.json"));
        return resolveAt(anchor, configs, backendOverride, pythonOverride);
    }

    public static Settings resolveAt(Path anchor, List<Path> configs, File backendOverride, File pythonOverride) throws IOException {
        Path backend = backendOverride == null ? null : backendOverride.toPath().toAbsolutePath().normalize();
        Path python = pythonOverride == null ? null : pythonOverride.toPath().toAbsolutePath().normalize();
        for (Path config : configs) {
            if (!Files.isRegularFile(config)) continue;
            JsonObject values = LoopSN2NCalibratedCommand.readResponse(config);
            if (backend == null && values.has("backend_root")) backend = resolveValue(config, values.get("backend_root").getAsString());
            if (python == null && values.has("python_executable")) python = resolveValue(config, values.get("python_executable").getAsString());
            break;
        }
        if (backend == null) {
            for (Path candidate : new Path[] {anchor.resolve("runtime"), anchor.resolve("../runtime"), anchor.resolve("../../runtime")}) {
                if (Files.isRegularFile(candidate.resolve("fiji_calibrated_plugin/backend/bridge.py"))) {
                    backend = candidate.toAbsolutePath().normalize(); break;
                }
            }
        }
        if (backend == null || !Files.isRegularFile(backend.resolve("fiji_calibrated_plugin/backend/bridge.py")))
            throw new IOException("Calibrated backend not found. Run python scripts/install.py --fiji <Fiji-directory> --python <Python-executable> from the complete release, or choose Advanced backend override.");
        if (python == null) {
            List<Path> candidates = new ArrayList<Path>();
            candidates.add(backend.resolve(".venv/Scripts/python.exe"));
            candidates.add(backend.resolve("../.venv/Scripts/python.exe"));
            candidates.add(backend.resolve(".venv/bin/python"));
            String conda = System.getenv("CONDA_PREFIX");
            if (conda != null) {
                candidates.add(Paths.get(conda).resolve("python.exe"));
                candidates.add(Paths.get(conda).resolve("bin/python"));
            }
            String path = System.getenv("PATH");
            if (path != null) for (String item : path.split(java.util.regex.Pattern.quote(File.pathSeparator))) {
                if (!item.isEmpty()) for (String exe : new String[] {"python.exe", "python3", "python"}) candidates.add(Paths.get(item).resolve(exe));
            }
            for (Path candidate : candidates) if (Files.isRegularFile(candidate)) { python = candidate.toAbsolutePath().normalize(); break; }
        }
        if (python == null || !Files.isRegularFile(python))
            throw new IOException("Python executable not found. Install dependencies and rerun python scripts/install.py --fiji <Fiji-directory> --python <Python-executable>, or choose Advanced Python override.");
        return new Settings(backend, python);
    }

    private static Path resolveValue(Path config, String value) {
        Path path = Paths.get(value);
        return (path.isAbsolute() ? path : config.toAbsolutePath().getParent().resolve(path)).normalize();
    }
}
