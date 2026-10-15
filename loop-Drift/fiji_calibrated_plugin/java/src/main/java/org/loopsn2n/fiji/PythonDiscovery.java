package org.loopsn2n.fiji;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CancellationException;
import java.util.concurrent.TimeUnit;

/** Bounded discovery of existing environments only; never installs packages or invokes a shell. */
final class PythonDiscovery {
    private static final int MAX_CANDIDATES = 30;
    private static final long TOTAL_MS = 180000;
    private static final long PROBE_MS = 45000;
    private static final Gson JSON = new Gson();
    interface Probe { Result check(Path python, boolean requireCuda, long timeoutMillis) throws Exception; }
    static final class Result {
        final boolean ok, cuda;
        final Path python;
        final String detail;
        Result(boolean ok, boolean cuda, Path python, String detail) { this.ok = ok; this.cuda = cuda; this.python = python; this.detail = detail; }
    }
    static final class Selection {
        final Path python;
        final String device;
        Selection(Path python, String device) { this.python = python; this.device = device; }
    }
    private PythonDiscovery() { }

    static Selection discover(Path backend, Path anchor, List<Path> configs, File override, String device,
            Path job, Path selectionFile, RuntimeResolver.Monitor monitor) throws Exception {
        final Path scratch = Files.createDirectories(job.resolve(".cache/runtime-discovery"));
        final long start = System.nanoTime();
        Path cached = readCached(selectionFile, monitor);
        List<Path> candidates = override == null ? candidates(backend, anchor, configs, cached, scratch, monitor) : new ArrayList<Path>();
        Probe probe = new ProcessProbe(backend, scratch, monitor);
        Selection result;
        try {
            result = select(candidates, override == null ? null : override.toPath().toAbsolutePath().normalize(),
                cached, device, probe, monitor, Math.max(1, TOTAL_MS - elapsedMillis(start)));
        }
        catch (IOException exc) {
            throw new IOException(exc.getMessage()
                .replace("<embedded-runtime>/requirements.txt", backend.resolve("requirements.txt").toString())
                .replace("job/.cache/runtime-discovery", scratch.toString()), exc);
        }
        persist(selectionFile, result, monitor);
        monitor.progress("Validated Python: " + result.python + " (" + result.device + ")");
        return result;
    }

    static Selection select(List<Path> candidates, Path explicit, Path cached, String device, Probe probe,
            RuntimeResolver.Monitor monitor, long budgetMillis) throws Exception {
        if (!Arrays.asList("auto", "cpu", "cuda").contains(device)) throw new IOException("Device must be auto, cpu, or cuda.");
        if (explicit != null) candidates = Arrays.asList(explicit);
        long started = System.nanoTime();
        Selection cpuFallback = null;
        StringBuilder failures = new StringBuilder();
        int checked = 0;
        for (Path candidate : candidates) {
            monitor.checkCancelled();
            long remaining = budgetMillis - elapsedMillis(started);
            if (remaining <= 0 || checked >= MAX_CANDIDATES) break;
            checked++;
            monitor.progress("Checking existing Python " + checked + ": " + candidate);
            Result result;
            try { result = probe.check(candidate, "cuda".equals(device), Math.min(PROBE_MS, remaining)); }
            catch (CancellationException exc) { throw exc; }
            catch (InterruptedException exc) { Thread.currentThread().interrupt(); throw exc; }
            catch (Exception exc) { result = new Result(false, false, candidate, exc.getClass().getSimpleName() + ": " + exc.getMessage()); }
            if (result.ok) {
                if ("cpu".equals(device)) return new Selection(result.python, "cpu");
                if ("cuda".equals(device)) {
                    if (result.cuda) return new Selection(result.python, "cuda");
                    result = new Result(false, false, candidate, "CUDA tensor validation was not confirmed");
                }
                else {
                    remaining = budgetMillis - elapsedMillis(started);
                    if (result.cuda && remaining > 0) {
                        try {
                            Result gpu = probe.check(result.python, true, Math.min(PROBE_MS, remaining));
                            if (gpu.ok && gpu.cuda) return new Selection(gpu.python, "cuda");
                            monitor.progress("CUDA execution did not pass for " + candidate + "; keeping CPU fallback.");
                        }
                        catch (CancellationException exc) { throw exc; }
                        catch (InterruptedException exc) { Thread.currentThread().interrupt(); throw exc; }
                        catch (Exception exc) { monitor.progress("CUDA check failed; CPU remains available: " + exc.getMessage()); }
                    }
                    if (cpuFallback == null) cpuFallback = new Selection(result.python, "cpu");
                    // Explicit selection never switches interpreters. A remembered valid CPU choice is
                    // reused for predictable/fast starts; clear the selection or override to rescan.
                    if (explicit != null || candidate.equals(cached)) return cpuFallback;
                    continue;
                }
            }
            if (failures.length() < 8000) failures.append("\n - ").append(candidate).append(": ").append(result.detail);
            monitor.progress("Skipping unusable Python: " + candidate + " (see .cache/runtime-discovery logs)");
        }
        if (cpuFallback != null) return cpuFallback;
        String prefix = explicit != null ? "The explicitly selected Python is not usable; no other interpreter was selected."
            : "No compatible existing Python environment was found after checking " + checked + " candidate(s).";
        throw new IOException(prefix + "\nUse Python 3.10+ with PyTorch 2.6+ and numpy, scipy, tifffile, scikit-image, matplotlib, Pillow."
            + "\nOne-time setup: run <your-python> -m pip install -r \"<embedded-runtime>/requirements.txt\" in your chosen environment,"
            + " then select its Python executable in Advanced override. This plugin never installs dependencies automatically."
            + ("cuda".equals(device) ? " CUDA mode also requires a working CUDA tensor test; choose cpu if appropriate." : "")
            + "\nDetailed probes: job/.cache/runtime-discovery" + failures);
    }

    private static List<Path> candidates(Path backend, Path anchor, List<Path> configs, Path cached, Path scratch,
            RuntimeResolver.Monitor monitor) throws Exception {
        Set<Path> paths = new LinkedHashSet<Path>();
        add(paths, cached);
        for (Path config : configs) {
            if (!Files.isRegularFile(config)) continue;
            try {
                JsonObject data = LoopSN2NCalibratedCommand.readResponse(config);
                if (data.has("python_executable")) {
                    Path value = Paths.get(data.get("python_executable").getAsString());
                    add(paths, value.isAbsolute() ? value : config.toAbsolutePath().getParent().resolve(value));
                }
            }
            catch (Exception exc) { monitor.progress("Ignoring stale Python configuration: " + config); }
        }
        for (Path base : Arrays.asList(backend, anchor)) {
            for (int depth = 0; depth < 4 && base != null; depth++, base = base.getParent()) {
                addEnvironment(paths, base.resolve(".venv"));
                addEnvironment(paths, base.resolve("venv"));
            }
        }
        for (String name : new String[] {"CONDA_PREFIX", "VIRTUAL_ENV"}) {
            String value = System.getenv(name); if (value != null && !value.isEmpty()) addEnvironment(paths, Paths.get(value));
        }
        String path = System.getenv("PATH");
        Path launcher = null;
        if (path != null) for (String directory : path.split(java.util.regex.Pattern.quote(File.pathSeparator))) {
            if (directory.isEmpty()) continue;
            try {
                Path base = Paths.get(directory);
                for (String name : new String[] {"python.exe", "python3", "python"}) add(paths, base.resolve(name));
                if (launcher == null && Files.isRegularFile(base.resolve("py.exe"))) launcher = base.resolve("py.exe");
            }
            catch (java.nio.file.InvalidPathException ignored) { }
        }
        Path home = Paths.get(System.getProperty("user.home"));
        String localAppData = System.getenv("LOCALAPPDATA");
        if (launcher == null && localAppData != null) {
            Path candidate = Paths.get(localAppData).resolve("Programs/Python/Launcher/py.exe");
            if (Files.isRegularFile(candidate)) launcher = candidate;
        }
        if (launcher == null && System.getenv("SystemRoot") != null) {
            Path candidate = Paths.get(System.getenv("SystemRoot")).resolve("py.exe");
            if (Files.isRegularFile(candidate)) launcher = candidate;
        }
        if (launcher != null) {
            Path log = scratch.resolve("python-launcher-" + UUID.randomUUID() + ".log");
            try {
                runBounded(new ProcessBuilder(launcher.toString(), "-0p"), log, scratch, 8000, monitor);
                if (Files.size(log) <= 256 * 1024) for (String line : Files.readAllLines(log, StandardCharsets.UTF_8)) {
                    java.util.regex.Matcher match = java.util.regex.Pattern.compile("([A-Za-z]:[\\\\/].*?python(?:[0-9.]*)?\\.exe)\\s*$", java.util.regex.Pattern.CASE_INSENSITIVE).matcher(line);
                    if (match.find()) add(paths, Paths.get(match.group(1)));
                }
            }
            catch (CancellationException exc) { throw exc; }
            catch (InterruptedException exc) { Thread.currentThread().interrupt(); throw exc; }
            catch (Exception exc) { monitor.progress("Python launcher discovery unavailable; checking other existing locations."); }
        }
        Path environments = home.resolve(".conda/environments.txt");
        if (Files.isRegularFile(environments) && Files.size(environments) <= 128 * 1024) {
            int count = 0;
            for (String line : Files.readAllLines(environments, StandardCharsets.UTF_8)) {
                if (++count > 100) break;
                try { if (!line.trim().isEmpty()) addEnvironment(paths, Paths.get(line.trim())); }
                catch (java.nio.file.InvalidPathException ignored) { }
            }
        }
        List<Path> roots = new ArrayList<Path>(Arrays.asList(home.resolve("miniconda3"), home.resolve("anaconda3"),
            home.resolve("Miniforge3"), home.resolve("mambaforge"), home.resolve(".conda")));
        if (localAppData != null) roots.add(Paths.get(localAppData).resolve("Continuum/anaconda3"));
        for (Path root : roots) {
            addEnvironment(paths, root);
            Path envs = root.resolve("envs");
            if (Files.isDirectory(envs)) try (DirectoryStream<Path> children = Files.newDirectoryStream(envs)) {
                int count = 0; for (Path child : children) { if (++count > 40) break; addEnvironment(paths, child); }
            }
            catch (IOException ignored) { }
        }
        if (localAppData != null) {
            Path root = Paths.get(localAppData).resolve("Programs/Python");
            if (Files.isDirectory(root)) try (DirectoryStream<Path> children = Files.newDirectoryStream(root)) {
                int count = 0; for (Path child : children) { if (++count > 40) break; addEnvironment(paths, child); }
            }
            catch (IOException ignored) { }
        }
        return new ArrayList<Path>(paths);
    }

    private static void addEnvironment(Set<Path> paths, Path root) {
        for (String name : new String[] {"python.exe", "Scripts/python.exe", "bin/python3", "bin/python"}) add(paths, root.resolve(name));
    }
    private static void add(Set<Path> paths, Path candidate) {
        if (candidate == null || paths.size() >= MAX_CANDIDATES) return;
        candidate = candidate.toAbsolutePath().normalize();
        if (!candidate.toString().toLowerCase(java.util.Locale.ROOT).contains("windowsapps") && Files.isRegularFile(candidate)) paths.add(candidate);
    }

    private static Path readCached(Path file, RuntimeResolver.Monitor monitor) {
        try {
            EmbeddedRuntime.assertSafePath(file);
            if (!Files.isRegularFile(file)) return null;
            if (Files.size(file) > 128 * 1024) throw new IOException("Oversized remembered Python configuration");
            try (Reader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
                JsonObject data = JSON.fromJson(reader, JsonObject.class);
                return Paths.get(data.get("python_executable").getAsString()).toAbsolutePath().normalize();
            }
        }
        catch (Exception exc) { monitor.progress("Remembered Python is stale; searching existing environments."); return null; }
    }
    private static void persist(Path file, Selection selection, RuntimeResolver.Monitor monitor) {
        try {
            EmbeddedRuntime.safeDirectories(file.getParent()); EmbeddedRuntime.assertSafePath(file);
            JsonObject data = new JsonObject();
            data.addProperty("python_executable", selection.python.toString());
            data.addProperty("validated_device", selection.device);
            data.addProperty("auto_policy", "prefer tested CUDA on discovery; reuse revalidated remembered Python on later starts");
            Path temporary = Files.createTempFile(file.getParent(), "python-selection-", ".tmp");
            try (Writer writer = Files.newBufferedWriter(temporary, StandardCharsets.UTF_8)) { JSON.toJson(data, writer); }
            Files.move(temporary, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        }
        catch (IOException exc) { monitor.progress("Python is usable but could not be remembered: " + exc.getMessage()); }
    }

    static final class ProcessProbe implements Probe {
        private final Path backend, scratch;
        private final RuntimeResolver.Monitor monitor;
        ProcessProbe(Path backend, Path scratch, RuntimeResolver.Monitor monitor) { this.backend = backend; this.scratch = scratch; this.monitor = monitor; }
        @Override public Result check(Path python, boolean requireCuda, long timeoutMillis) throws Exception {
            if (!Files.isRegularFile(python)) return new Result(false, false, python, "executable does not exist");
            Path checker = backend.resolve("scripts/check_runtime.py");
            if (!Files.isRegularFile(checker)) throw new IOException("Embedded runtime preflight is missing: " + checker);
            List<String> command = new ArrayList<String>(Arrays.asList(python.toString(), "-B", checker.toString(), "--json"));
            if (requireCuda) command.add("--require-cuda");
            Path log = scratch.resolve("probe-" + UUID.randomUUID() + ".json.log");
            ProcessBuilder process = new ProcessBuilder(command).directory(backend.toFile());
            int exit = runBounded(process, log, scratch, timeoutMillis, monitor);
            if (Files.size(log) > 1024 * 1024) return new Result(false, false, python, "oversized preflight output: " + log);
            String raw = new String(Files.readAllBytes(log), StandardCharsets.UTF_8);
            int begin = raw.indexOf('{'), end = raw.lastIndexOf('}');
            if (begin < 0 || end < begin) return new Result(false, false, python, "no preflight JSON; see " + log);
            JsonObject data = JSON.fromJson(raw.substring(begin, end + 1), JsonObject.class);
            boolean ok = exit == 0 && data.has("ok") && data.get("ok").getAsBoolean();
            boolean cuda = data.has(requireCuda ? "cuda_execution_tested" : "cuda_available") && data.get(requireCuda ? "cuda_execution_tested" : "cuda_available").getAsBoolean();
            Path reported = data.has("python") ? Paths.get(data.get("python").getAsString()).toAbsolutePath().normalize() : python;
            if (!Files.isRegularFile(reported)) ok = false;
            return new Result(ok, cuda, reported, data.has("errors") ? data.get("errors").toString() : "preflight failed; see " + log);
        }
    }

    private static int runBounded(ProcessBuilder builder, Path log, Path scratch, long timeoutMillis,
            RuntimeResolver.Monitor monitor) throws Exception {
        Path temporary = Files.createDirectories(scratch.resolve("tmp"));
        builder.environment().put("PYTHONDONTWRITEBYTECODE", "1");
        builder.environment().put("PYTHONUTF8", "1");
        builder.environment().put("PYTHONNOUSERSITE", "1");
        builder.environment().remove("PYTHONPATH");
        builder.environment().remove("PYTHONHOME");
        builder.environment().put("MPLCONFIGDIR", Files.createDirectories(scratch.resolve("matplotlib")).toString());
        builder.environment().put("XDG_CACHE_HOME", scratch.toString());
        for (String key : new String[] {"TEMP", "TMP", "TMPDIR"}) builder.environment().put(key, temporary.toString());
        builder.redirectErrorStream(true); builder.redirectOutput(log.toFile());
        monitor.checkCancelled();
        Process child = builder.start();
        long started = System.nanoTime();
        try {
            while (!child.waitFor(150, TimeUnit.MILLISECONDS)) {
                monitor.checkCancelled();
                if (elapsedMillis(started) >= timeoutMillis) throw new IOException("Python check timed out after " + timeoutMillis / 1000 + " s; see " + log);
            }
            monitor.checkCancelled();
            return child.exitValue();
        }
        finally {
            if (child.isAlive()) {
                child.destroyForcibly();
                boolean interrupted = Thread.interrupted();
                try { if (!child.waitFor(3, TimeUnit.SECONDS)) throw new IOException("Python preflight could not be terminated; see " + log); }
                finally { if (interrupted) Thread.currentThread().interrupt(); }
            }
        }
    }
    private static long elapsedMillis(long start) { return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start); }
}
