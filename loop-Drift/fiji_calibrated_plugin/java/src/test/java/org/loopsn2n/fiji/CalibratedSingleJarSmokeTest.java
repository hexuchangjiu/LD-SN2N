package org.loopsn2n.fiji;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/** Headless tests: mocked discovery policy is explicitly separate from real subprocess probes. */
public final class CalibratedSingleJarSmokeTest {
    private static int checks;
    private CalibratedSingleJarSmokeTest() { }
    public static void main(String[] args) throws Exception {
        if (args.length < 1 || args.length > 2) throw new IllegalArgumentException("usage: <fresh-E-test-root> [real-python]");
        Path root = Files.createDirectories(Paths.get(args[0]).toAbsolutePath().resolve("single JAR 中文 空格"));
        cacheTests(root);
        discoveryPolicyTests(root);
        if (args.length == 2) realProcessTests(root, Paths.get(args[1]).toAbsolutePath());
        System.out.println("SINGLE_JAR_SMOKE_OK checks=" + checks + " real_python=" + (args.length == 2));
    }

    private static void cacheTests(Path root) throws Exception {
        final Map<String, byte[]> files = new LinkedHashMap<String, byte[]>();
        files.put("fiji_calibrated_plugin/backend/bridge.py", "# backend\n".getBytes(StandardCharsets.UTF_8));
        files.put("scripts/check_runtime.py", "# checker\n".getBytes(StandardCharsets.UTF_8));
        files.put("requirements.txt", "torch>=2.6\n".getBytes(StandardCharsets.UTF_8));
        final EmbeddedRuntime.Source source = source(files, null);
        final Path cache = root.resolve("cache");
        Path first = EmbeddedRuntime.extract(cache, source, RuntimeResolver.SILENT);
        require(Files.isRegularFile(first.resolve("requirements.txt")), "Embedded extraction missing file");
        require(first.equals(EmbeddedRuntime.extract(cache, source, RuntimeResolver.SILENT)), "Cache reuse changed identity");
        Files.write(first.resolve("requirements.txt"), "tampered".getBytes(StandardCharsets.UTF_8));
        Path restored = EmbeddedRuntime.extract(cache, source, RuntimeResolver.SILENT);
        require(Arrays.equals(files.get("requirements.txt"), Files.readAllBytes(restored.resolve("requirements.txt"))), "Tampered cache trusted");
        Files.write(first.resolve("surprise.py"), new byte[] {1});
        EmbeddedRuntime.extract(cache, source, RuntimeResolver.SILENT);
        require(!Files.exists(first.resolve("surprise.py")), "Unexpected executable retained");
        Files.delete(first.resolve("requirements.txt"));
        EmbeddedRuntime.extract(cache, source, RuntimeResolver.SILENT);
        require(Files.isRegularFile(first.resolve("requirements.txt")), "Missing cache file not repaired");
        final AtomicReference<Throwable> failure = new AtomicReference<Throwable>();
        Runnable concurrent = new Runnable() {
            @Override public void run() { try { EmbeddedRuntime.extract(cache.resolve("parallel"), source, RuntimeResolver.SILENT); }
                catch (Throwable exc) { failure.compareAndSet(null, exc); } }
        };
        Thread left = new Thread(concurrent), right = new Thread(concurrent);
        left.start(); right.start(); left.join(10000); right.join(10000);
        require(!left.isAlive() && !right.isAlive() && failure.get() == null, "Concurrent extraction failed: " + failure.get());
        for (String invalid : new String[] {"../escape.py", "/absolute.py", "C:/absolute.py", "a\\b.py", "a//b.py", "a/../b.py", "CON", "a./x.py"}) {
            try { EmbeddedRuntime.validateName(invalid); throw new AssertionError("Unsafe manifest accepted: " + invalid); }
            catch (IOException expected) { checks++; }
        }
        JsonObject duplicate = manifest(files); duplicate.getAsJsonArray("files").add(duplicate.getAsJsonArray("files").get(0));
        expectExtractionFailure(cache, source(files, duplicate), "Duplicate manifest path accepted");
        JsonObject corrupt = manifest(files); corrupt.getAsJsonArray("files").get(0).getAsJsonObject().addProperty("sha256", String.join("", Collections.nCopies(64, "0")));
        expectExtractionFailure(root.resolve("bad-resource"), source(files, corrupt), "Corrupt JAR resource accepted");
        Path link = root.resolve("unsafe-link");
        try {
            Files.createSymbolicLink(link, cache);
            expectExtractionFailure(link, source, "Linked cache accepted");
        }
        catch (UnsupportedOperationException exc) { System.out.println("SYMLINK_TEST_UNAVAILABLE " + exc); }
        catch (java.nio.file.FileSystemException exc) { System.out.println("SYMLINK_TEST_UNAVAILABLE " + exc); }
        try {
            EmbeddedRuntime.extract(root.resolve("cancelled"), source, cancelled()); throw new AssertionError("Extraction cancellation ignored");
        }
        catch (CancellationException expected) { checks++; }
    }

    private static void discoveryPolicyTests(Path root) throws Exception {
        final Path bad = root.resolve("bad-python.exe"), cpu = root.resolve("Python 中文.exe"), gpu = root.resolve("GPU Python.exe");
        final AtomicInteger probes = new AtomicInteger();
        PythonDiscovery.Probe mock = new PythonDiscovery.Probe() {
            @Override public PythonDiscovery.Result check(Path python, boolean cuda, long timeout) {
                probes.incrementAndGet();
                boolean works = python.equals(gpu) || (python.equals(cpu) && !cuda);
                return new PythonDiscovery.Result(works, python.equals(gpu), python, works ? "mock accepted" : "mock missing dependencies");
            }
        };
        PythonDiscovery.Selection choice = PythonDiscovery.select(Arrays.asList(bad, cpu), null, null, "cpu", mock, RuntimeResolver.SILENT, 1000);
        require(choice.python.equals(cpu), "First invalid environment was not skipped");
        choice = PythonDiscovery.select(Arrays.asList(cpu, gpu), null, null, "auto", mock, RuntimeResolver.SILENT, 1000);
        require(choice.python.equals(gpu) && choice.device.equals("cuda"), "Auto did not prefer actual CUDA probe");
        choice = PythonDiscovery.select(Arrays.asList(bad, cpu), null, bad, "auto", mock, RuntimeResolver.SILENT, 1000);
        require(choice.python.equals(cpu), "Stale remembered Python was not skipped");
        choice = PythonDiscovery.select(Arrays.asList(cpu, gpu), null, cpu, "auto", mock, RuntimeResolver.SILENT, 1000);
        require(choice.python.equals(cpu), "Revalidated remembered Python policy changed");
        choice = PythonDiscovery.select(Arrays.asList(gpu), cpu, null, "auto", mock, RuntimeResolver.SILENT, 1000);
        require(choice.python.equals(cpu), "Explicit usable Python switched interpreters");
        try { PythonDiscovery.select(Arrays.asList(cpu), bad, null, "auto", mock, RuntimeResolver.SILENT, 1000); throw new AssertionError("Explicit invalid override fell back"); }
        catch (IOException expected) { require(expected.getMessage().contains("explicitly selected"), "Missing explicit error"); }
        try { PythonDiscovery.select(Arrays.asList(bad), null, null, "auto", mock, RuntimeResolver.SILENT, 1000); throw new AssertionError("All-invalid accepted"); }
        catch (IOException expected) { require(expected.getMessage().contains("never installs"), "No dependency setup guidance"); }
        try { PythonDiscovery.select(Arrays.asList(cpu), cpu, null, "cuda", mock, RuntimeResolver.SILENT, 1000); throw new AssertionError("CPU accepted for CUDA"); }
        catch (IOException expected) { checks++; }
        try { PythonDiscovery.select(Arrays.asList(cpu), null, null, "cpu", mock, cancelled(), 1000); throw new AssertionError("Discovery cancellation ignored"); }
        catch (CancellationException expected) { checks++; }
        require(probes.get() >= 12, "Mock policy tests did not probe environments");
    }

    private static void realProcessTests(Path root, Path python) throws Exception {
        System.setProperty("loopsn2n.cacheDir", root.resolve("real cache").toString());
        Path job = Files.createDirectories(root.resolve("real job"));
        RuntimeResolver.Settings selected = RuntimeResolver.resolve(python.toFile(), "cpu", job, RuntimeResolver.SILENT);
        require(selected.backend.startsWith(root.resolve("real cache")), "Did not use embedded JAR runtime");
        require(selected.device.equals("cpu"), "Explicit CPU did not stay CPU");
        RuntimeResolver.Settings remembered = RuntimeResolver.resolve(null, "auto", Files.createDirectories(root.resolve("remembered job")), RuntimeResolver.SILENT);
        require(remembered.python.equals(selected.python), "Remembered validated executable changed");
        Path bad = root.resolve("not an executable.exe"); Files.write(bad, new byte[] {0});
        PythonDiscovery.ProcessProbe real = new PythonDiscovery.ProcessProbe(selected.backend,
            Files.createDirectories(root.resolve("real sequential/.cache")), RuntimeResolver.SILENT);
        PythonDiscovery.Selection afterBad = PythonDiscovery.select(Arrays.asList(bad, selected.python), null, null, "cpu", real, RuntimeResolver.SILENT, 90000);
        require(afterBad.python.equals(selected.python), "Real invalid executable did not fall through");
        Path sleeper = Files.createDirectories(root.resolve("sleep backend/scripts"));
        Files.write(sleeper.resolve("check_runtime.py"), "import time\ntime.sleep(30)\n".getBytes(StandardCharsets.UTF_8));
        PythonDiscovery.ProcessProbe slow = new PythonDiscovery.ProcessProbe(sleeper.getParent(),
            Files.createDirectories(root.resolve("timeout/.cache")), RuntimeResolver.SILENT);
        long start = System.nanoTime();
        try { slow.check(python, false, 500); throw new AssertionError("Probe timeout ignored"); }
        catch (IOException expected) { require(java.util.concurrent.TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - start) < 5, "Probe termination unbounded"); }
        System.out.println("REAL_PYTHON_PROBES_OK python=" + selected.python + " backend=" + selected.backend);
    }

    private static RuntimeResolver.Monitor cancelled() {
        return new RuntimeResolver.Monitor() {
            @Override public void progress(String message) { }
            @Override public void checkCancelled() { throw new CancellationException("test cancel"); }
        };
    }
    private static EmbeddedRuntime.Source source(final Map<String, byte[]> files, JsonObject manifest) throws IOException {
        final byte[] bytes = new Gson().toJson(manifest == null ? manifest(files) : manifest).getBytes(StandardCharsets.UTF_8);
        return new EmbeddedRuntime.Source() {
            @Override public InputStream open(String name) {
                byte[] value = name.equals("manifest.json") ? bytes : files.get(name);
                return value == null ? null : new ByteArrayInputStream(value);
            }
        };
    }
    private static JsonObject manifest(Map<String, byte[]> files) throws IOException {
        JsonObject result = new JsonObject(); result.addProperty("version", "0.5.0"); JsonArray list = new JsonArray();
        for (Map.Entry<String, byte[]> file : files.entrySet()) {
            JsonObject value = new JsonObject(); value.addProperty("path", file.getKey());
            value.addProperty("bytes", file.getValue().length); value.addProperty("sha256", EmbeddedRuntime.sha256(file.getValue())); list.add(value);
        }
        result.add("files", list); return result;
    }
    private static void expectExtractionFailure(Path cache, EmbeddedRuntime.Source source, String message) throws Exception {
        try { EmbeddedRuntime.extract(cache, source, RuntimeResolver.SILENT); throw new AssertionError(message); }
        catch (IOException expected) { checks++; }
    }
    private static void require(boolean condition, String message) { if (!condition) throw new AssertionError(message); checks++; }
}
