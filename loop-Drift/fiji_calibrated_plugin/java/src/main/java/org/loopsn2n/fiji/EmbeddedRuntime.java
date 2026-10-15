package org.loopsn2n.fiji;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;

/** Hash-bound, locked extraction of the backend carried by this exact JAR. */
final class EmbeddedRuntime {
    static final String PREFIX = "loopsn2n-runtime/";
    interface Source { InputStream open(String name) throws IOException; }
    static final Source CLASSPATH = new Source() {
        @Override public InputStream open(String name) { return EmbeddedRuntime.class.getClassLoader().getResourceAsStream(PREFIX + name); }
    };
    private static final LinkOption[] NO_LINKS = {LinkOption.NOFOLLOW_LINKS};
    private EmbeddedRuntime() { }

    static Path extract(Path cache, Source source, RuntimeResolver.Monitor monitor) throws Exception {
        byte[] manifest;
        try (InputStream stream = source.open("manifest.json")) {
            if (stream == null) throw new IOException("This JAR has no embedded backend. Install the complete single-JAR 0.5.0 release.");
            manifest = readBounded(stream, 1024 * 1024);
        }
        JsonObject specification;
        try { specification = new Gson().fromJson(new String(manifest, StandardCharsets.UTF_8), JsonObject.class); }
        catch (RuntimeException exc) { throw new IOException("Invalid embedded runtime manifest", exc); }
        if (specification == null || !specification.has("version") || !specification.has("files"))
            throw new IOException("Incomplete embedded runtime manifest");
        String version = specification.get("version").getAsString();
        if (!version.equals(LoopSN2NCalibratedCommand.VERSION)) throw new IOException("Backend/JAR version mismatch: " + version);
        final Map<String, JsonObject> entries = new LinkedHashMap<String, JsonObject>();
        long total = 0;
        try {
            for (JsonElement item : specification.getAsJsonArray("files")) {
                JsonObject file = item.getAsJsonObject();
                String name = file.get("path").getAsString();
                validateName(name);
                long length = file.get("bytes").getAsLong();
                if (length < 0 || length > 16 * 1024 * 1024 || !file.get("sha256").getAsString().matches("[0-9a-f]{64}"))
                    throw new IOException("Invalid embedded file metadata: " + name);
                if (entries.put(name, file) != null) throw new IOException("Duplicate embedded file: " + name);
                total += length;
            }
        }
        catch (RuntimeException exc) { throw new IOException("Invalid embedded file list", exc); }
        if (entries.size() > 2048 || total > 64 * 1024 * 1024 ||
            !entries.containsKey("fiji_calibrated_plugin/backend/bridge.py") || !entries.containsKey("scripts/check_runtime.py"))
            throw new IOException("Embedded backend is incomplete or exceeds safe extraction limits");
        cache = cache.toAbsolutePath().normalize();
        safeDirectories(cache);
        String key = version + "-" + sha256(manifest);
        Path destination = cache.resolve(key);
        Path lockPath = cache.resolve(key + ".lock");
        assertSafePath(lockPath);
        try (FileChannel channel = FileChannel.open(lockPath, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
            FileLock lock = null;
            long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(60);
            while (lock == null) {
                monitor.checkCancelled();
                try { lock = channel.tryLock(); }
                catch (OverlappingFileLockException busy) { /* Another Fiji thread is extracting. */ }
                if (lock == null) {
                    if (System.nanoTime() >= deadline) throw new IOException("Timed out waiting for the embedded runtime cache lock.");
                    monitor.progress("Waiting for another LoopSN2N runtime preparation...");
                    Thread.sleep(150);
                }
            }
            try {
                assertSafePath(destination);
                if (Files.exists(destination, NO_LINKS)) {
                    // Unsafe links are never moved or followed. Ordinary corruption is retained for inspection.
                    assertSafeTree(destination);
                    if (verify(destination, entries, monitor)) return destination;
                    Path quarantine = cache.resolve(key + ".quarantine-" + UUID.randomUUID());
                    monitor.progress("Runtime cache changed; preserving it and restoring the verified JAR copy...");
                    Files.move(destination, quarantine, StandardCopyOption.ATOMIC_MOVE);
                }
                monitor.progress("Preparing the backend embedded in this JAR...");
                Path staging = Files.createTempDirectory(cache, key + ".partial-");
                // Failed staging directories are deliberately retained; no broad recursive cleanup.
                for (Map.Entry<String, JsonObject> item : entries.entrySet()) {
                    monitor.checkCancelled();
                    Path target = staging.resolve(item.getKey()).normalize();
                    if (!target.startsWith(staging)) throw new IOException("Unsafe runtime extraction target");
                    safeDirectories(target.getParent());
                    byte[] contents;
                    try (InputStream stream = source.open(item.getKey())) {
                        if (stream == null) throw new IOException("Missing embedded runtime resource: " + item.getKey());
                        contents = readBounded(stream, 16 * 1024 * 1024);
                    }
                    JsonObject expected = item.getValue();
                    if (contents.length != expected.get("bytes").getAsLong() || !sha256(contents).equals(expected.get("sha256").getAsString()))
                        throw new IOException("Embedded runtime hash mismatch: " + item.getKey());
                    Files.write(target, contents, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
                }
                if (!verify(staging, entries, monitor)) throw new IOException("Extracted runtime failed integrity verification");
                // Staging and destination share a filesystem. Never publish a partial tree.
                Files.move(staging, destination, StandardCopyOption.ATOMIC_MOVE);
                return destination;
            }
            finally { lock.release(); }
        }
    }

    static void validateName(String name) throws IOException {
        if (name == null || name.isEmpty() || name.startsWith("/") || name.contains("\\") || name.contains(":") || name.endsWith("/"))
            throw new IOException("Unsafe embedded path: " + name);
        for (String part : name.split("/", -1)) {
            if (part.isEmpty() || part.equals(".") || part.equals("..") || part.endsWith(".") || part.endsWith(" ") ||
                part.matches("(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\\..*)?") || !part.matches("[A-Za-z0-9_. -]+"))
                throw new IOException("Unsafe embedded path: " + name);
        }
    }

    private static boolean verify(final Path directory, final Map<String, JsonObject> entries,
            final RuntimeResolver.Monitor monitor) throws IOException {
        if (!Files.isDirectory(directory, NO_LINKS)) return false;
        final boolean[] valid = {true};
        Files.walkFileTree(directory, new SimpleFileVisitor<Path>() {
            @Override public FileVisitResult preVisitDirectory(Path path, BasicFileAttributes attrs) throws IOException {
                monitor.checkCancelled(); assertSafePath(path); return FileVisitResult.CONTINUE;
            }
            @Override public FileVisitResult visitFile(Path path, BasicFileAttributes attrs) throws IOException {
                monitor.checkCancelled(); assertSafePath(path);
                String name = directory.relativize(path).toString().replace('\\', '/');
                JsonObject expected = entries.get(name);
                if (!attrs.isRegularFile() || expected == null || attrs.size() != expected.get("bytes").getAsLong() ||
                    !sha256(Files.readAllBytes(path)).equals(expected.get("sha256").getAsString())) valid[0] = false;
                return FileVisitResult.CONTINUE;
            }
        });
        for (String name : entries.keySet()) if (!Files.isRegularFile(directory.resolve(name), NO_LINKS)) valid[0] = false;
        return valid[0];
    }

    private static void assertSafeTree(Path directory) throws IOException {
        Files.walkFileTree(directory, new SimpleFileVisitor<Path>() {
            @Override public FileVisitResult preVisitDirectory(Path path, BasicFileAttributes attrs) throws IOException {
                assertSafePath(path); return FileVisitResult.CONTINUE;
            }
            @Override public FileVisitResult visitFile(Path path, BasicFileAttributes attrs) throws IOException {
                assertSafePath(path); return FileVisitResult.CONTINUE;
            }
        });
    }

    static void safeDirectories(Path path) throws IOException {
        assertSafePath(path); Files.createDirectories(path); assertSafePath(path);
    }

    static void assertSafePath(Path path) throws IOException {
        for (Path current = path.toAbsolutePath().normalize(); current != null; current = current.getParent()) {
            if (!Files.exists(current, NO_LINKS)) continue;
            BasicFileAttributes attrs = Files.readAttributes(current, BasicFileAttributes.class, NO_LINKS);
            if (attrs.isSymbolicLink() || attrs.isOther()) throw new IOException("Refusing linked/reparse runtime cache path: " + current);
        }
    }

    static byte[] readBounded(InputStream input, int limit) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192]; int count;
        while ((count = input.read(buffer)) != -1) {
            if (output.size() + count > limit) throw new IOException("Runtime resource exceeds safe size limit");
            output.write(buffer, 0, count);
        }
        return output.toByteArray();
    }

    static String sha256(byte[] bytes) throws IOException {
        try {
            StringBuilder result = new StringBuilder();
            for (byte b : MessageDigest.getInstance("SHA-256").digest(bytes)) result.append(String.format("%02x", b & 255));
            return result.toString();
        }
        catch (java.security.NoSuchAlgorithmException impossible) { throw new IOException(impossible); }
    }
}
