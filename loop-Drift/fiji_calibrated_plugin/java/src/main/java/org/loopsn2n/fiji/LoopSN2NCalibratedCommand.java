package org.loopsn2n.fiji;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import ij.measure.Calibration;
import ij.measure.ResultsTable;
import java.awt.BorderLayout;
import java.awt.Desktop;
import java.awt.Dimension;
import java.awt.Window;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;
import org.scijava.command.Command;
import org.scijava.command.ContextCommand;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.widget.FileWidget;

/** Parallel research entry point. It does not load or modify the strict plugin. */
@Plugin(type = Command.class,
    menuPath = "Plugins>LoopSN2N>Calibrate noise and assess (research)")
public final class LoopSN2NCalibratedCommand extends ContextCommand {
    public static final String VERSION = "0.5.0";
    public static final int MIN_STATIC_REPEAT_FRAMES = 34;
    private static final Gson JSON = new GsonBuilder().setPrettyPrinting().create();
    private static final AtomicBoolean RUNNING = new AtomicBoolean(false);

    @Parameter(label = "Opened stack: >=34 static repeats; selected full plane only (uint16 / float32)")
    private ImagePlus inputImage;

    @Parameter(label = "I confirm all frames are static repeats, not a Z-stack or moving time series", persist = false)
    private boolean staticRepeats = false;

    @Parameter(label = "Supported SN2N AUnet checkpoint (.pt / .pth)", style = FileWidget.OPEN_STYLE)
    private File checkpointFile;

    @Parameter(label = "Allow trusted legacy checkpoint compatibility (off = safe weights only)", persist = false)
    private boolean trustCheckpoint = false;

    @Parameter(label = "Output root directory", style = FileWidget.DIRECTORY_STYLE)
    private File outputRoot = new File(System.getProperty("user.home"), "LoopSN2N_outputs");

    @Parameter(label = "Compute device", choices = { "auto", "cuda", "cpu" })
    private String device = "auto";

    @Parameter(label = "Developer override: backend directory (blank = backend inside this JAR)", required = false, persist = false, style = FileWidget.DIRECTORY_STYLE)
    private File projectDirectory;

    @Parameter(label = "Advanced override: Python executable (blank = automatically find a compatible environment)", required = false, persist = false, style = FileWidget.OPEN_STYLE)
    private File pythonExecutable;

    @Override
    public void run() {
        if (!RUNNING.compareAndSet(false, true)) {
            IJ.error("LoopSN2N research", "A research assessment is still running or stopping. Wait for it to exit before starting another.");
            return;
        }
        final ResearchWorker worker = new ResearchWorker();
        boolean interruptedDispatch = false;
        try {
            // Capture the selected plane before a background task can observe later UI navigation.
            final int queryIndex = inputImage == null ? -1 : inputImage.getCurrentSlice() - 1;
            worker.queryIndex = queryIndex;
            final ProgressDialog[] dialog = new ProgressDialog[1];
            onEdt(new Runnable() {
                @Override public void run() {
                    dialog[0] = new ProgressDialog(worker);
                    worker.dialog = dialog[0];
                    worker.submitted.set(true);
                    worker.execute();
                    dialog[0].setVisible(true);
                }
            });
            JsonObject response = worker.get();
            presentResult(response, worker.jobDirectory, worker.displayCalibration);
        }
        catch (InterruptedException exc) {
            interruptedDispatch = true;
            worker.requestStop();
            // Never release the single-worker guard while a worker is still alive.
            Thread.currentThread().interrupt();
            IJ.error("LoopSN2N research", "The command was interrupted. A stop was requested; wait for worker termination before retrying.");
        }
        catch (ExecutionException exc) {
            IJ.error("LoopSN2N research", "Assessment failed: " + exc.getCause());
        }
        catch (Exception exc) {
            worker.requestStop();
            IJ.error("LoopSN2N research", "Assessment could not start: " + exc.getMessage());
        }
        finally {
            // An interrupted invokeAndWait may still have an EDT launch queued.
            if (worker.isDone() || (!interruptedDispatch && !worker.submitted.get())) RUNNING.set(false);
        }
    }

    /** Validate axes without guessing that arbitrary Z/T data are stationary repeats. */
    public static void validateRepeatStack(ImagePlus image, boolean staticRepeats) {
        if (image == null) throw new IllegalArgumentException("Open a repeat stack first.");
        if (!staticRepeats) throw new IllegalArgumentException("Confirm static repeats explicitly. A moving series or Z-stack is outside this research entry point.");
        if (image.getNChannels() != 1) throw new IllegalArgumentException("Only one channel is supported; split channels explicitly before assessment.");
        if (image.getNSlices() > 1 && image.getNFrames() > 1)
            throw new IllegalArgumentException("Mixed Z/T stacks are not supported.");
        if ((image.isHyperStack() || image.getOpenAsHyperStack()) && image.getNSlices() > 1)
            throw new IllegalArgumentException("A declared Z hyperstack is not a static-repeat stack. Do not relabel genuine Z data as repeats.");
        if (image.getStackSize() < MIN_STATIC_REPEAT_FRAMES)
            throw new IllegalArgumentException("Noise calibration requires at least 34 static repeat frames: the query pair is excluded, leaving 8 fit and 8 held-out pairs. Only the selected full plane is mapped.");
        if (image.getType() != ImagePlus.GRAY16 && image.getType() != ImagePlus.GRAY32)
            throw new IllegalArgumentException("Use an unmodified uint16 or float32 grayscale stack. RGB, indexed and 8-bit conversions are not accepted.");
    }

    /** Public headless integration entry point: exports full planes, preserving raw pixels. */
    public static Path exportAndWriteRequest(ImagePlus image, Path jobDirectory, Path checkpoint,
            int queryIndex, String device, boolean staticRepeats, boolean trustCheckpoint) throws IOException {
        return exportAndWriteRequest(image, jobDirectory, checkpoint, queryIndex, device,
            staticRepeats, trustCheckpoint, new StopCheck() {
                @Override public void check() {
                    if (Files.exists(jobDirectory.resolve("STOP_REQUESTED.json")))
                        throw new CancellationException("Stop requested before backend launch.");
                }
            });
    }

    private interface StopCheck { void check(); }

    private static Path exportAndWriteRequest(ImagePlus image, Path jobDirectory, Path checkpoint,
            int queryIndex, String device, boolean staticRepeats, boolean trustCheckpoint,
            StopCheck stop) throws IOException {
        validateRepeatStack(image, staticRepeats);
        validateRequestSettings(checkpoint, device, trustCheckpoint);
        if (queryIndex < 0 || queryIndex >= image.getStackSize())
            throw new IllegalArgumentException("Query index is outside the source stack.");
        Files.createDirectories(jobDirectory);
        rejectExistingJobFiles(jobDirectory);
        stop.check();
        Path stackPath = jobDirectory.resolve("source_stack.tif");
        // Duplicate raw processors directly: ImagePlus.duplicate() can crop to an active ROI.
        // Locking the source only during the copy prevents UI edits without changing pixels/ROI/plane.
        if (!image.lockSilently()) throw new IOException("The source image is busy. Finish its current operation first.");
        ImagePlus snapshot;
        try {
            ImageStack copied = new ImageStack(image.getWidth(), image.getHeight());
            for (int slice = 1; slice <= image.getStackSize(); slice++) {
                stop.check();
                copied.addSlice(image.getStack().getSliceLabel(slice), image.getStack().getProcessor(slice).duplicate());
            }
            snapshot = new ImagePlus("LoopSN2N static repeats", copied);
            snapshot.setCalibration(image.getCalibration().copy());
            // The explicit static-repeat declaration authorizes temporal repeat export,
            // never relabeling an input that validateRepeatStack identified as genuine Z.
            snapshot.setDimensions(1, 1, image.getStackSize());
            snapshot.setProperty("Info", "LoopSN2N static_repeats; full-field raw pixels; source channels=1; query_index=" + queryIndex);
        }
        finally { image.unlock(); }
        stop.check();
        if (!new FileSaver(snapshot).saveAsTiffStack(stackPath.toString()))
            throw new IOException("Fiji could not export the full-precision repeat stack.");
        stop.check(); // Never launch Python or write a request for an interrupted export.
        JsonObject request = prepareRequest(stackPath, checkpoint, jobDirectory, queryIndex, device,
            trustCheckpoint, snapshot.getCalibration());
        Path requestPath = jobDirectory.resolve("request.json");
        writeNewJson(requestPath, request);
        return requestPath;
    }

    /** Build the explicit bridge contract; no per-frame scaling, ROI crop, or dtype conversion. */
    public static JsonObject prepareRequest(Path stackPath, Path checkpoint, Path jobDirectory,
            int queryIndex, String device, boolean trustCheckpoint, Calibration calibration) {
        JsonObject request = new JsonObject();
        request.addProperty("schema_version", 1);
        request.addProperty("stack_path", stackPath.toAbsolutePath().toString());
        request.addProperty("checkpoint_path", checkpoint.toAbsolutePath().toString());
        request.addProperty("output_dir", jobDirectory.toAbsolutePath().toString());
        request.addProperty("query_index", queryIndex);
        request.addProperty("domain", "static_repeats");
        request.addProperty("device", device);
        request.addProperty("trust_checkpoint", trustCheckpoint);
        request.addProperty("sn2n_profile", "native_fullframe");
        if (calibration != null) {
            JsonObject physical = new JsonObject();
            physical.addProperty("pixel_width", calibration.pixelWidth);
            physical.addProperty("pixel_height", calibration.pixelHeight);
            physical.addProperty("pixel_depth", calibration.pixelDepth);
            physical.addProperty("unit", calibration.getUnit());
            physical.addProperty("x_unit", calibration.getXUnit());
            physical.addProperty("y_unit", calibration.getYUnit());
            physical.addProperty("z_unit", calibration.getZUnit());
            physical.addProperty("frame_interval", calibration.frameInterval);
            physical.addProperty("time_unit", calibration.getTimeUnit());
            physical.addProperty("x_origin", calibration.xOrigin);
            physical.addProperty("y_origin", calibration.yOrigin);
            physical.addProperty("z_origin", calibration.zOrigin);
            request.add("pixel_calibration", physical);
        }
        return request;
    }

    private static void validateRequestSettings(Path checkpoint, String device, boolean trust) {
        if (checkpoint == null || !Files.isRegularFile(checkpoint))
            throw new IllegalArgumentException("Choose a readable local checkpoint file.");
        // False deliberately reaches the backend's safe weights-only loader.
        // Legacy compatibility is an explicit, non-persisted opt-in, never a requirement.
        if (!Arrays.asList("auto", "cuda", "cpu").contains(device))
            throw new IllegalArgumentException("Device must be auto, cuda, or cpu.");
    }

    private static void rejectExistingJobFiles(Path job) throws IOException {
        for (String name : new String[] { "source_stack.tif", "request.json", "response.json", "manifest.json", "manifest_json" }) {
            if (Files.exists(job.resolve(name))) throw new IOException("Refusing to overwrite an existing job artifact: " + job.resolve(name));
        }
    }

    private static Path createFreshJob(Path root) throws IOException {
        Files.createDirectories(root);
        String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss_SSS").format(new Date());
        return Files.createDirectory(root.resolve("calibrated_" + stamp + "_" + UUID.randomUUID().toString().substring(0, 8)));
    }

    private static void writeNewJson(Path path, JsonObject value) throws IOException {
        try (Writer writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) { JSON.toJson(value, writer); }
    }

    public static JsonObject readResponse(Path path) throws IOException {
        try (Reader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            JsonObject response = JSON.fromJson(reader, JsonObject.class);
            if (response == null) throw new IOException("Empty backend response.");
            return response;
        }
    }

    /** Same process-launch and terminal-response path used by the GUI, without opening windows. */
    public static JsonObject runBackendAndValidate(Path request, Path python, Path project) throws Exception {
        return runBackendAndValidate(request, python, project, new BackendObserver() {
            @Override public void update(long start) { }
        });
    }

    private interface BackendObserver { void update(long start); }

    private static JsonObject runBackendAndValidate(Path request, Path python, Path project,
            BackendObserver observer) throws Exception {
        JsonObject payload = readResponse(request);
        Path job = Paths.get(payload.get("output_dir").getAsString());
        Path response = job.resolve("response.json");
        if (Files.exists(response) || Files.exists(job.resolve("manifest.json")))
            throw new IOException("Refusing to launch over an existing response or manifest.");
        Path bridge = project.resolve("fiji_calibrated_plugin/backend/bridge.py");
        if (!Files.isRegularFile(python) || !Files.isRegularFile(bridge))
            throw new IOException("Python executable or calibrated backend is missing.");
        ProcessBuilder builder = new ProcessBuilder(python.toAbsolutePath().toString(), bridge.toAbsolutePath().toString(),
            "--request", request.toAbsolutePath().toString(), "--response", response.toAbsolutePath().toString());
        builder.directory(project.toFile());
        configurePythonEnvironment(builder, job);
        builder.redirectErrorStream(true);
        builder.redirectOutput(job.resolve("worker.log").toFile());
        Process child = builder.start();
        long start = System.nanoTime();
        try {
            while (!child.waitFor(500, TimeUnit.MILLISECONDS)) observer.update(start);
            if (!Files.isRegularFile(response))
                throw new IOException("Backend exited with code " + child.exitValue() + " without response.json. See worker.log.");
            JsonObject terminal = readResponse(response);
            String status = text(terminal, "status");
            if (!Arrays.asList("COMPLETE", "CANCELLED", "FAILED").contains(status))
                throw new IOException("Unexpected backend terminal status: " + status);
            if (child.exitValue() != 0 && "COMPLETE".equals(status))
                throw new IOException("Backend reported COMPLETE but exited with code " + child.exitValue());
            return terminal;
        }
        finally {
            // A method return must mean the child really exited, including interruption/error paths.
            boolean interrupted = Thread.interrupted();
            if (child.isAlive()) {
                Path marker = job.resolve("STOP_REQUESTED.json");
                if (!Files.exists(marker)) {
                    JsonObject stop = new JsonObject();
                    stop.addProperty("stop_requested", true);
                    stop.addProperty("source", "fiji_process_cleanup");
                    try { writeNewJson(marker, stop); }
                    catch (IOException exc) { IJ.log("Stop marker unavailable; waiting for backend exit: " + exc.getMessage()); }
                }
                while (child.isAlive()) {
                    try { child.waitFor(500, TimeUnit.MILLISECONDS); }
                    catch (InterruptedException exc) { interrupted = true; }
                }
            }
            if (interrupted) Thread.currentThread().interrupt();
        }
    }

    private static String text(JsonObject object, String key) {
        return object != null && object.has(key) && !object.get(key).isJsonNull() ? object.get(key).getAsString() : "not reported";
    }

    /** Keep Python caches and scratch files in the chosen output, not the Fiji installation. */
    private static void configurePythonEnvironment(ProcessBuilder builder, Path job) throws IOException {
        Path cache = Files.createDirectories(job.resolve(".cache"));
        Path temporary = Files.createDirectories(cache.resolve("tmp"));
        builder.environment().put("PYTHONDONTWRITEBYTECODE", "1");
        builder.environment().put("PYTHONUTF8", "1");
        builder.environment().put("PYTHONNOUSERSITE", "1");
        builder.environment().remove("PYTHONPATH");
        builder.environment().remove("PYTHONHOME");
        builder.environment().put("MPLCONFIGDIR", Files.createDirectories(cache.resolve("matplotlib")).toString());
        builder.environment().put("XDG_CACHE_HOME", cache.toString());
        for (String key : new String[] {"TEMP", "TMP", "TMPDIR"})
            builder.environment().put(key, temporary.toString());
    }

    private static JsonObject object(JsonObject parent, String key) {
        return parent != null && parent.has(key) && parent.get(key).isJsonObject() ? parent.getAsJsonObject(key) : new JsonObject();
    }

    private static JsonObject frontendTerminal(String status, String code, String message) {
        JsonObject result = new JsonObject();
        result.addProperty("status", status);
        result.addProperty("execution_status", status);
        result.addProperty("calibration_status", "NOT_RUN");
        JsonObject decision = new JsonObject();
        decision.addProperty("label", "NOT_ASSESSED");
        decision.addProperty("reason", message);
        decision.add("diagnostic_checks", new JsonObject());
        result.add("decision", decision);
        result.add("warnings", new JsonArray());
        result.add("artifacts", new JsonObject());
        if (!"CANCELLED".equals(status)) {
            JsonObject error = new JsonObject();
            error.addProperty("code", code);
            error.addProperty("message", message);
            result.add("error", error);
        }
        return result;
    }

    private final class ResearchWorker extends SwingWorker<JsonObject, String> {
        private final AtomicBoolean stopRequested = new AtomicBoolean(false);
        private final AtomicBoolean submitted = new AtomicBoolean(false);
        private volatile Path jobDirectory;
        private ProgressDialog dialog;
        private int queryIndex;
        private Calibration displayCalibration;

        void requestStop() {
            stopRequested.set(true);
            try { writeStopMarker(); }
            catch (IOException exc) { IJ.log("LoopSN2N could not write stop marker: " + exc.getMessage()); }
            publish("Stop requested. Waiting for the current safe point and worker exit; do not start another job.");
        }

        private void writeStopMarker() throws IOException {
            if (jobDirectory == null) return;
            Path marker = jobDirectory.resolve("STOP_REQUESTED.json");
            if (!Files.exists(marker)) {
                JsonObject request = new JsonObject();
                request.addProperty("stop_requested", true);
                request.addProperty("source", "fiji_frontend");
                request.addProperty("requested_at_epoch_ms", System.currentTimeMillis());
                try { writeNewJson(marker, request); }
                catch (java.nio.file.FileAlreadyExistsException ignored) { /* Idempotent stop. */ }
            }
        }

        private void checkStop() {
            if (stopRequested.get() || (jobDirectory != null && Files.exists(jobDirectory.resolve("STOP_REQUESTED.json"))))
                throw new CancellationException("Cancelled before backend launch; any partial export is not an assessment result.");
        }

        @Override protected JsonObject doInBackground() {
            JsonObject terminal;
            try {
                publish("Preparing a new research job; checking repeat-domain declaration and image axes...");
                jobDirectory = createFreshJob(outputRoot.toPath());
                if (stopRequested.get()) writeStopMarker();
                checkStop();
                validateRepeatStack(inputImage, staticRepeats);
                validateRequestSettings(checkpointFile == null ? null : checkpointFile.toPath(), device, trustCheckpoint);
                RuntimeResolver.Settings runtime = RuntimeResolver.resolve(projectDirectory, pythonExecutable, device, jobDirectory,
                    new RuntimeResolver.Monitor() {
                        @Override public void progress(String message) { publish(message); }
                        @Override public void checkCancelled() { checkStop(); }
                    });
                projectDirectory = runtime.backend.toFile();
                pythonExecutable = runtime.python.toFile();
                publish("Using verified backend " + projectDirectory + " and Python " + pythonExecutable + " (" + runtime.device + ")");
                Path bridge = projectDirectory.toPath().resolve("fiji_calibrated_plugin/backend/bridge.py");
                if (!Files.isRegularFile(bridge)) throw new IOException("Research backend not found: " + bridge);
                if (pythonExecutable == null || !pythonExecutable.isFile()) throw new IOException("Python executable is not available.");
                checkStop();
                publish("Copying and exporting every full-precision repeat frame; the active ROI is ignored...");
                displayCalibration = inputImage.getCalibration().copy();
                Path request = exportAndWriteRequest(inputImage, jobDirectory, checkpointFile.toPath(),
                    queryIndex, runtime.device, staticRepeats, trustCheckpoint, new StopCheck() {
                        @Override public void check() { checkStop(); }
                    });
                checkStop();
                publish("Calibrating noise and assessing the current full plane. See worker.log for detail...");
                terminal = runBackendAndValidate(request, pythonExecutable.toPath(), projectDirectory.toPath(), new BackendObserver() {
                    @Override public void update(long start) {
                        if (stopRequested.get()) {
                            try { writeStopMarker(); }
                            catch (IOException exc) { IJ.log("LoopSN2N could not write stop marker: " + exc.getMessage()); }
                            publish("Stopping safely; waiting for Python exit (" + elapsedSeconds(start) + " s elapsed)...");
                        }
                        else publish(backendProgress(start));
                    }
                });
                return terminal;
            }
            catch (CancellationException exc) { terminal = frontendTerminal("CANCELLED", "CANCELLED", exc.getMessage()); }
            catch (Exception exc) { terminal = frontendTerminal("FAILED", "FRONTEND_OR_WORKER_ERROR", exc.getClass().getSimpleName() + ": " + exc.getMessage()); }
            // Do not replace backend evidence, even if it was invalid or incomplete.
            if (jobDirectory != null) {
                try {
                    Path response = jobDirectory.resolve("response.json");
                    writeNewJson(Files.exists(response) ? jobDirectory.resolve("frontend_status.json") : response, terminal);
                }
                catch (IOException exc) { IJ.log("LoopSN2N could not record frontend terminal state: " + exc.getMessage()); }
            }
            return terminal;
        }

        private String backendProgress(long start) {
            Path progress = jobDirectory.resolve("progress.json");
            if (Files.isRegularFile(progress)) {
                try {
                    JsonObject value = readResponse(progress);
                    String counts = value.has("done") && value.has("total")
                        ? " (" + value.get("done").getAsInt() + "/" + value.get("total").getAsInt() + ")" : "";
                    return "Backend: " + text(value, "stage") + counts + "; " + elapsedSeconds(start)
                        + " s elapsed. Cancel requests a safe stop.";
                }
                catch (Exception ignored) { /* Backend may be atomically replacing the progress file. */ }
            }
            return "Backend running: calibration / diagnostic / output (" + elapsedSeconds(start)
                + " s elapsed). Cancel requests a safe stop.";
        }

        @Override protected void process(List<String> messages) {
            if (dialog != null && !messages.isEmpty()) dialog.update(messages.get(messages.size() - 1));
        }

        @Override protected void done() {
            RUNNING.set(false);
            if (dialog != null) dialog.dispose();
        }
    }

    private static long elapsedSeconds(long start) { return TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - start); }

    private static final class ProgressDialog extends JDialog {
        private final JTextArea message = new JTextArea(4, 62);
        ProgressDialog(final ResearchWorker worker) {
            super((Window) null, "LoopSN2N research: preparing / calibrating / assessing", ModalityType.APPLICATION_MODAL);
            setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
            message.setEditable(false);
            message.setLineWrap(true);
            message.setWrapStyleWord(true);
            message.setText("Preparing assessment...");
            JProgressBar progress = new JProgressBar();
            progress.setIndeterminate(true);
            JButton cancel = new JButton("Request safe stop");
            cancel.addActionListener(event -> {
                cancel.setEnabled(false);
                cancel.setText("Waiting for worker exit...");
                worker.requestStop();
            });
            JPanel bottom = new JPanel(new BorderLayout(8, 8));
            bottom.add(progress, BorderLayout.CENTER);
            bottom.add(cancel, BorderLayout.EAST);
            add(new JScrollPane(message), BorderLayout.CENTER);
            add(bottom, BorderLayout.SOUTH);
            setPreferredSize(new Dimension(730, 185));
            pack();
            setLocationRelativeTo(null);
        }
        void update(String value) { message.setText(value); }
    }

    private static void presentResult(final JsonObject response, final Path job, final Calibration sourceCalibration) throws Exception {
        onEdt(new Runnable() {
            @Override public void run() {
                JsonObject decision = object(response, "decision");
                JsonObject artifacts = object(response, "artifacts");
                ResultsTable table = new ResultsTable();
                table.incrementCounter();
                table.addValue("Execution", text(response, "execution_status"));
                table.addValue("Calibration", text(response, "calibration_status"));
                table.addValue("Research decision", text(decision, "label"));
                table.addValue("Reason", text(decision, "reason"));
                table.addValue("Scope", "Static repeats; one selected full 2D plane; native_fullframe");
                table.addValue("Output", job == null ? "not created" : job.toString());
                table.show("LoopSN2N calibrated research result");
                if ("COMPLETE".equals(text(response, "status"))) {
                    openImage(artifacts, "d1_tiff", "D1 - full-precision first-pass output", sourceCalibration);
                    openImage(artifacts, "map_rgb_tiff", "Structure-aware loop Map - blue attenuation / orange enhancement", sourceCalibration);
                    openImage(artifacts, "overview_png", "LoopSN2N overview and fixed legend", null);
                }
                // Reports explain calibration failures as well as successful assessments.
                openFileIfPresent(artifacts, "report_html");
                StringBuilder summary = new StringBuilder();
                summary.append("EXECUTION: ").append(text(response, "execution_status"));
                summary.append("\nCALIBRATION: ").append(text(response, "calibration_status"));
                summary.append("\nRESEARCH DECISION: ").append(text(decision, "label"));
                summary.append("\n\n").append(text(decision, "reason"));
                summary.append("\n\nWarnings:\n");
                JsonElement warnings = response.get("warnings");
                if (warnings != null && warnings.isJsonArray() && warnings.getAsJsonArray().size() > 0) {
                    for (JsonElement warning : warnings.getAsJsonArray()) summary.append("- ").append(warning.isJsonPrimitive() ? warning.getAsString() : JSON.toJson(warning)).append('\n');
                }
                else summary.append("None reported.\n");
                if (response.has("error")) summary.append("\nError:\n").append(JSON.toJson(response.get("error"))).append('\n');
                if (decision.has("diagnostic_checks")) summary.append("\nDiagnostic checks:\n").append(JSON.toJson(decision.get("diagnostic_checks"))).append('\n');
                summary.append("\nScope: one selected full 2D plane, not all repeats.\nMap is a loop-response diagnostic, not a pixel-error probability.\n");
                summary.append("New-domain formal decisions remain DEFER; calibration and successful execution do not certify restoration accuracy.\n");
                summary.append("Map display retains D1 in background; structural support has 1-pixel protection. Blue/orange and scalar Map values are unchanged.\n");
                summary.append("No intensity conversion or contrast operation was applied to the opened quantitative D1 pixels.\n");
                summary.append("\nOutput: ").append(job == null ? "not created" : job);
                JDialog details = new JDialog((Window) null, "LoopSN2N research result", java.awt.Dialog.ModalityType.MODELESS);
                JTextArea contents = new JTextArea(summary.toString(), 24, 86);
                contents.setEditable(false);
                contents.setLineWrap(true);
                contents.setWrapStyleWord(true);
                contents.setCaretPosition(0);
                details.add(new JScrollPane(contents), BorderLayout.CENTER);
                JPanel buttons = new JPanel();
                JButton folder = new JButton("Open output folder");
                folder.setEnabled(job != null);
                folder.addActionListener(event -> openLocalFile(job.toFile()));
                buttons.add(folder);
                JButton report = new JButton("Open report");
                report.setEnabled(validArtifact(artifacts, "report_html") != null);
                report.addActionListener(event -> openFileIfPresent(artifacts, "report_html"));
                buttons.add(report);
                JButton rawMap = new JButton("Open original full Map");
                rawMap.setEnabled(validArtifact(artifacts, "map_unmasked_rgb_tiff") != null);
                rawMap.addActionListener(event -> openImage(artifacts, "map_unmasked_rgb_tiff", "Original unmasked bivariate Map", sourceCalibration));
                buttons.add(rawMap);
                JButton close = new JButton("Close");
                close.addActionListener(event -> details.dispose());
                buttons.add(close);
                details.add(buttons, BorderLayout.SOUTH);
                details.pack();
                details.setLocationRelativeTo(null);
                details.setVisible(true);
                IJ.showStatus("LoopSN2N research: " + text(response, "execution_status") + " / " + text(response, "calibration_status") + " / " + text(decision, "label"));
            }
        });
    }

    private static File validArtifact(JsonObject artifacts, String key) {
        String path = text(artifacts, key);
        File file = new File(path);
        return file.isAbsolute() && file.isFile() ? file : null;
    }

    private static void openImage(JsonObject artifacts, String key, String title, Calibration calibration) {
        File file = validArtifact(artifacts, key);
        if (file == null) return;
        ImagePlus image = IJ.openImage(file.getAbsolutePath());
        if (image == null) { IJ.log("LoopSN2N could not open image: " + file); return; }
        if (calibration != null) image.setCalibration(calibration.copy());
        image.setTitle(title);
        image.show(); // Preserve TIFF's native precision. Never convert, equalize or apply contrast.
    }

    private static void openFileIfPresent(JsonObject artifacts, String key) {
        File file = validArtifact(artifacts, key);
        if (file != null) openLocalFile(file);
    }

    private static void openLocalFile(File file) {
        try {
            if (Desktop.isDesktopSupported()) Desktop.getDesktop().open(file);
            else IJ.log("Open locally: " + file);
        }
        catch (Exception exc) { IJ.log("Could not open " + file + ": " + exc.getMessage()); }
    }

    private static void onEdt(Runnable action) throws Exception {
        if (SwingUtilities.isEventDispatchThread()) action.run();
        else SwingUtilities.invokeAndWait(action);
    }
}
