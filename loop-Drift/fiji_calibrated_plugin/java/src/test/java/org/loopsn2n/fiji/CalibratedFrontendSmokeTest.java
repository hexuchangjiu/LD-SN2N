package org.loopsn2n.fiji;

import com.google.gson.JsonObject;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Roi;
import ij.measure.Calibration;
import ij.process.FloatProcessor;
import ij.process.ShortProcessor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.CancellationException;

/** Synthetic precision, calibration, axes, no-crop and overwrite/cancel regressions. */
public final class CalibratedFrontendSmokeTest {
    private CalibratedFrontendSmokeTest() { }
    public static void main(String[] args) throws Exception {
        if (args.length != 1) throw new IllegalArgumentException("usage: CalibratedFrontendSmokeTest <new-output-directory>");
        Path root = Files.createDirectory(Paths.get(args[0]));
        Path checkpoint = root.resolve("可信-test-placeholder.pth");
        Files.write(checkpoint, new byte[] { 0 }); // No loader or model is called by this test.
        for (boolean floating : new boolean[] { false, true }) {
            ImagePlus image = makeImage(floating);
            Roi roi = new Roi(1, 1, 2, 2);
            image.setRoi(roi);
            image.setSlice(2);
            double originalMin = image.getDisplayRangeMin();
            double originalMax = image.getDisplayRangeMax();
            Path job = root.resolve(floating ? "float32" : "uint16");
            Path request = LoopSN2NCalibratedCommand.exportAndWriteRequest(image, job, checkpoint, 1, "cpu", true, true);
            JsonObject payload = LoopSN2NCalibratedCommand.readResponse(request);
            ImagePlus copied = IJ.openImage(payload.get("stack_path").getAsString());
            CalibratedStackExportHarness.assertRawPixelsEqual(image, copied);
            require(copied.getNFrames() == 34 && copied.getNSlices() == 1, "Repeat export must encode time/repeat axes, not Z.");
            require(image.getRoi() == roi && image.getCurrentSlice() == 2, "Source ROI or plane changed.");
            require(image.getDisplayRangeMin() == originalMin && image.getDisplayRangeMax() == originalMax, "Source display range changed.");
            require(copied.getWidth() == 7 && copied.getHeight() == 5, "ROI cropped the full field.");
            require(Math.abs(copied.getCalibration().pixelWidth - 0.125) < 1e-9, "TIFF pixel calibration lost.");
            require(Math.abs(copied.getCalibration().pixelHeight - 0.25) < 1e-9, "TIFF Y calibration lost.");
            require(Math.abs(payload.getAsJsonObject("pixel_calibration").get("frame_interval").getAsDouble() - 0.4) < 1e-9, "Request time calibration lost.");
            require(Math.abs(payload.getAsJsonObject("pixel_calibration").get("x_origin").getAsDouble() - 2.5) < 1e-9, "Request origin lost.");
            require(payload.get("checkpoint_path").getAsString().contains("可信"), "UTF-8 checkpoint path changed.");
            require(payload.get("query_index").getAsInt() == 1, "Query index changed.");
            require("static_repeats".equals(payload.get("domain").getAsString()), "Domain missing.");
            expectRejected(() -> LoopSN2NCalibratedCommand.exportAndWriteRequest(image, job, checkpoint, 1, "cpu", true, true), "Existing output must not be overwritten.");
            expectRejected(() -> LoopSN2NCalibratedCommand.validateRepeatStack(image, false), "Static repeat declaration must be required.");
            Path safeRequest = LoopSN2NCalibratedCommand.exportAndWriteRequest(image, root.resolve("safe-" + floating), checkpoint, 1, "cpu", true, false);
            require(!LoopSN2NCalibratedCommand.readResponse(safeRequest).get("trust_checkpoint").getAsBoolean(), "Safe weights-only choice changed.");
            expectRejected(() -> LoopSN2NCalibratedCommand.exportAndWriteRequest(image, root.resolve("invalid-query"), checkpoint, 34, "cpu", true, true), "Invalid query index must fail.");
            image.close();
            copied.close();
        }
        ImagePlus z = makeImage(false);
        z.setOpenAsHyperStack(true);
        expectRejected(() -> LoopSN2NCalibratedCommand.validateRepeatStack(z, true), "Declared Z hyperstack must fail.");
        ImagePlus channels = makeImage(false);
        channels.setDimensions(2, 17, 1);
        expectRejected(() -> LoopSN2NCalibratedCommand.validateRepeatStack(channels, true), "Multichannel must fail.");
        ImagePlus repeatsT = makeImage(false);
        repeatsT.setDimensions(1, 1, 34);
        repeatsT.setOpenAsHyperStack(true);
        LoopSN2NCalibratedCommand.validateRepeatStack(repeatsT, true);
        Path repeatsJob = root.resolve("time-axis-repeats");
        Path repeatsRequest = LoopSN2NCalibratedCommand.exportAndWriteRequest(repeatsT, repeatsJob, checkpoint, 1, "auto", true, true);
        ImagePlus repeatsCopy = IJ.openImage(LoopSN2NCalibratedCommand.readResponse(repeatsRequest).get("stack_path").getAsString());
        CalibratedStackExportHarness.assertRawPixelsEqual(repeatsT, repeatsCopy);
        ImagePlus mixed = makeImage(false);
        mixed.getStack().addSlice(mixed.getStack().getProcessor(1).duplicate());
        mixed.getStack().addSlice(mixed.getStack().getProcessor(1).duplicate());
        mixed.setDimensions(1, 18, 2);
        expectRejected(() -> LoopSN2NCalibratedCommand.validateRepeatStack(mixed, true), "Mixed Z/T dimensions must fail.");
        ImagePlus single = new ImagePlus("single", new ShortProcessor(7, 5));
        expectRejected(() -> LoopSN2NCalibratedCommand.validateRepeatStack(single, true), "A single frame cannot establish repeat noise.");
        Path cancelled = Files.createDirectory(root.resolve("cancelled"));
        Files.write(cancelled.resolve("STOP_REQUESTED.json"), "{}".getBytes(java.nio.charset.StandardCharsets.UTF_8));
        expectRejected(() -> LoopSN2NCalibratedCommand.exportAndWriteRequest(makeImage(false), cancelled, checkpoint, 0, "cpu", true, true), "Stop marker must prevent export.");
        require(!Files.exists(cancelled.resolve("request.json")) && !Files.exists(cancelled.resolve("source_stack.tif")), "Cancelled export created input artifacts.");
        System.out.println("FIJI_CALIBRATED_FRONTEND_SMOKE_OK uint16=float32=exact ROI_ignored=true calibration=preserved axes=checked UTF8=checked stop=checked overwrite=refused");
    }

    private static ImagePlus makeImage(boolean floating) {
        ImageStack stack = new ImageStack(7, 5);
        for (int frame = 0; frame < 34; frame++) {
            if (floating) {
                float[] values = new float[35];
                for (int i = 0; i < values.length; i++) values[i] = (float) ((i - 12.0) * 13.3763 + frame * 0.07);
                values[0] = -0.0f;
                values[1] = 1.2345678e-20f;
                stack.addSlice(new FloatProcessor(7, 5, values));
            }
            else {
                short[] values = new short[35];
                for (int i = 0; i < values.length; i++) values[i] = (short) (i * 1900 + frame * 3);
                values[0] = (short) 65535;
                stack.addSlice(new ShortProcessor(7, 5, values, null));
            }
        }
        ImagePlus image = new ImagePlus(floating ? "float" : "uint16", stack);
        Calibration calibration = image.getCalibration();
        calibration.pixelWidth = 0.125;
        calibration.pixelHeight = 0.25;
        calibration.pixelDepth = 0.5;
        calibration.setUnit("um");
        calibration.frameInterval = 0.4;
        calibration.xOrigin = 2.5;
        image.setCalibration(calibration);
        return image;
    }

    private interface CheckedOperation { void run() throws Exception; }
    private static void expectRejected(CheckedOperation operation, String message) throws Exception {
        try { operation.run(); }
        catch (IllegalArgumentException | java.io.IOException | CancellationException expected) { return; }
        throw new AssertionError(message);
    }
    private static void require(boolean condition, String message) { if (!condition) throw new AssertionError(message); }
}
