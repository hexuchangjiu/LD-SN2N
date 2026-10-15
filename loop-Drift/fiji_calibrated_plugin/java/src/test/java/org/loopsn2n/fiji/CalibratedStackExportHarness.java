package org.loopsn2n.fiji;

import com.google.gson.JsonObject;
import ij.IJ;
import ij.ImagePlus;
import java.nio.file.Path;
import java.nio.file.Paths;

/** Real ImagePlus -> untouched full-precision TIFF -> bridge request; does not run a model. */
public final class CalibratedStackExportHarness {
    private CalibratedStackExportHarness() { }

    public static void main(String[] args) throws Exception {
        if (args.length < 3 || args.length > 6)
            throw new IllegalArgumentException("usage: CalibratedStackExportHarness <stack.tif> <checkpoint.pth> <new-job-directory> [auto|cuda|cpu] [query_index_0based] [allow_trusted_legacy=true|false]");
        ImagePlus source = IJ.openImage(args[0]);
        if (source == null) throw new IllegalStateException("Cannot open input image.");
        int queryIndex = args.length > 4 ? Integer.parseInt(args[4]) : 0;
        String device = args.length > 3 ? args[3] : "auto";
        if (args.length > 5 && !"true".equals(args[5]) && !"false".equals(args[5]))
            throw new IllegalArgumentException("Trusted legacy flag must be true or false.");
        boolean trustedLegacy = args.length > 5 && Boolean.parseBoolean(args[5]);
        Path request = LoopSN2NCalibratedCommand.exportAndWriteRequest(source, Paths.get(args[2]), Paths.get(args[1]),
            queryIndex, device, true, trustedLegacy);
        JsonObject payload = LoopSN2NCalibratedCommand.readResponse(request);
        ImagePlus exported = IJ.openImage(payload.get("stack_path").getAsString());
        assertRawPixelsEqual(source, exported);
        if (exported.getNFrames() != source.getStackSize() || exported.getNSlices() != 1)
            throw new AssertionError("Export did not encode the declared temporal repeats.");
        if (payload.get("query_index").getAsInt() != queryIndex) throw new AssertionError("Query index changed.");
        if (!"static_repeats".equals(payload.get("domain").getAsString())) throw new AssertionError("Domain changed.");
        if (!"native_fullframe".equals(payload.get("sn2n_profile").getAsString())) throw new AssertionError("Adapter changed.");
        System.out.println("FIJI_CALIBRATED_EXPORT_OK shape=" + source.getStackSize() + "x" + source.getHeight() + "x" + source.getWidth()
            + " bitDepth=" + source.getBitDepth() + " query_index=" + queryIndex + " request=" + request.toAbsolutePath());
        source.close();
        exported.close();
    }

    public static void assertRawPixelsEqual(ImagePlus source, ImagePlus exported) {
        if (exported == null || source.getBitDepth() != exported.getBitDepth() || source.getWidth() != exported.getWidth()
                || source.getHeight() != exported.getHeight() || source.getStackSize() != exported.getStackSize())
            throw new AssertionError("TIFF dimensions or precision changed.");
        for (int slice = 1; slice <= source.getStackSize(); slice++) {
            Object original = source.getStack().getPixels(slice);
            Object copied = exported.getStack().getPixels(slice);
            if (original instanceof short[]) {
                if (!java.util.Arrays.equals((short[]) original, (short[]) copied)) throw new AssertionError("uint16 pixels changed.");
            }
            else if (original instanceof float[]) {
                float[] first = (float[]) original;
                float[] second = (float[]) copied;
                for (int index = 0; index < first.length; index++)
                    if (Float.floatToRawIntBits(first[index]) != Float.floatToRawIntBits(second[index]))
                        throw new AssertionError("float32 pixels changed at slice " + slice + ", index " + index);
            }
            else throw new AssertionError("Unsupported source precision in test.");
        }
    }
}
