package org.loopsn2n.fiji;

import net.imagej.ImageJ;
import org.scijava.command.Command;
import org.scijava.plugin.PluginInfo;

/** Real SciJava discovery with both the installed strict and new research plugins present. */
public final class CalibratedPluginDiscoverySmokeTest {
    static { net.imagej.patcher.LegacyInjector.preinit(); }
    private CalibratedPluginDiscoverySmokeTest() { }
    public static void main(String[] args) {
        ImageJ imagej = new ImageJ();
        try {
            int matches = 0;
            for (PluginInfo<Command> info : imagej.plugin().getPluginsOfType(Command.class)) {
                if (LoopSN2NCalibratedCommand.class.getName().equals(info.getClassName())) {
                    if (!info.getMenuPath().toString().contains("Calibrate noise and assess (research)"))
                        throw new AssertionError("Unexpected research menu path: " + info.getMenuPath());
                    matches++;
                }
            }
            if (matches != 1) throw new AssertionError("Expected exactly one research command, observed " + matches);
            System.out.println("FIJI_CALIBRATED_DISCOVERY_OK class=" + LoopSN2NCalibratedCommand.class.getName()
                + " menu=Plugins>LoopSN2N>Calibrate noise and assess (research)");
        }
        finally { imagej.context().dispose(); }
    }
}
