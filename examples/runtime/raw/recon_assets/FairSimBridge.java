// Thin headless adapter to GPL-2.0 fairSIM v1.4.1; no reconstruction algorithm reimplementation.
import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import org.fairsim.linalg.*;
import org.fairsim.sim_algorithm.*;
import org.fairsim.utils.*;

public class FairSimBridge {
    static void writeFloat(Path path, Vec2d.Real image) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(image.vectorSize()*4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : image.vectorData()) {
            if (!Float.isFinite(v)) throw new IOException("Nonfinite reconstruction");
            b.putFloat(v);
        }
        Files.write(path, b.array(), StandardOpenOption.CREATE_NEW);
    }
    public static void main(String[] args) throws Exception {
        try {
            if(args.length!=1) throw new IllegalArgumentException("Supply one properties file");
            Properties c = new Properties();
            try(InputStream s=Files.newInputStream(Paths.get(args[0]))) { c.load(s); }
            int n=Integer.parseInt(c.getProperty("size"));
            Path out=Paths.get(c.getProperty("output")); Files.createDirectories(out);
            byte[] bytes=Files.readAllBytes(Paths.get(c.getProperty("input")));
            if(bytes.length!=9*n*n*4) throw new IllegalArgumentException("Expect nine square float32 frames");
            ByteBuffer buf=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            Vec2d.Cplx[][] fft=new Vec2d.Cplx[3][3];
            int fade=Integer.parseInt(c.getProperty("fade", "10"));
            boolean phaseFast=c.getProperty("order", "phase_fast").equals("phase_fast");
            for(int j=0;j<9;j++) {
                Vec2d.Real r=Vec2d.createReal(n,n);
                for(int k=0;k<n*n;k++) { float v=buf.getFloat(); if(!Float.isFinite(v)) throw new IOException("Nonfinite input"); r.set(k,v); }
                if(fade>0) SimUtils.fadeBorderCos(r,fade);
                int a=phaseFast ? j/3 : j%3, p=phaseFast ? j%3 : j/3;
                fft[a][p]=Vec2d.createCplx(n,n); fft[a][p].copy(r); Transforms.fft2d(fft[a][p],false);
            }
            OtfProvider otf;
            if(c.containsKey("otf_radial_mrc")) {
                byte[] ob=Files.readAllBytes(Paths.get(c.getProperty("otf_radial_mrc")));
                ByteBuffer oh=ByteBuffer.wrap(ob).order(ByteOrder.LITTLE_ENDIAN);
                int samples=oh.getInt(0);
                if(oh.getInt(4)!=1 || oh.getInt(8)!=1 || oh.getInt(12)!=4 || ob.length!=1024+samples*8)
                    throw new IOException("Unsupported radial OTF layout");
                float[] curve=new float[samples*2]; oh.position(1024);
                for(int i=0;i<curve.length;i++) curve[i]=oh.getFloat();
                Conf oc=new Conf("fairsim"); Conf.Folder of=oc.r().mk("otf2d");
                of.newDbl("NA").setVal(Double.parseDouble(c.getProperty("na","1.3")));
                of.newInt("emission").setVal(Integer.parseInt(c.getProperty("emission_nm","525")));
                Conf.Folder od=of.mk("data"); od.newInt("bands").setVal(1);
                od.newInt("samples").setVal(samples); od.newDbl("cycles").setVal(oh.getFloat(40));
                od.newData("band-0").setVal(Conf.toByte(curve));
                otf=OtfProvider.loadFromConfig(oc);
            }
            else if(c.containsKey("otf_config")) otf=OtfProvider.loadFromConfig(Conf.loadFile(c.getProperty("otf_config")));
            else otf=OtfProvider.fromEstimate(Double.parseDouble(c.getProperty("na","1.3")),
                    Double.parseDouble(c.getProperty("emission_nm","525")),Double.parseDouble(c.getProperty("otf_a","1.0")));
            otf.switchAttenuation(false);
            SimParam param=SimParam.create(2,3,3,n,Double.parseDouble(c.getProperty("pixel_um","0.0626")),otf);
            param.setWienerFilter(Double.parseDouble(c.getProperty("wiener","0.05")));
            param.setApoCutoff(Double.parseDouble(c.getProperty("apo_cutoff","2.0")));
            param.setApoBend(Double.parseDouble(c.getProperty("apo_bend","0.9")));
            if(c.containsKey("frozen_params")) {
                SimParam old=SimParam.loadConfig(Conf.loadFile(c.getProperty("frozen_params")).r());
                for(int a=0;a<3;a++) {
                    param.dir(a).setPxPy(old.dir(a).px(1),old.dir(a).py(1));
                    param.dir(a).setPhaOff(old.dir(a).getPhaOff());
                    param.dir(a).setModulation(1,old.dir(a).getRawModulations()[1]);
                }
            } else SimAlgorithm.estimateParameters(param,fft,1,0.6,null,0,null);
            Conf save=new Conf("fairsim"); param.saveConfig(save.r()); otf.saveConfig(save);
            save.saveFile(out.resolve("parameters.xml").toFile());
            StringBuilder csv=new StringBuilder("angle,kx_pixel,ky_pixel,phase_rad,raw_modulation,used_modulation\n");
            for(int a=0;a<3;a++) csv.append(String.format(Locale.ROOT,"%d,%.12g,%.12g,%.12g,%.12g,%.12g%n",a,
                param.dir(a).px(1),param.dir(a).py(1),param.dir(a).getPhaOff(),param.dir(a).getRawModulations()[1],param.dir(a).getModulations()[1]));
            Files.writeString(out.resolve("illumination.csv"),csv.toString(),StandardOpenOption.CREATE_NEW);
            Vec2d.Real wf=Vec2d.createReal(2*n,2*n);
            Vec2d.Real recon=SimAlgorithm.runReconstruction(param,fft,null,-1,true,SimParam.CLIPSCALE.NONE,wf,null,null);
            writeFloat(out.resolve("reconstruction.f32"),recon); writeFloat(out.resolve("widefield.f32"),wf);
            System.out.println("B0_RECONSTRUCTION_COMPLETE "+out);
        } finally { SimpleMT.shutdown(); }
        System.exit(0);
    }
}
