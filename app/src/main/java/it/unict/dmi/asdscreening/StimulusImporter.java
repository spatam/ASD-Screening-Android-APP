package it.unict.dmi.asdscreening;

import android.content.ContentResolver;
import android.content.Context;
import android.net.Uri;

import androidx.documentfile.provider.DocumentFile;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Copies the Saliency4ASD stimuli from a folder the user picks into private storage.
 *
 * The user downloads TrainingDataset.rar from Zenodo (record 2647418), extracts it and picks
 * either the TrainingData/Images folder or one of its parents. Every file is checked against the
 * SHA-256 list in assets/stimuli_sha256.txt, so the app only ever shows the exact pictures the
 * models were trained on.
 */
public final class StimulusImporter {

    public interface ProgressListener {
        void onProgress(int done, int total);
    }

    public static final class Result {
        public final int verified;
        public final int missing;
        public final int rejected;

        Result(int verified, int missing, int rejected) {
            this.verified = verified;
            this.missing = missing;
            this.rejected = rejected;
        }

        public boolean isComplete() {
            return verified == StimulusRepository.COUNT;
        }
    }

    private static final String MANIFEST = "stimuli_sha256.txt";
    private static final Pattern STIMULUS_NAME = Pattern.compile("0*(\\d{1,3})\\.png", Pattern.CASE_INSENSITIVE);
    private static final int MAX_DEPTH = 3;

    private StimulusImporter() {}

    public static Result importFrom(Context context, Uri treeUri, ProgressListener listener) throws IOException {
        String[] expected = readManifest(context);
        DocumentFile root = DocumentFile.fromTreeUri(context, treeUri);
        if (root == null) throw new IOException("Cannot open the selected folder");
        Map<Integer, DocumentFile> found = findStimuli(root, 0);

        File dir = StimulusRepository.importedDir(context);
        if (!dir.isDirectory() && !dir.mkdirs()) throw new IOException("Cannot create " + dir);

        ContentResolver resolver = context.getContentResolver();
        int verified = 0, missing = 0, rejected = 0;
        for (int i = 0; i < StimulusRepository.COUNT; i++) {
            File target = new File(dir, StimulusRepository.fileName(i));
            DocumentFile source = found.get(i + 1);
            if (source == null) {
                if (target.isFile() && expected[i].equals(sha256(target))) verified++;
                else missing++;
            } else if (copyVerified(resolver, source.getUri(), target, expected[i])) {
                verified++;
            } else {
                rejected++;
            }
            if (listener != null) listener.onProgress(i + 1, StimulusRepository.COUNT);
        }
        return new Result(verified, missing, rejected);
    }

    /** Numbered PNGs of the first folder, searched breadth-first, that holds the most of them. */
    static Map<Integer, DocumentFile> findStimuli(DocumentFile dir, int depth) {
        Map<Integer, DocumentFile> here = new HashMap<>();
        DocumentFile[] children = dir.listFiles();
        for (DocumentFile child : children) {
            String name = child.getName();
            if (child.isFile() && name != null) {
                Matcher m = STIMULUS_NAME.matcher(name);
                if (m.matches()) {
                    int index = Integer.parseInt(m.group(1));
                    if (index >= 1 && index <= StimulusRepository.COUNT) here.put(index, child);
                }
            }
        }
        if (here.size() >= StimulusRepository.COUNT || depth >= MAX_DEPTH) return here;
        Map<Integer, DocumentFile> best = here;
        for (DocumentFile child : children) {
            if (!child.isDirectory()) continue;
            Map<Integer, DocumentFile> sub = findStimuli(child, depth + 1);
            if (sub.size() > best.size()) best = sub;
            if (best.size() >= StimulusRepository.COUNT) break;
        }
        return best;
    }

    private static boolean copyVerified(ContentResolver resolver, Uri source, File target, String expectedSha)
            throws IOException {
        File partial = new File(target.getPath() + ".part");
        MessageDigest digest = newSha256();
        try (InputStream in = resolver.openInputStream(source);
             OutputStream out = new FileOutputStream(partial)) {
            if (in == null) throw new IOException("Cannot read " + source);
            byte[] buffer = new byte[64 * 1024];
            int n;
            while ((n = in.read(buffer)) != -1) {
                digest.update(buffer, 0, n);
                out.write(buffer, 0, n);
            }
        }
        if (!expectedSha.equals(hex(digest.digest()))) {
            partial.delete();
            return false;
        }
        if (target.exists() && !target.delete()) throw new IOException("Cannot replace " + target);
        if (!partial.renameTo(target)) throw new IOException("Cannot write " + target);
        return true;
    }

    private static String[] readManifest(Context context) throws IOException {
        String[] expected = new String[StimulusRepository.COUNT];
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open(MANIFEST), StandardCharsets.US_ASCII))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.trim().split("\\s+");
                if (parts.length != 2) continue;
                Matcher m = STIMULUS_NAME.matcher(parts[1]);
                if (m.matches()) expected[Integer.parseInt(m.group(1)) - 1] = parts[0].toLowerCase(Locale.ROOT);
            }
        }
        for (String sha : expected) {
            if (sha == null) throw new IOException(MANIFEST + " is incomplete");
        }
        return expected;
    }

    private static String sha256(File file) throws IOException {
        MessageDigest digest = newSha256();
        try (InputStream in = new FileInputStream(file)) {
            byte[] buffer = new byte[64 * 1024];
            int n;
            while ((n = in.read(buffer)) != -1) digest.update(buffer, 0, n);
        }
        return hex(digest.digest());
    }

    private static MessageDigest newSha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    private static String hex(byte[] bytes) {
        char[] digits = "0123456789abcdef".toCharArray();
        char[] out = new char[bytes.length * 2];
        for (int i = 0; i < bytes.length; i++) {
            out[2 * i] = digits[(bytes[i] >> 4) & 0xF];
            out[2 * i + 1] = digits[bytes[i] & 0xF];
        }
        return new String(out);
    }
}
