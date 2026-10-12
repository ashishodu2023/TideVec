package io.tidevec;

import io.tidevec.TideVecClient.*;
import org.junit.jupiter.api.*;

import java.io.IOException;
import java.net.ServerSocket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static io.tidevec.TideVecClient.Filters.*;
import static org.junit.jupiter.api.Assertions.*;

/** Get / scroll / rich filters / as_of / hybrid / aliases against a live server. */
class RetrievalTest {
    private static final long T0 = 1_700_000_000_000L;
    private static Process server;
    private static Path dataDir;
    private static TideVecClient db;

    @BeforeAll
    static void startServer() throws Exception {
        String fixedPort = System.getenv("TIDEVEC_TEST_PORT");
        int port;
        if (fixedPort != null && !fixedPort.isEmpty()) {
            port = Integer.parseInt(fixedPort);
        } else {
            String bin = Optional.ofNullable(System.getenv("TIDEVEC_SERVER_BIN"))
                .orElse("../../build/tidevec-server");
            Assumptions.assumeTrue(Files.isExecutable(Path.of(bin)),
                "tidevec-server not found at " + bin);
            try (ServerSocket s = new ServerSocket(0)) { port = s.getLocalPort(); }
            dataDir = Files.createTempDirectory("tidevec-java-retrieval");
            server = new ProcessBuilder(bin, "--port", String.valueOf(port),
                    "--data-dir", dataDir.toString(), "--no-require-auth", "--quiet",
                    "--rate-limit", "100000")
                .redirectErrorStream(true)
                .redirectOutput(ProcessBuilder.Redirect.DISCARD)
                .start();
        }
        db = TideVecClient.builder("127.0.0.1", port).timeout(Duration.ofSeconds(10)).build();
        long deadline = System.currentTimeMillis() + 10_000;
        while (!db.ping()) {
            assertTrue(System.currentTimeMillis() < deadline, "server did not start");
            Thread.sleep(100);
        }
    }

    @AfterAll
    static void stopServer() throws Exception {
        if (server != null) {
            server.destroy();
            server.waitFor();
        }
        if (dataDir != null) {
            try (var walk = Files.walk(dataDir)) {
                walk.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
    }

    private static void seed(String name) throws IOException {
        try { db.dropCollection(name); } catch (TideVecException e) { assertTrue(e.isNotFound()); }
        db.createCollection(CollectionConfig.builder(name).dim(4).nShards(2).textField("text").build());
        List<Vector> vecs = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            String text = i == 17 ? "error code TV-4711 on shutdown" : "plain note " + i;
            vecs.add(Vector.builder(String.format("r%02d", i), new float[]{1f, i / 50f, 0f, 0f})
                .createdAt(T0 + i * 1000L)
                .payload(Map.of("lang", i % 2 == 1 ? "en" : "de",
                                "year", String.valueOf(2000 + i), "text", text))
                .build());
        }
        db.upsert(name, vecs);
    }

    private static List<String> sortedIds(SearchResponse res) {
        List<String> ids = new ArrayList<>();
        for (SearchHit h : res.hits()) ids.add(h.id());
        Collections.sort(ids);
        return ids;
    }

    @Test
    void getScrollAndCount() throws IOException {
        seed("java_retrieval");
        assertEquals("text", db.getCollection("java_retrieval").textField());

        StoredVector rec = db.get("java_retrieval", "r07", true).orElseThrow();
        assertEquals("en", rec.payload().get("lang"));
        assertEquals(4, rec.embedding().length);
        assertEquals(T0 + 7000, rec.createdAt());
        assertTrue(db.get("java_retrieval", "nope", false).isEmpty());

        List<StoredVector> many = db.getMany("java_retrieval", List.of("r03", "zz", "r01"), false);
        assertEquals(List.of("r03", "r01"), many.stream().map(StoredVector::id).toList());
        assertNull(many.get(0).embedding());

        List<String> seen = new ArrayList<>();
        String cursor = null;
        int pages = 0;
        do {
            ScrollPage page = db.scroll("java_retrieval",
                ScrollOptions.builder().limit(16).cursor(cursor).build());
            page.records().forEach(r -> seen.add(r.id()));
            cursor = page.nextCursor();
            pages++;
        } while (cursor != null);
        assertEquals(4, pages);
        assertEquals(50, seen.size());

        List<String> en = db.listIds("java_retrieval",
            ScrollOptions.builder().limit(7).where(eq("lang", "en")).build());
        assertEquals(25, en.size());
        assertEquals("r01", en.get(0));

        assertEquals(28, db.count("java_retrieval",
            or(eq("lang", "de"), range("year", 2045, null)), null));
        assertEquals(10, db.count("java_retrieval", null, T0 + 9000));
        assertEquals(10, db.deleteWhere("java_retrieval", range("year", 2040, null)));
        assertEquals(40, db.count("java_retrieval"));
    }

    @Test
    void richFilterAsOfAndHybrid() throws IOException {
        seed("java_hybrid");
        float[] q = {1f, 0f, 0f, 0f};
        SearchResponse res = db.search("java_hybrid", q, SearchOptions.builder().topK(50)
            .temporalBlend(0f)
            .where(and(in("lang", List.of("en")), range("year", 2010, 2020))).build());
        assertEquals(List.of("r11", "r13", "r15", "r17", "r19"), sortedIds(res));

        res = db.search("java_hybrid", q, SearchOptions.builder().topK(50)
            .where(not(range("year", 2002, null))).build());
        assertEquals(List.of("r00", "r01"), sortedIds(res));

        res = db.search("java_hybrid", q, SearchOptions.builder().topK(50).asOf(T0 + 4000).build());
        assertEquals(5, res.hits().size());

        res = db.search("java_hybrid", null, SearchOptions.builder().topK(3).text("TV-4711").build());
        assertEquals(1, res.hits().size());
        assertEquals("r17", res.hits().get(0).id());
        assertTrue(res.hits().get(0).textScore() > 0);

        res = db.search("java_hybrid", q, SearchOptions.builder().topK(3).text("tv-4711")
            .hybrid("weighted", 0.2f).build());
        assertEquals("r17", res.hits().get(0).id());

        assertThrows(IllegalArgumentException.class,
            () -> db.search("java_hybrid", null, SearchOptions.builder().build()));
    }

    @Test
    void aliasesSwapAtomically() throws IOException {
        for (String n : List.of("java_kb_v1", "java_kb_v2")) {
            try { db.dropCollection(n); } catch (TideVecException e) { assertTrue(e.isNotFound()); }
            db.createCollection(CollectionConfig.builder(n).dim(2).nShards(1).build());
            db.upsert(n, List.of(Vector.builder(n + "_doc", new float[]{1f, 0f}).build()));
        }
        assertNull(db.setAlias("java_kb", "java_kb_v1"));
        SearchOptions top1 = SearchOptions.builder().topK(1).build();
        assertEquals("java_kb_v1_doc", db.search("java_kb", new float[]{1f, 0f}, top1).hits().get(0).id());

        assertEquals("java_kb_v1", db.setAlias("java_kb", "java_kb_v2"));
        assertEquals("java_kb_v2_doc", db.search("java_kb", new float[]{1f, 0f}, top1).hits().get(0).id());
        CollectionInfo info = db.getCollection("java_kb");
        assertEquals("java_kb_v2", info.name());
        assertTrue(info.aliases().contains("java_kb"));
        assertEquals("java_kb_v2", db.listAliases().get("java_kb"));

        db.deleteAlias("java_kb");
        TideVecException e = assertThrows(TideVecException.class, () -> db.deleteAlias("java_kb"));
        assertTrue(e.isNotFound());
    }

    @Test
    void compactReclaimsTombstones() throws IOException {
        String name = "java_compact";
        try { db.dropCollection(name); } catch (TideVecException e) { assertTrue(e.isNotFound()); }
        db.createCollection(CollectionConfig.builder(name).dim(4).nShards(2)
                                            .compactionRatio(1000).build());
        List<Vector> vecs = new ArrayList<>();
        for (int i = 0; i < 120; i++)
            vecs.add(Vector.builder(String.format("k%03d", i),
                                    new float[]{1f, i % 9, i % 13, i})
                .payload(Map.of("keep", String.valueOf(i % 4 == 0))).build());
        db.upsert(name, vecs);
        assertEquals(90, db.deleteWhere(name, eq("keep", "false")));

        CompactionStatus before = db.compactionStatus(name);
        assertEquals(90, before.tombstones());
        assertFalse(before.running());
        CompactionStatus after = db.compact(name, true);
        assertEquals(2, after.started());
        assertFalse(after.running());
        assertEquals(0, after.tombstones());
        assertEquals(90, after.lastReclaimed());
        assertTrue(after.indexBytes() < before.indexBytes());
        assertEquals(30, db.count(name, eq("keep", "true"), null));
        assertEquals(0, db.compact(name, false).started());
    }
}
