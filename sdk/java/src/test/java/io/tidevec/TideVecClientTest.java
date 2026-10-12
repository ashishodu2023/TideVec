package io.tidevec;

import io.tidevec.TideVecClient.*;
import org.junit.jupiter.api.*;

import java.io.IOException;
import java.net.ServerSocket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests against a live tidevec-server. Uses TIDEVEC_TEST_PORT
 * if set (server on localhost), otherwise starts TIDEVEC_SERVER_BIN
 * (default ../../build/tidevec-server). Skipped when neither is available.
 */
class TideVecClientTest {
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
            dataDir = Files.createTempDirectory("tidevec-java-test");
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

    private static void fresh(CollectionConfig cfg) throws IOException {
        try { db.dropCollection(cfg.name()); } catch (TideVecException e) { assertTrue(e.isNotFound()); }
        db.createCollection(cfg);
    }

    private static float[] randVec(Random r, int dim) {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) v[i] = r.nextFloat() * 2 - 1;
        return v;
    }

    @Test
    void searchRanksFreshAboveStaleAndRoundTripsPayload() throws IOException {
        Random r = new Random(1);
        fresh(CollectionConfig.builder("java_search").dim(8).halfLifeMs(HalfLife.ONE_WEEK).build());
        float[] q = randVec(r, 8);
        long old = System.currentTimeMillis() - 30L * 24 * 3600 * 1000;
        String tricky = "quote\" backslash\\ newline\n é";
        List<Vector> vs = new ArrayList<>(List.of(
            Vector.builder("fresh", q).payload(Map.of("note", tricky, "lang", "java")).build(),
            Vector.builder("stale", q).createdAt(old).build()));
        for (int i = 0; i < 10; i++) vs.add(Vector.builder("n" + i, randVec(r, 8)).build());
        db.upsert("java_search", vs);
        assertEquals(12, db.count("java_search"));

        SearchResponse res = db.search("java_search", q, SearchOptions.builder().topK(12).build());
        assertEquals("fresh", res.hits().get(0).id());
        assertEquals(tricky, res.hits().get(0).payload().get("note"));
        SearchHit stale = res.hits().stream().filter(h -> h.id().equals("stale")).findFirst().orElseThrow();
        assertTrue(stale.stalenessWarning());
        assertTrue(stale.score() < res.hits().get(0).score());
        assertEquals(old, stale.createdAt());

        SearchResponse semantic = db.search("java_search", q,
            SearchOptions.builder().topK(2).temporalBlend(0f).build());
        assertEquals(semantic.hits().get(0).score(), semantic.hits().get(1).score(), 1e-6);

        SearchResponse filtered = db.search("java_search", q,
            SearchOptions.builder().topK(5).filter("lang=java").build());
        assertEquals(List.of("fresh"), filtered.hits().stream().map(SearchHit::id).toList());
    }

    @Test
    void supersessionMarksOlderVersion() throws IOException {
        fresh(CollectionConfig.builder("java_super").dim(4).supersession(0.9f, "doc").build());
        assertTrue(db.getCollection("java_super").supersessionEnabled());
        long hourAgo = System.currentTimeMillis() - 3_600_000L;
        db.upsert("java_super", List.of(Vector.builder("policy_v1", new float[]{1f, 0.1f, 0f, 0f})
            .payload(Map.of("doc", "refund")).createdAt(hourAgo).build()));
        db.upsert("java_super", List.of(Vector.builder("policy_v2", new float[]{1f, 0.12f, 0f, 0f})
            .payload(Map.of("doc", "refund")).build()));

        SearchResponse res = db.search("java_super", new float[]{1f, 0.1f, 0f, 0f},
            SearchOptions.builder().topK(2).temporalBlend(0f).build());
        Map<String, SearchHit> byId = new HashMap<>();
        res.hits().forEach(h -> byId.put(h.id(), h));
        assertTrue(byId.get("policy_v1").isSuperseded());
        assertEquals(List.of("policy_v2"), byId.get("policy_v1").supersededBy());
        assertFalse(byId.get("policy_v2").isSuperseded());

        SearchResponse penalised = db.search("java_super", new float[]{1f, 0.1f, 0f, 0f},
            SearchOptions.builder().topK(2).temporalBlend(0f).supersededPenalty(0.5f).build());
        assertEquals("policy_v2", penalised.hits().get(0).id());
    }

    @Test
    void deleteByFilterIdsAndEdges() throws IOException {
        Random r = new Random(2);
        fresh(CollectionConfig.builder("java_delete").dim(4).nShards(2).build());
        List<Vector> vs = new ArrayList<>();
        for (int i = 0; i < 10; i++)
            vs.add(Vector.builder("v" + i, randVec(r, 4))
                .payload(Map.of("grp", i % 2 == 0 ? "drop" : "keep")).build());
        db.upsert("java_delete", vs);

        assertEquals(5, db.deleteWhere("java_delete", "grp=drop"));
        assertEquals(1, db.deleteVectors("java_delete", List.of("v1", "missing")));
        assertEquals(4, db.count("java_delete"));
        assertThrows(IllegalArgumentException.class, () -> db.deleteWhere("java_delete", "no-equals"));

        assertEquals(1, db.addEdges("java_delete", List.of(Edge.contradicts("v3", "v5"))));
        SearchResponse res = db.search("java_delete", randVec(r, 4),
            SearchOptions.builder().topK(4).mode("contradiction_check").build());
        assertEquals(4, res.hits().size());
        db.setTemporal("java_delete", HalfLife.ONE_DAY, 0.5f);
    }

    @Test
    void errorsAndStatus() throws IOException {
        TideVecException e = assertThrows(TideVecException.class,
            () -> db.search("nope", new float[]{1f}, SearchOptions.builder().build()));
        assertTrue(e.isNotFound());
        assertEquals("standalone", db.clusterStatus().get("mode"));
        assertEquals("0.3.0", db.info().get("version"));
        assertTrue(db.listCollections().stream().anyMatch(c -> !c.name().isEmpty()) ||
                   db.listCollections().isEmpty());
    }
}
