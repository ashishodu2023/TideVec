"""
TideVec Python SDK
====================
Temporally-aware causal vector database client.

Install:
    pip install tidevec-py

Quick start:
    from tidevec import TideVec

    db = TideVec("localhost:6399")

    db.create_collection("docs", dim=768)
    db.upsert("docs", [{"id": "v1", "embedding": [0.1, ...], "payload": {"src": "wiki"}}])
    results = db.search("docs", query_vector=[0.1, ...], top_k=10, temporal_blend=0.3)
    for r in results:
        print(r.id, r.score, r.temporal_score)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union

import grpc

try:
    from . import tidevec_pb2 as pb
    from . import tidevec_pb2_grpc as pb_grpc
except ImportError:
    import tidevec_pb2 as pb          # type: ignore
    import tidevec_pb2_grpc as pb_grpc  # type: ignore


# ================================================================
# Typed result objects
# ================================================================

@dataclass
class SearchHit:
    """One result from a vector search."""
    id:               str
    score:            float
    vector_score:     float
    temporal_score:   float
    payload:          Dict[str, str]
    created_at:       int
    staleness_warning: bool        = False
    staleness_reason:  str         = ""
    causal_neighbors:  List[str]   = field(default_factory=list)
    contradicted_by:   List[str]   = field(default_factory=list)

    @classmethod
    def _from_proto(cls, r) -> "SearchHit":
        return cls(
            id               = r.id,
            score            = r.score,
            vector_score     = r.vector_score,
            temporal_score   = r.temporal_score,
            payload          = dict(r.payload),
            created_at       = r.created_at,
            staleness_warning= r.staleness_warning,
            staleness_reason = r.staleness_reason,
            causal_neighbors = list(r.causal_neighbors),
            contradicted_by  = list(r.contradicted_by),
        )

    def __repr__(self) -> str:
        return (f"SearchHit(id={self.id!r}, score={self.score:.4f}, "
                f"temporal={self.temporal_score:.3f})")


@dataclass
class SearchResponse:
    """Full response from a search call."""
    hits:       List[SearchHit]
    count:      int
    latency_ms: float = 0.0
    query_id:   str   = ""
    strategy:   str   = ""
    staleness_warnings:   List[dict] = field(default_factory=list)
    contradiction_alerts: List[dict] = field(default_factory=list)

    def __iter__(self) -> Iterator[SearchHit]:
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, idx: int) -> SearchHit:
        return self.hits[idx]


@dataclass
class CollectionInfo:
    name:       str
    n_vectors:  int
    n_shards:   int
    dim:        int
    index_type: str
    metric:     str


# ================================================================
# Temporal preset helpers
# ================================================================

class HalfLife:
    """Pre-built temporal decay presets."""
    ONE_HOUR  = 3_600_000
    ONE_DAY   = 86_400_000
    ONE_WEEK  = 604_800_000
    ONE_MONTH = 2_592_000_000
    ONE_YEAR  = 31_536_000_000

    AGENT_SESSION  = ONE_HOUR
    NEWS_FEED      = ONE_DAY
    SUPPORT_TICKET = ONE_MONTH
    DOCUMENT_STORE = ONE_YEAR


# ================================================================
# Main client
# ================================================================

class TideVec:
    """
    TideVec Python client.

    Supports both REST (via requests) and gRPC (via grpc).
    gRPC is preferred for high-throughput production use.

    Args:
        host:       Server host:port  (default "localhost:6399")
        api_key:    Optional API key  (set via X-Api-Key header)
        timeout:    Default RPC timeout in seconds
        use_grpc:   Use gRPC transport (default True if grpc available)
        tls:        Use TLS for gRPC (default False for local dev)
    """

    def __init__(
        self,
        host:     str  = "localhost:6399",
        api_key:  str  = "",
        timeout:  float = 30.0,
        use_grpc: bool  = True,
        tls:      bool  = False,
    ):
        self._host    = host
        self._api_key = api_key
        self._timeout = timeout
        self._channel: Optional[grpc.Channel] = None
        self._stub:    Optional[pb_grpc.TideVecStub] = None

        if use_grpc:
            self._connect_grpc(tls)

    def _connect_grpc(self, tls: bool) -> None:
        creds = grpc.ssl_channel_credentials() if tls else None
        options = [
            ("grpc.max_send_message_length",    256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ("grpc.keepalive_time_ms",          30_000),
        ]
        if creds:
            self._channel = grpc.secure_channel(self._host, creds, options)
        else:
            self._channel = grpc.insecure_channel(self._host, options)
        self._stub = pb_grpc.TideVecStub(self._channel)

    def _meta(self) -> List[tuple]:
        if self._api_key:
            return [("x-api-key", self._api_key)]
        return []

    def close(self) -> None:
        if self._channel:
            self._channel.close()

    def __enter__(self) -> "TideVec":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ---- Health ------------------------------------------------

    def health(self) -> dict:
        """Check server health."""
        resp = self._stub.Health(
            pb.HealthRequest(), timeout=self._timeout, metadata=self._meta())
        return {
            "status":        resp.status,
            "version":       resp.version,
            "collections":   resp.collections,
            "timestamp_ms":  resp.timestamp_ms,
            "gpu_available": resp.gpu_available,
            "tpu_available": resp.tpu_available,
        }

    def ping(self) -> bool:
        """Return True if server is reachable."""
        try:
            self.health()
            return True
        except grpc.RpcError:
            return False

    # ---- Collections -------------------------------------------

    def create_collection(
        self,
        name:         str,
        dim:          int,
        index_type:   str   = "tvindex",
        metric:       str   = "cosine",
        n_shards:     int   = 4,
        n_replicas:   int   = 1,
        write_quorum: int   = 1,
        half_life_ms: int   = HalfLife.ONE_MONTH,
        temporal_blend: float = 0.3,
    ) -> str:
        """Create a new collection."""
        metric_enum = {
            "cosine": pb.Metric.COSINE,
            "l2":     pb.Metric.L2,
            "dot":    pb.Metric.DOT,
        }.get(metric.lower(), pb.Metric.COSINE)

        req = pb.CreateCollectionRequest(
            name         = name,
            dim          = dim,
            index_type   = index_type,
            metric       = metric_enum,
            n_shards     = n_shards,
            n_replicas   = n_replicas,
            write_quorum = write_quorum,
            temporal     = pb.TemporalConfig(
                half_life_ms    = half_life_ms,
                temporal_blend  = temporal_blend,
            ),
        )
        resp = self._stub.CreateCollection(
            req, timeout=self._timeout, metadata=self._meta())
        return resp.name

    def drop_collection(self, name: str) -> bool:
        resp = self._stub.DropCollection(
            pb.DropCollectionRequest(name=name),
            timeout=self._timeout, metadata=self._meta())
        return resp.status == "ok"

    def get_collection(self, name: str) -> CollectionInfo:
        resp = self._stub.GetCollection(
            pb.GetCollectionRequest(name=name),
            timeout=self._timeout, metadata=self._meta())
        info = resp.info
        return CollectionInfo(
            name=info.name, n_vectors=info.n_vectors,
            n_shards=info.n_shards, dim=info.dim,
            index_type=info.index_type, metric=str(info.metric),
        )

    def list_collections(self) -> List[CollectionInfo]:
        resp = self._stub.ListCollections(
            pb.ListCollectionsRequest(),
            timeout=self._timeout, metadata=self._meta())
        return [
            CollectionInfo(
                name=c.name, n_vectors=c.n_vectors,
                n_shards=c.n_shards, dim=c.dim,
                index_type=c.index_type, metric=str(c.metric),
            ) for c in resp.collections
        ]

    def set_temporal(
        self, name: str,
        half_life_ms: int,
        temporal_blend: float = 0.3,
    ) -> None:
        """Update temporal decay settings for a collection."""
        self._stub.UpdateTemporal(
            pb.UpdateTemporalRequest(
                name=name,
                config=pb.TemporalConfig(
                    half_life_ms   = half_life_ms,
                    temporal_blend = temporal_blend,
                ),
            ),
            timeout=self._timeout, metadata=self._meta())

    # ---- Vectors -----------------------------------------------

    def upsert(
        self,
        collection: str,
        vectors:    List[Dict[str, Any]],
    ) -> int:
        """
        Upsert vectors into a collection.

        Each vector dict:
            {
                "id":        str,
                "embedding": List[float],
                "payload":   Dict[str, str]  (optional),
                "ttl_seconds": int           (optional),
                "edges": [{"target_id":str, "type":"CAUSES", "weight":0.9}]
            }
        """
        edge_type_map = {
            "CAUSES": pb.EdgeType.CAUSES, "CONTRADICTS": pb.EdgeType.CONTRADICTS,
            "UPDATES": pb.EdgeType.UPDATES, "RELATED_TO": pb.EdgeType.RELATED_TO,
            "ENTITY_OF": pb.EdgeType.ENTITY_OF, "SUPPORTS": pb.EdgeType.SUPPORTS,
        }

        pb_vecs = []
        for v in vectors:
            edges = [
                pb.CausalEdge(
                    target_id = e["target_id"],
                    type      = edge_type_map.get(e.get("type", "RELATED_TO"),
                                                  pb.EdgeType.RELATED_TO),
                    weight    = float(e.get("weight", 1.0)),
                )
                for e in v.get("edges", [])
            ]
            pb_vecs.append(pb.Vector(
                id          = v["id"],
                embedding   = v["embedding"],
                payload     = {str(k): str(val) for k, val in v.get("payload", {}).items()},
                ttl_seconds = v.get("ttl_seconds", 0),
                edges       = edges,
            ))

        resp = self._stub.Upsert(
            pb.UpsertRequest(collection=collection, vectors=pb_vecs),
            timeout=self._timeout, metadata=self._meta())
        return resp.inserted

    def delete(self, collection: str, ids: List[str]) -> int:
        resp = self._stub.Delete(
            pb.DeleteRequest(collection=collection, ids=ids),
            timeout=self._timeout, metadata=self._meta())
        return resp.deleted

    # ---- Search ------------------------------------------------

    def search(
        self,
        collection:     str,
        query_vector:   List[float],
        top_k:          int   = 10,
        temporal_blend: float = 0.3,
        mode:           str   = "vector_only",
        causal_hops:    int   = 1,
        filter:         str   = "",
        metric:         str   = "cosine",
        include_trace:  bool  = False,
        include_staleness_warnings: bool = True,
        device:         str   = "auto",
    ) -> SearchResponse:
        """
        Search for nearest neighbours with temporal scoring.

        Args:
            collection:      Collection name
            query_vector:    Query embedding (must match collection dim)
            top_k:           Number of results
            temporal_blend:  0.0 = pure vector, 1.0 = pure temporal
            mode:            "vector_only" | "causal_expand" |
                             "contradiction_check" | "entity_resolve"
            filter:          "key=value" payload filter
            include_trace:   Return RetrievalTrace metadata

        Returns:
            SearchResponse with .hits list of SearchHit
        """
        mode_map = {
            "vector_only":         pb.QueryMode.VECTOR_ONLY,
            "causal_expand":       pb.QueryMode.CAUSAL_EXPAND,
            "contradiction_check": pb.QueryMode.CONTRADICTION_CHECK,
            "entity_resolve":      pb.QueryMode.ENTITY_RESOLVE,
        }
        device_map = {
            "auto": pb.Device.AUTO, "cpu": pb.Device.CPU,
            "gpu":  pb.Device.GPU,  "tpu": pb.Device.TPU,
        }

        opts = pb.SearchOptions(
            top_k                      = top_k,
            temporal_blend             = temporal_blend,
            mode                       = mode_map.get(mode, pb.QueryMode.VECTOR_ONLY),
            causal_hops                = causal_hops,
            filter                     = filter,
            metric                     = pb.Metric.COSINE,
            include_trace              = include_trace,
            include_staleness_warnings = include_staleness_warnings,
            device_hint                = device_map.get(device.lower(), pb.Device.AUTO),
        )

        resp = self._stub.Search(
            pb.SearchRequest(
                collection = collection,
                vector     = query_vector,
                options    = opts,
            ),
            timeout=self._timeout, metadata=self._meta())

        hits = [SearchHit._from_proto(r) for r in resp.results]
        return SearchResponse(
            hits       = hits,
            count      = resp.count,
            latency_ms = resp.trace.latency_ms if include_trace else 0.0,
            query_id   = resp.trace.query_id   if include_trace else "",
            strategy   = resp.trace.strategy   if include_trace else "",
        )

    def batch_search(
        self,
        collection:    str,
        query_vectors: List[List[float]],
        top_k:         int   = 10,
        temporal_blend: float = 0.3,
        device:        str   = "auto",
    ) -> List[SearchResponse]:
        """Batch multiple queries in one GPU/TPU call."""
        device_map = {"auto": pb.Device.AUTO, "gpu": pb.Device.GPU, "tpu": pb.Device.TPU}
        requests = [
            pb.SearchRequest(
                collection = collection,
                vector     = q,
                options    = pb.SearchOptions(
                    top_k          = top_k,
                    temporal_blend = temporal_blend,
                    device_hint    = device_map.get(device.lower(), pb.Device.AUTO),
                ),
            ) for q in query_vectors
        ]
        resp = self._stub.BatchSearch(
            pb.BatchSearchRequest(collection=collection, queries=requests),
            timeout=self._timeout, metadata=self._meta())
        return [
            SearchResponse(
                hits  = [SearchHit._from_proto(r) for r in sr.results],
                count = sr.count,
            ) for sr in resp.responses
        ]

    def search_stream(
        self,
        collection:    str,
        query_vector:  List[float],
        top_k:         int = 10,
    ) -> Iterator[SearchHit]:
        """Server-streaming search — results arrive as they're found."""
        opts = pb.SearchOptions(top_k=top_k)
        for result in self._stub.SearchStream(
            pb.SearchRequest(
                collection=collection, vector=query_vector, options=opts),
            timeout=self._timeout, metadata=self._meta()):
            yield SearchHit._from_proto(result)

    # ---- Graph -------------------------------------------------

    def add_edges(
        self,
        collection: str,
        edges: List[Dict[str, Any]],
    ) -> int:
        """
        Add causal edges between vectors.

        edges: [{"src": "v1", "tgt": "v2", "type": "CAUSES", "weight": 0.9}]
        """
        edge_type_map = {
            "CAUSES": pb.EdgeType.CAUSES, "CONTRADICTS": pb.EdgeType.CONTRADICTS,
            "UPDATES": pb.EdgeType.UPDATES, "RELATED_TO": pb.EdgeType.RELATED_TO,
            "ENTITY_OF": pb.EdgeType.ENTITY_OF, "SUPPORTS": pb.EdgeType.SUPPORTS,
        }
        pb_edges = [
            pb.Edge(
                src    = e["src"],
                tgt    = e["tgt"],
                type   = edge_type_map.get(e.get("type", "RELATED_TO"),
                                           pb.EdgeType.RELATED_TO),
                weight = float(e.get("weight", 1.0)),
            ) for e in edges
        ]
        resp = self._stub.AddEdges(
            pb.AddEdgesRequest(collection=collection, edges=pb_edges),
            timeout=self._timeout, metadata=self._meta())
        return resp.added

    # ---- Convenience wrappers ----------------------------------

    def upsert_one(
        self,
        collection: str,
        id:         str,
        embedding:  List[float],
        payload:    Optional[Dict[str, str]] = None,
        ttl_seconds: int = 0,
    ) -> None:
        """Upsert a single vector."""
        self.upsert(collection, [{
            "id": id, "embedding": embedding,
            "payload": payload or {},
            "ttl_seconds": ttl_seconds,
        }])

    def search_one(
        self,
        collection:    str,
        query_vector:  List[float],
        top_k:         int = 1,
        temporal_blend: float = 0.3,
    ) -> Optional[SearchHit]:
        """Search and return the single nearest neighbour."""
        resp = self.search(collection, query_vector, top_k=top_k,
                           temporal_blend=temporal_blend)
        return resp.hits[0] if resp.hits else None


# ================================================================
# Async client (asyncio)
# ================================================================

class AsyncTideVec:
    """
    Async TideVec client for use with asyncio.

    Usage:
        async with AsyncTideVec("localhost:6399") as db:
            await db.upsert("docs", [...])
            results = await db.search("docs", query_vector=[...])
    """

    def __init__(self, host: str = "localhost:6399", api_key: str = ""):
        self._host    = host
        self._api_key = api_key
        self._channel = None
        self._stub    = None

    async def _ensure_connected(self):
        if self._stub is None:
            self._channel = grpc.aio.insecure_channel(self._host)
            self._stub    = pb_grpc.TideVecStub(self._channel)

    async def __aenter__(self) -> "AsyncTideVec":
        await self._ensure_connected()
        return self

    async def __aexit__(self, *_) -> None:
        if self._channel:
            await self._channel.close()

    def _meta(self):
        return [("x-api-key", self._api_key)] if self._api_key else []

    async def health(self) -> dict:
        await self._ensure_connected()
        resp = await self._stub.Health(pb.HealthRequest(), metadata=self._meta())
        return {"status": resp.status, "version": resp.version}

    async def upsert(self, collection: str, vectors: List[Dict]) -> int:
        await self._ensure_connected()
        # Reuse sync conversion logic
        sync = TideVec.__new__(TideVec)
        sync._stub    = self._stub
        sync._timeout = 30.0
        sync._api_key = self._api_key
        # For async: delegate to sync stub (grpc.aio supports await)
        pb_vecs = [pb.Vector(id=v["id"], embedding=v["embedding"],
                             payload={str(k): str(val)
                                      for k, val in v.get("payload",{}).items()})
                   for v in vectors]
        resp = await self._stub.Upsert(
            pb.UpsertRequest(collection=collection, vectors=pb_vecs),
            metadata=self._meta())
        return resp.inserted

    async def search(
        self,
        collection:    str,
        query_vector:  List[float],
        top_k:         int   = 10,
        temporal_blend: float = 0.3,
    ) -> SearchResponse:
        await self._ensure_connected()
        opts = pb.SearchOptions(top_k=top_k, temporal_blend=temporal_blend)
        resp = await self._stub.Search(
            pb.SearchRequest(collection=collection,
                             vector=query_vector, options=opts),
            metadata=self._meta())
        return SearchResponse(
            hits  = [SearchHit._from_proto(r) for r in resp.results],
            count = resp.count,
        )
