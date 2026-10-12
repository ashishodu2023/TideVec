// ================================================================
// tidevec_native — in-process TideVec engine for Python
//
// Wraps CollectionRegistry (the same storage stack the REST server
// uses: WAL, shards, segment store, supersession) so `tidevec.Local`
// behaves like a server without running one. JSON shapes match the
// REST API exactly so the pure-Python SDK can reuse its parsers.
// ================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <tidevec/api/collection_registry.hpp>

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace tidevec;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

class Database {
public:
    Database(const std::string& data_dir, bool segment_store) {
        RegistryConfig cfg;
        cfg.data_dir          = data_dir;
        cfg.use_segment_store = segment_store;
        registry_ = std::make_unique<CollectionRegistry>(cfg);
        registry_->load_and_recover();
    }

    void create_collection(const std::string& params_json) {
        auto j = json::parse(params_json);
        CollectionRegistry::CreateParams p;
        p.name       = j.at("name").get<std::string>();
        p.dim        = j.value("dim", 768UL);
        p.index_type = j.value("index_type", std::string("tvindex"));
        p.metric     = j.value("metric", std::string("cosine"));
        p.n_shards   = j.value("n_shards", 1UL);
        if (p.n_shards == 0) p.n_shards = 1;
        if (j.contains("temporal"))     p.temporal     = temporal_cfg_from_json(j["temporal"]);
        if (j.contains("agent"))        p.agent        = agent_cfg_from_json(j["agent"]);
        if (j.contains("supersession")) p.supersession = supersession_cfg_from_json(j["supersession"]);
        if (j.contains("text_field") && !j["text_field"].is_null())
            p.text_field = j["text_field"].get<std::string>();
        p.payload_index = j.value("payload_index", true);
        p.compaction_ratio = j.value("compaction_ratio", 1.0);
        if (!(p.compaction_ratio > 0))
            throw std::invalid_argument("compaction_ratio must be > 0");
        registry_->create(p);
    }

    // Same response as POST /v1/collections/{name}/compact.
    std::string compact(const std::string& name, bool wait) {
        auto& col = registry_->get(name);
        CompactionStatus s;
        {
            py::gil_scoped_release release;
            s = col.compact(wait);
        }
        return compaction_status_to_json(s).dump();
    }

    std::string compaction_status(const std::string& name) {
        return compaction_status_to_json(registry_->get(name).compaction_status()).dump();
    }

    // Same shape as GET /v1/collections/{name}.
    std::string describe(const std::string& name) {
        auto& col = registry_->get(name);
        const std::string real = registry_->resolve(name);
        json info = {
            {"name",         real},
            {"n_vectors",    col.total_vectors()},
            {"n_shards",     col.n_shards()},
            {"backend",      "embedded"},
            {"supersession", supersession_cfg_to_json(col.supersession_config())},
            {"text_field",   col.text_field()},
            {"aliases",      registry_->aliases_of(real)},
        };
        if (auto p = registry_->params(real)) {
            info["dim"]           = p->dim;
            info["index_type"]    = p->index_type;
            info["metric"]        = p->metric;
            info["payload_index"] = p->payload_index;
        }
        if (real != name) info["alias"] = name;
        return info.dump();
    }

    bool drop_collection(const std::string& name) { return registry_->drop(name); }
    bool has_collection(const std::string& name) const { return registry_->exists(name); }

    std::string list_collections() const {
        json arr = json::array();
        for (const auto& c : registry_->list())
            arr.push_back({{"name", c.name}, {"n_vectors", c.n_vectors},
                           {"n_shards", c.n_shards}, {"dim", c.dim},
                           {"index_type", c.index_type}, {"metric", c.metric},
                           {"backend", "embedded"}});
        return arr.dump();
    }

    std::size_t count(const std::string& name) {
        return registry_->get(name).total_vectors();
    }

    // `meta_json` is a JSON array of REST-style vector objects without
    // "embedding"; embeddings arrive as an (n, dim) float32 array.
    std::size_t upsert(const std::string& name, FloatArray embeddings,
                       const std::string& meta_json) {
        auto meta = json::parse(meta_json);
        auto buf  = embeddings.request();
        if (buf.ndim != 2)
            throw std::invalid_argument("embeddings must be a 2-D array");
        auto n   = static_cast<std::size_t>(buf.shape[0]);
        auto dim = static_cast<std::size_t>(buf.shape[1]);
        if (!meta.is_array() || meta.size() != n)
            throw std::invalid_argument("metadata count does not match embeddings");

        auto& col = registry_->get(name);
        const float* data = static_cast<const float*>(buf.ptr);
        std::vector<CortexVector> vecs;
        vecs.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            auto& m = meta[i];
            m["embedding"] = json::array();
            CortexVector v = vector_from_json(m);
            v.embedding.assign(data + i * dim, data + (i + 1) * dim);
            vecs.push_back(std::move(v));
        }

        py::gil_scoped_release release;
        for (const auto& v : vecs) col.upsert(v);
        return vecs.size();
    }

    std::size_t remove(const std::string& name, const std::vector<std::string>& ids) {
        auto& col = registry_->get(name);
        py::gil_scoped_release release;
        std::size_t n = 0;
        for (const auto& id : ids) n += col.remove(id) ? 1 : 0;
        return n;
    }

    std::size_t remove_where(const std::string& name, const std::string& filter) {
        auto& col = registry_->get(name);
        py::gil_scoped_release release;
        return col.remove_where(filter);
    }

    // `filter_json`: a JSON filter object (or legacy "key=value" string).
    std::size_t remove_where_json(const std::string& name, const std::string& filter_json) {
        auto f = filter_from_json(json::parse(filter_json));
        if (!f) throw std::invalid_argument("filter must not be empty");
        auto& col = registry_->get(name);
        py::gil_scoped_release release;
        return col.remove_where(f);
    }

    // Same shape as POST /v1/collections/{name}/get.
    std::string get(const std::string& name, const std::vector<std::string>& ids,
                    bool with_vector, bool with_payload) {
        auto& col = registry_->get(name);
        json vecs = json::array(), missing = json::array();
        for (const auto& id : ids) {
            if (auto v = col.get(id, with_vector))
                vecs.push_back(stored_vector_to_json(*v, with_vector, with_payload));
            else
                missing.push_back(id);
        }
        return json{{"vectors", vecs}, {"missing", missing}}.dump();
    }

    // Same body / response as POST /v1/collections/{name}/scroll.
    std::string scroll(const std::string& name, const std::string& body_json) {
        auto j = json::parse(body_json);
        QueryOptions opts;
        apply_filter_fields(j, opts);
        bool with_vector  = j.value("with_vector", false);
        bool with_payload = j.value("with_payload", true);
        std::string cursor = j.contains("cursor") && !j["cursor"].is_null()
            ? j["cursor"].get<std::string>() : "";
        std::size_t limit = j.value("limit", std::size_t{100});
        if (limit == 0) throw std::invalid_argument("limit must be positive");
        auto& col = registry_->get(name);
        ManagedCollection::ScrollPage page;
        {
            py::gil_scoped_release release;
            page = col.scroll(cursor, limit, opts, with_vector);
        }
        json vecs = json::array();
        for (const auto& v : page.vectors)
            vecs.push_back(stored_vector_to_json(v, with_vector, with_payload));
        return json{{"vectors", vecs}, {"count", page.vectors.size()},
                    {"next_cursor", page.next_cursor ? json(*page.next_cursor)
                                                     : json(nullptr)}}.dump();
    }

    // Body as POST /v1/collections/{name}/count ({"filter":..,"as_of":..}).
    std::size_t count_where(const std::string& name, const std::string& body_json) {
        QueryOptions opts;
        apply_filter_fields(json::parse(body_json), opts);
        auto& col = registry_->get(name);
        py::gil_scoped_release release;
        return col.count(opts);
    }

    std::string set_alias(const std::string& alias, const std::string& collection) {
        return registry_->set_alias(alias, collection);
    }
    bool delete_alias(const std::string& alias) { return registry_->delete_alias(alias); }
    std::string list_aliases() const {
        json arr = json::array();
        for (const auto& [a, c] : registry_->aliases())
            arr.push_back({{"alias", a}, {"collection", c}});
        return arr.dump();
    }

    std::size_t add_edges(const std::string& name, const std::string& edges_json) {
        auto edges = json::parse(edges_json);
        auto& col = registry_->get(name);
        std::size_t n = 0;
        for (const auto& e : edges) {
            col.add_edge(e.at("src").get<std::string>(), e.at("tgt").get<std::string>(),
                         edge_type_from_str(e.at("type").get<std::string>()),
                         e.value("weight", 1.0f));
            ++n;
        }
        return n;
    }

    // Returns the REST response's `data` object as a JSON string. An
    // empty query array means keyword-only search (opts must carry "text").
    std::string search(const std::string& name, FloatArray query,
                       const std::string& opts_json) {
        auto buf = query.request();
        const float* q = static_cast<const float*>(buf.ptr);
        std::vector<float> qv(q, q + buf.size);
        QueryOptions opts = query_opts_from_json(json::parse(opts_json));
        if (qv.empty() && opts.text.empty())
            throw std::invalid_argument("search needs a query vector or text");

        auto& col = registry_->get(name);
        RetrievalTrace trace;
        std::vector<SearchResult> results;
        {
            py::gil_scoped_release release;
            results = col.search(qv, opts, opts.include_trace ? &trace : nullptr);
        }

        json out;
        out["results"] = json::array();
        for (const auto& r : results) out["results"].push_back(result_to_json(r));
        out["count"] = results.size();
        if (opts.include_trace) out["trace"] = trace_to_json(trace);
        return out.dump();
    }

    void set_temporal(const std::string& name, const std::string& cfg_json) {
        registry_->get(name).set_temporal_config(
            temporal_cfg_from_json(json::parse(cfg_json)));
    }

private:
    std::unique_ptr<CollectionRegistry> registry_;
};

} // namespace

PYBIND11_MODULE(tidevec_native, m) {
    m.doc() = "In-process TideVec engine (used by tidevec.Local)";
    py::class_<Database>(m, "Database")
        .def(py::init<const std::string&, bool>(),
             py::arg("data_dir"), py::arg("segment_store") = true)
        .def("create_collection", &Database::create_collection)
        .def("drop_collection",   &Database::drop_collection)
        .def("has_collection",    &Database::has_collection)
        .def("list_collections",  &Database::list_collections)
        .def("count",             &Database::count)
        .def("upsert",            &Database::upsert)
        .def("remove",            &Database::remove)
        .def("remove_where",      &Database::remove_where)
        .def("remove_where_json", &Database::remove_where_json)
        .def("add_edges",         &Database::add_edges)
        .def("search",            &Database::search)
        .def("set_temporal",      &Database::set_temporal)
        .def("describe",          &Database::describe)
        .def("get",               &Database::get,
             py::arg("name"), py::arg("ids"), py::arg("with_vector") = false,
             py::arg("with_payload") = true)
        .def("scroll",            &Database::scroll)
        .def("count_where",       &Database::count_where)
        .def("set_alias",         &Database::set_alias)
        .def("delete_alias",      &Database::delete_alias)
        .def("list_aliases",      &Database::list_aliases)
        .def("compact",           &Database::compact,
             py::arg("name"), py::arg("wait") = false)
        .def("compaction_status", &Database::compaction_status);
}
