#pragma once
// ================================================================
// Bm25Index — sparse keyword index (Okapi BM25) over one text field
//
// Documents are keyed by the owning index's internal ids. Replacing or
// removing a document bumps its version; postings carrying an older
// version are skipped at query time and dropped on the next rebuild.
// Document frequencies and lengths are kept exact.
//
// Tokenizer: lower-cased ASCII letters and digits; bytes >= 0x80 are
// kept so UTF-8 words stay whole; everything else separates tokens
// ("ERR-1234" -> "err", "1234").
// ================================================================

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tidevec {

class Bm25Index {
public:
    struct Params {
        float k1 = 1.2f;
        float b  = 0.75f;
    };

    Bm25Index() = default;
    explicit Bm25Index(Params p) : p_(p) {}

    static std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> out;
        std::string cur;
        for (unsigned char c : text) {
            if (std::isalnum(c) || c >= 0x80) {
                cur.push_back(static_cast<char>(c < 0x80 ? std::tolower(c) : c));
            } else if (!cur.empty()) {
                out.push_back(std::move(cur));
                cur.clear();
            }
        }
        if (!cur.empty()) out.push_back(std::move(cur));
        return out;
    }

    bool contains(uint64_t doc) const { return doc < docs_.size() && docs_[doc].live; }

    void add(uint64_t doc, const std::string& text) {
        if (contains(doc)) return;
        if (doc >= docs_.size()) docs_.resize(doc + 1);
        auto tokens = tokenize(text);
        std::unordered_map<std::string, uint32_t> tf;
        for (auto& t : tokens) ++tf[t];
        auto& d = docs_[doc];
        ++d.ver;
        d.len  = static_cast<uint32_t>(tokens.size());
        d.live = true;
        for (auto& [term, n] : tf) {
            postings_[term].push_back({doc, n, d.ver});
            ++df_[term];
        }
        ++n_docs_;
        total_len_ += d.len;
    }

    // `text` must be what was indexed for `doc`.
    void remove(uint64_t doc, const std::string& text) {
        if (!contains(doc)) return;
        auto& d = docs_[doc];
        std::unordered_set<std::string> seen;
        for (auto& t : tokenize(text)) {
            if (!seen.insert(t).second) continue;
            auto it = df_.find(t);
            if (it != df_.end() && it->second > 0 && --it->second == 0) df_.erase(it);
        }
        --n_docs_;
        total_len_ -= d.len;
        d.live = false;
        ++d.ver;
    }

    void clear() {
        postings_.clear();
        df_.clear();
        docs_.clear();
        n_docs_ = 0;
        total_len_ = 0;
    }

    std::size_t doc_count() const { return n_docs_; }

    // Top-k (score, doc) pairs, best first, among docs where ok(doc).
    template <typename Eligible>
    std::vector<std::pair<float, uint64_t>>
    search(const std::string& query, std::size_t k, Eligible&& ok) const {
        std::vector<std::pair<float, uint64_t>> out;
        if (n_docs_ == 0 || k == 0) return out;
        std::unordered_set<std::string> terms;
        for (auto& t : tokenize(query)) terms.insert(std::move(t));

        const double N = static_cast<double>(n_docs_);
        const double avgdl = std::max(1.0, static_cast<double>(total_len_) / N);
        std::unordered_map<uint64_t, float> acc;
        for (const auto& term : terms) {
            auto dit = df_.find(term);
            if (dit == df_.end()) continue;
            const double df = dit->second;
            const double idf = std::log(1.0 + (N - df + 0.5) / (df + 0.5));
            for (const auto& post : postings_.at(term)) {
                const auto& d = docs_[post.doc];
                if (!d.live || d.ver != post.ver) continue;
                const double tf = post.tf;
                const double norm = tf + p_.k1 * (1.0 - p_.b + p_.b * d.len / avgdl);
                acc[post.doc] += static_cast<float>(idf * tf * (p_.k1 + 1.0) / norm);
            }
        }
        out.reserve(acc.size());
        for (const auto& [doc, s] : acc)
            if (ok(doc)) out.emplace_back(s, doc);
        auto cmp = [](const auto& a, const auto& b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        };
        if (out.size() > k) {
            std::partial_sort(out.begin(), out.begin() + k, out.end(), cmp);
            out.resize(k);
        } else {
            std::sort(out.begin(), out.end(), cmp);
        }
        return out;
    }

private:
    struct Posting {
        uint64_t doc;
        uint32_t tf;
        uint32_t ver;
    };
    struct Doc {
        uint32_t len  = 0;
        uint32_t ver  = 0;
        bool     live = false;
    };

    Params p_;
    std::unordered_map<std::string, std::vector<Posting>> postings_;
    std::unordered_map<std::string, uint32_t> df_;
    std::vector<Doc> docs_;
    std::size_t n_docs_   = 0;
    uint64_t    total_len_ = 0;
};

} // namespace tidevec
