# tidevec-embedded

The TideVec engine compiled as a Python extension, so you can use TideVec
without running a server. You normally don't install this directly:

```bash
pip install "tidevec[embedded]"
```

```python
import tidevec

db = tidevec.Local("./tidevec_data")   # same API as tidevec.TideVec(...)
db.create_collection("docs", dim=768, supersession=True)
```

Data is persisted under the given directory (write-ahead log + segments)
and recovered automatically on the next `tidevec.Local(...)` call.

Building from source requires a C++20 compiler (GCC 12+ / Clang 15+).
