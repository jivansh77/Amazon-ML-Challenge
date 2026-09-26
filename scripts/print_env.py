"""Print the versions of every package the pipeline uses (for requirements.txt)."""
import importlib.metadata as md, platform, sys
print("python", platform.python_version())
for p in ["polars", "pyarrow", "numpy", "scipy", "scikit-learn", "lightgbm", "xgboost", "rapidfuzz", "sparse_dot_topn",
          "indic_transliteration", "torch", "transformers", "sentence-transformers", "tokenizers"]:
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        print(p, "MISSING")
