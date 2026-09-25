"""Loading of the challenge TSV files and writing of submission files."""
import os
import polars as pl

SCHEMA = {"entity_id": pl.Utf8, "business_name": pl.Utf8,
          "business_address": pl.Utf8, "country": pl.Utf8}


def find_dataset_dir(root):
    """Return the directory that contains train/ and test/ (handles Kaggle nesting)."""
    for cand in [root, os.path.join(root, "dataset"),
                 os.path.join(root, "student_resource", "dataset")]:
        if os.path.isdir(os.path.join(cand, "train")):
            return cand
    raise FileNotFoundError(f"no train/ folder under {root}")


def read_source(path):
    """Read one *_sourceN.tsv file. Quoting is disabled: fields may contain quotes."""
    df = pl.read_csv(path, separator="\t", quote_char=None, schema_overrides=SCHEMA,
                     missing_utf8_is_empty_string=False)
    return df.with_columns(pl.col("business_address").fill_null(""),
                           pl.col("business_name").fill_null(""))


def read_split(dataset_dir, split):
    """Return (s1, s23) for split in {'train','test'}; s23 has a `src` column (2 or 3)."""
    d = os.path.join(dataset_dir, split)
    s1 = read_source(os.path.join(d, f"{split}_source1.tsv"))
    s2 = read_source(os.path.join(d, f"{split}_source2.tsv")).with_columns(pl.lit(2, pl.Int8).alias("src"))
    s3 = read_source(os.path.join(d, f"{split}_source3.tsv")).with_columns(pl.lit(3, pl.Int8).alias("src"))
    return s1, pl.concat([s2, s3])


def read_ground_truth(dataset_dir):
    """Return exploded ground-truth edges (s1, m) and the list of all S1 ids."""
    gt = pl.read_csv(os.path.join(dataset_dir, "train", "train_ground_truth.tsv"), separator="\t",
                     quote_char=None, schema_overrides={"source1_entity_id": pl.Utf8,
                                                        "matched_entity_ids": pl.Utf8})
    edges = (gt.select(pl.col("source1_entity_id").alias("s1"),
                       pl.col("matched_entity_ids").str.split(",").alias("m"))
             .explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != "")))
    return edges


def write_id_lists(path, s1_ids, pairs, col):
    """Write one row per S1 id with a comma-joined list of matched ids from `pairs` (s1, m)."""
    agg = pairs.group_by("s1").agg(pl.col("m").unique(maintain_order=True).str.join(",").alias(col))
    out = (pl.DataFrame({"source1_entity_id": s1_ids})
           .join(agg.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left")
           .with_columns(pl.col(col).fill_null("")))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for a, b in out.iter_rows():
            f.write(f"{a}\t{b}\n")
