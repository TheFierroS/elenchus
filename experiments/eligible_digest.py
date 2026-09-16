"""One-off, read-only: a digest of exactly which rows the dataset is made of.

Run before and after a change that should not alter the dataset; the two
digests must be identical. Covers every eligible row's id, package, level,
identity and normalised-content key, so a changed dedup, a dropped row or a
renamed file all change it.

Run from the repo root:  python experiments/eligible_digest.py
"""

import hashlib
import os
import time

from elenchus.corpus.dataset import content_key, eligible_rows
from elenchus.db import connect
from elenchus.evaluation.store import dataset_fingerprint


def digest(conn):
    rows = eligible_rows(conn)
    material = sorted(
        (row["function_id"], row["package"], row["opt_level"], row["name"],
         row["decl_file"], content_key(row["listing"]))
        for row in rows)
    text = "\n".join("\t".join(str(field) for field in entry) for entry in material)
    return len(rows), hashlib.sha256(text.encode()).hexdigest(), dataset_fingerprint(conn)


def main():
    started = time.perf_counter()
    count, sha, fingerprint = digest(connect(os.environ["ELENCHUS_DB"]))
    print(f"eligible rows : {count:,}")
    print(f"row digest    : {sha}")
    print(f"fingerprint   : {fingerprint}")
    print(f"seconds       : {time.perf_counter() - started:.1f}")


if __name__ == "__main__":
    main()
