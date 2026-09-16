"""One-off, read-only: where a training run's time goes before its first step.

A 50-step smoke run logs a 25 s epoch but takes minutes to finish. The first
measurement showed that the cost was not reading the database: every
eligible_rows call, rows given or not, normalised all 56,747 listings again
for the dedup (31.6 s each, four per run). With content_key remembering each
listing's key, only the first dedup in a process should pay; the "again"
line shows whether it does.

Run from the repo root:  python experiments/load_timing.py
"""

import os
import time

T0 = time.perf_counter()
lap_at = [T0]
laps = []


def lap(name):
    now = time.perf_counter()
    laps.append((name, now - lap_at[0]))
    lap_at[0] = now


import torch  # noqa: E402

lap(f"import torch {torch.__version__}")

from elenchus.corpus.dataset import eligible_rows, load_splits  # noqa: E402
from elenchus.db import connect  # noqa: E402
from elenchus.encoder.data import load_examples  # noqa: E402
from elenchus.encoder.normalise import normalise  # noqa: E402
from elenchus.evaluation.store import dataset_fingerprint  # noqa: E402
from elenchus.evaluation.task import retrieval_task  # noqa: E402

lap("import elenchus")


def main():
    conn = connect(os.environ["ELENCHUS_DB"])
    lap("connect")

    rows = list(eligible_rows(conn))
    lap(f"eligible_rows, first call ({len(rows):,} rows)")

    rows = list(eligible_rows(conn))
    lap("eligible_rows again, same process")

    chars = sum(len(r["listing"] or "") for r in rows)
    lap(f"  (listing text in memory: {chars / 1e6:.0f} MB)")

    dataset_fingerprint(conn, rows)
    lap("dataset_fingerprint on rows already read")

    assignment = load_splits(conn)
    train_rows = [r for r in rows if assignment.get(r["package"]) == "train"]
    for r in train_rows:
        tuple(normalise(r["listing"]))
    lap(f"normalise train listings only ({len(train_rows):,})")

    load_examples(conn, "train", rows)
    lap("load_examples train on rows already read")

    retrieval_task(conn, split="val", rows=rows)
    lap("retrieval_task val on rows already read")

    total = time.perf_counter() - T0
    print(f"{'stage':56} {'seconds':>8}")
    for name, seconds in laps:
        print(f"{name:56} {seconds:8.1f}")
    print(f"{'total':56} {total:8.1f}")


if __name__ == "__main__":
    main()
