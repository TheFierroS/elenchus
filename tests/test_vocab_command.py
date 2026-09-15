"""Tests for `elenchus vocab`: built from train, saved with its provenance."""

import types

from elenchus.corpus.dataset import store_splits
from elenchus.encoder import cli as module
from elenchus.encoder.vocab import Vocab


def add_function(conn, package, name, mnemonics):
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO binaries (path, sha256, arch, imported_at) "
            "VALUES (?, ?, 'x', datetime('now'))", (f"/{package}", package))
        binary_id = conn.execute(
            "SELECT id FROM binaries WHERE sha256 = ?", (package,)).fetchone()[0]
        conn.execute(
            "INSERT OR IGNORE INTO corpus_binaries (binary_id, package, compiler, "
            "opt_level, stripped) VALUES (?, ?, 'gcc', 'O0', 1)", (binary_id, package))
        count = conn.execute(
            "SELECT COUNT(*) FROM functions WHERE binary_id = ?", (binary_id,)
        ).fetchone()[0]
        fid = conn.execute(
            "INSERT INTO functions (binary_id, address, size, raw_name) "
            "VALUES (?, ?, 32, 'F')", (binary_id, 0x1000 + count)).lastrowid
        conn.execute(
            "INSERT INTO ground_truth (binary_id, function_id, address, name, "
            "return_type, param_types, decl_file, decl_line) "
            "VALUES (?, ?, ?, ?, 'int', '[]', ?, 1)",
            (binary_id, fid, 0x1000 + count, name, f"src/{package}/{name}.c"))
        listing = "\n".join(f"{i:x}\t{m}\tRAX RBX\t" for i, m in enumerate(mnemonics))
        conn.execute("INSERT INTO function_code VALUES (?, ?, ?, 32, ?, ?)",
                     (fid, binary_id, len(mnemonics), f"h{fid}", listing))


def test_the_command_saves_a_train_only_vocabulary_and_reports_val(db, tmp_path,
                                                                   monkeypatch, capsys):
    for i in range(3):
        add_function(db, "trainpkg", f"t{i}", ["push", "xor"] * 5 + [f"pad{i}"])
    for i in range(2):
        add_function(db, "valpkg", f"v{i}", ["push", "imul"] * 5 + [f"q{i}"])
    store_splits(db, {"trainpkg": "train", "valpkg": "val"})
    monkeypatch.setattr(module, "connect", lambda _p: db)

    out = tmp_path / "models" / "vocab.json"
    code = module.cmd_vocab(types.SimpleNamespace(db=":memory:", out=str(out),
                                                  min_functions=2))
    printed = capsys.readouterr().out

    assert code == 0
    vocab = Vocab.load(out)
    assert "xor" in vocab.tokens and "imul" not in vocab.tokens
    assert vocab.source["dataset_fingerprint"]
    assert vocab.source["train_functions"] == 3
    assert "val coverage" in printed and "[UNK]" in printed
