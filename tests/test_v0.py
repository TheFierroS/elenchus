"""V0's bucketing, checked on the fixtures.

Running V0 for real needs the corpus binaries on disk; the logic that decides
a pair's bucket does not, so it is tested here on cases.c's functions, whose
outcomes are known: a plain function completes and agrees, one calling an
unstubbed import buckets as import-missing, a non-terminating one as budget,
a by-value struct as signature-declined.
"""

import json
from pathlib import Path

import pytest

from elenchus.corpus.dwarf import ground_truth

pytest.importorskip("unicorn")

from elenchus.emulation.harness import Loader  # noqa: E402
from elenchus.emulation.stubs import BLOCK_MEMORY, resolver  # noqa: E402
from elenchus.emulation.v0 import Bucket, V0Report, bucket_pair  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "emulation"
BUF = 0x0000_2000_0000_0000


@pytest.fixture(scope="module")
def env():
    o0 = Loader(FIXTURES / "cases_O0.dll")
    o3 = Loader(FIXTURES / "cases_O3.dll")
    s0 = {f.name: f for f in ground_truth(FIXTURES / "cases_O0.dll")}
    s3 = {f.name: f for f in ground_truth(FIXTURES / "cases_O3.dll")}
    return o0, o3, s0, s3


def run_pair(env, name, resolver=None, inputs=None):
    o0, o3, s0, s3 = env
    return bucket_pair(o0, s0[name].address, o3, s3[name].address,
                       json.dumps(s3[name].abi), resolver=resolver,
                       inputs=inputs or [[7, 5, 2, 1], [1, 1, 1, 1]])


def test_a_plain_function_completes_and_agrees(env):
    result = run_pair(env, "add3")
    assert result.bucket == Bucket.COMPLETED
    assert result.agreement == Bucket.AGREED


def test_a_function_calling_an_unstubbed_import_buckets_as_missing(env):
    """With no resolver, duplicate's strlen call has no stub."""
    result = run_pair(env, "duplicate", inputs=[[BUF]])
    assert result.bucket == Bucket.IMPORT_MISSING
    assert result.detail in {"strlen", "malloc"}


def test_the_same_import_completes_with_the_family(env):
    """With every family, duplicate completes and agrees."""
    result = run_pair(env, "duplicate", resolver=resolver, inputs=[[BUF]])
    assert result.bucket == Bucket.COMPLETED
    assert result.agreement == Bucket.AGREED


def test_a_partial_family_still_reports_the_missing_import(env):
    """Block memory only: duplicate reaches strlen and reports it, so V0's
    histogram would count strlen as the import to serve next."""
    result = run_pair(env, "duplicate", resolver=BLOCK_MEMORY.get, inputs=[[BUF]])
    assert result.bucket == Bucket.IMPORT_MISSING
    assert result.detail in {"strlen", "malloc"}


def test_a_nonterminating_function_buckets_as_budget(env):
    o0, o3, s0, s3 = env
    result = bucket_pair(o0, s0["spin"].address, o3, s3["spin"].address,
                         json.dumps(s3["spin"].abi), inputs=[[5]], budget=50_000)
    assert result.bucket == Bucket.BUDGET


def test_a_by_value_struct_is_signature_declined(env):
    result = run_pair(env, "pair_sum")
    assert result.bucket == Bucket.SIGNATURE_DECLINED


def test_a_variadic_function_is_signature_declined(env):
    result = run_pair(env, "vsum")
    assert result.bucket == Bucket.SIGNATURE_DECLINED


# ------------------------------------------------------- the report


def test_the_report_counts_buckets_and_the_import_histogram(env):
    report = V0Report(layer="bare")
    report.add(run_pair(env, "add3"))
    report.add(run_pair(env, "mix64"))
    report.add(run_pair(env, "duplicate", inputs=[[BUF]]))
    report.add(run_pair(env, "duplicate", inputs=[[BUF]]))

    assert report.buckets[Bucket.COMPLETED] == 2
    assert report.buckets[Bucket.IMPORT_MISSING] == 2
    assert report.agreement[Bucket.AGREED] == 2
    # the histogram names what to stub next
    assert sum(report.imports_missing.values()) == 2
    assert report.total == 4


def test_the_report_keeps_disagreements_for_reading():
    """A completed pair that disagreed is a false refutation to investigate;
    the report keeps it, not just a count."""
    from elenchus.emulation.v0 import PairResult
    report = V0Report(layer="stubbed")
    report.add(PairResult("pkg", "f", Bucket.COMPLETED, Bucket.DISAGREED,
                          detail="return value differs"))
    assert len(report.disagreements) == 1
    assert report.disagreements[0].detail == "return value differs"


def test_a_fault_is_its_own_bucket_not_load_failed(env):
    """null_read dereferences address 0: a fault while running, distinct from
    a binary that could not be loaded. The separation is the point - a mixed
    bucket hid what was really happening."""
    o0, o3, s0, s3 = env
    result = bucket_pair(o0, s0["null_read"].address, o3, s3["null_read"].address,
                         json.dumps(s3["null_read"].abi), inputs=[[0]])
    assert result.bucket == Bucket.FAULT


def test_an_unsupported_instruction_buckets_on_the_exact_status():
    """The bucket comes from the harness Status enum, not from matching words
    in a message, so each cause is counted exactly."""
    from elenchus.emulation.compare import Comparison, InputResult, Verdict
    from elenchus.emulation.harness import Status
    from elenchus.emulation.v0 import _inconclusive_bucket

    comparison = Comparison(Verdict.INCONCLUSIVE, [
        InputResult(Verdict.INCONCLUSIVE, "Q unsupported instruction", {},
                    status=Status.UNSUPPORTED_INSTRUCTION),
    ])
    assert _inconclusive_bucket(comparison).bucket == Bucket.UNSUPPORTED


def test_a_run_with_nothing_to_compare_is_unjudged():
    """A pair that ran but whose only observation was garbage-dependent
    decided nothing - it is UNJUDGED, not a fault or a load failure."""
    from elenchus.emulation.compare import Comparison, InputResult, Verdict
    from elenchus.emulation.v0 import _inconclusive_bucket

    comparison = Comparison(Verdict.INCONCLUSIVE, [
        InputResult(Verdict.INCONCLUSIVE, "return depended on the fill", {}),
    ])
    assert _inconclusive_bucket(comparison).bucket == Bucket.UNJUDGED


def test_generated_inputs_let_a_pointer_function_run(env):
    """count_nonzero(const unsigned char *p, size_t n) takes a pointer. With
    generated inputs its pointer gets a real buffer, so it completes instead
    of faulting on a small-integer pointer - the whole reason for the
    generator."""
    o0, o3, s0, s3 = env
    result = bucket_pair(o0, s0["count_nonzero"].address,
                         o3, s3["count_nonzero"].address,
                         json.dumps(s3["count_nonzero"].abi))   # no inputs= : generated
    assert result.bucket == Bucket.COMPLETED
    assert result.agreement == Bucket.AGREED
