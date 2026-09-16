"""The command line chooses the CUDA allocator before anything touches CUDA.

Measured on the val pool and queries: 2.40 GiB of tensors reserved 6.52 GiB
with PyTorch's default allocator and 2.67 GiB with expandable segments, the
vectors identical. PyTorch reads the setting when CUDA first allocates, so
it only works if it is in the environment before a command imports torch.
"""

import os
import subprocess
import sys
import types

from elenchus import cli


def test_the_allocator_is_set_when_the_user_has_not_chosen_one():
    environ = {}
    assert cli.default_cuda_allocator(environ) == "expandable_segments:True"
    assert environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_a_users_own_allocator_setting_is_kept():
    environ = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}
    assert cli.default_cuda_allocator(environ) == "max_split_size_mb:128"
    assert environ == {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}


def test_main_sets_it_before_the_command_runs(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    seen = []

    def command(_args):
        seen.append(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
        return 0

    args = types.SimpleNamespace(db="x.db", func=command)
    monkeypatch.setattr(cli, "build_parser",
                        lambda: types.SimpleNamespace(parse_args=lambda _argv: args))
    assert cli.main(["--db", "x.db"]) == 0
    assert seen == ["expandable_segments:True"]


def test_loading_the_command_line_does_not_load_torch():
    """If it did, the allocator would be chosen before main() could set it."""
    environ = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, elenchus.cli; print('torch' in sys.modules)"],
        capture_output=True, text=True, env=environ, check=True)
    assert result.stdout.strip() == "False"
