"""Shared fixtures for the test suite."""

import pytest

from elenchus.db import connect


@pytest.fixture
def db(tmp_path):
    """A fresh, migrated database in a temp directory, torn down after."""
    path = tmp_path / "test.db"
    conn = connect(path)
    yield conn
    conn.close()
