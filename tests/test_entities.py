"""Tests for the entity registry and the CHECK constraint it generates."""

from elenchus.entities import ENTITY_TABLES, entity_kind_check


def test_check_is_sorted_and_quoted():
    """The constraint lists every entity name, sorted, single-quoted."""
    constraint = entity_kind_check()

    for name in ENTITY_TABLES:
        assert f"'{name}'" in constraint

    names = sorted(ENTITY_TABLES)
    expected = ", ".join(f"'{name}'" for name in names)
    assert expected in constraint


def test_check_rejects_bad_names(monkeypatch):
    """A name that could break the SQL is refused, not silently emitted."""
    import pytest

    from elenchus import entities

    bad = dict(entities.ENTITY_TABLES)
    bad["data'type"] = "x"
    monkeypatch.setattr(entities, "ENTITY_TABLES", bad)

    with pytest.raises(ValueError):
        entities.entity_kind_check()
