"""Set up the retrieval task everything later is measured on.

The question is always the same one. Here is a function from a stripped
binary; somewhere in a pool of known functions is the same source function,
compiled differently. Can you find it?

Query and pool come from different optimisation levels on purpose, and from
the two furthest apart: -O0 is close to a literal translation of the source,
-O3 is the compiler having unrolled loops, folded constants, reordered
everything and inlined whatever it liked. Asking -O0 against -O0 would be
easy and would measure nothing anyone needs. The agent will never meet a
binary compiled the same way as its corpus.

Only one split is used at a time, and the split was assigned by package, so
a query and its answer always come from the same package and never from a
package the model trained on. That property was built in week 4; this module
relies on it rather than re-establishing it.

A source function identifies itself by (package, source file, name). Not by
name alone: two libraries both have an init, and two files in one library
both have a helper. Not by address, which changes with every recompilation.

One query can have several correct answers. Libraries are full of functions
that compile to exactly the same code - forwarding wrappers, one-line
accessors, and in a parser combinator library, dozens of them. Asking any
method to pick the intended one out of a set of identical candidates is
asking it to guess, and scoring the guess would cap every measurement below
what the task actually allows. So the answer is the equivalence class: every
pool function whose normalised form matches the true one.
"""

from collections import defaultdict
from dataclasses import dataclass

from elenchus.corpus.dataset import content_key, eligible_rows, load_splits


@dataclass(frozen=True)
class Sample:
    """One compiled function, with everything a baseline might look at."""

    function_id: int
    key: tuple
    name: str
    package: str
    opt_level: str
    listing: str
    n_instructions: int
    n_blocks: int

    @property
    def mnemonics(self):
        """The opcode sequence, in order."""
        return [
            line.split("\t")[1]
            for line in self.listing.splitlines()
            if line.count("\t") >= 3
        ]

    @property
    def imports(self):
        """The set of imported functions this one calls.

        Kept separate from the rest because it survives optimisation
        untouched: whatever the compiler does to a function, a call to
        memcpy is still a call to memcpy.
        """
        found = set()
        for line in self.listing.splitlines():
            parts = line.split("\t")
            if len(parts) >= 4 and parts[3].startswith("IMPORT:"):
                _, _, rest = parts[3].partition(":")
                _, _, name = rest.partition("!")
                found.add(name or rest)
        return found

    @property
    def n_calls(self):
        """How many calls this function makes, of any kind."""
        return sum(
            1 for line in self.listing.splitlines()
            if line.count("\t") >= 3 and line.split("\t")[3].startswith(
                ("CALL:", "IMPORT:", "TAIL:")
            )
        )


def block_counts(conn):
    """Return {function_id: basic block count} for the whole database."""
    return {
        row["function_id"]: row["n"]
        for row in conn.execute(
            "SELECT function_id, COUNT(*) AS n FROM basic_blocks "
            "WHERE function_id IS NOT NULL GROUP BY function_id"
        )
    }


def load_samples(conn, split, rows=None):
    """Return {optimisation level: [Sample]} for one split of the dataset."""
    assignment = load_splits(conn)
    blocks = block_counts(conn)

    by_level = {}
    for row in eligible_rows(conn, rows):
        if assignment.get(row["package"]) != split:
            continue

        sample = Sample(
            function_id=row["function_id"],
            key=(row["package"], row["decl_file"], row["name"]),
            name=row["name"],
            package=row["package"],
            opt_level=row["opt_level"],
            listing=row["listing"],
            n_instructions=row["n_instructions"],
            n_blocks=blocks.get(row["function_id"], 0),
        )
        by_level.setdefault(row["opt_level"], []).append(sample)

    return by_level


def equivalence_classes(pool):
    """Return {pool index: set of indices with identical normalised code}."""
    groups = defaultdict(set)
    for index, sample in enumerate(pool):
        groups[content_key(sample.listing)].add(index)

    return {
        index: groups[content_key(sample.listing)]
        for index, sample in enumerate(pool)
    }


def retrieval_task(conn, split="test", query_opt="O0", pool_opt="O3", rows=None):
    """Return (queries, pool, gold) for one cross-optimisation retrieval task.

    gold[i] is the set of pool indices that count as answering queries[i]:
    the true counterpart, plus anything compiled to identical code.

    Queries whose counterpart is missing are dropped rather than counted as
    failures. At -O3 the compiler inlines roughly a quarter of all functions,
    and those have no -O3 body to find; scoring them as misses would measure
    inlining, not retrieval.
    """
    by_level = load_samples(conn, split, rows)

    pool = by_level.get(pool_opt, [])
    where = {sample.key: index for index, sample in enumerate(pool)}
    classes = equivalence_classes(pool)

    queries = []
    gold = []
    for sample in by_level.get(query_opt, []):
        index = where.get(sample.key)
        if index is not None:
            queries.append(sample)
            gold.append(classes[index])

    return queries, pool, gold
