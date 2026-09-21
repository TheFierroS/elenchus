#!/bin/sh
# Rebuild the emulation test fixtures from cases.c.
#
# The DLLs are committed, like tests/fixtures/sample.dll, so the tests need
# neither MinGW nor a compiler: this script is the record of how they were
# made. Flags match the corpus build (elenchus/corpus/build.py): the same
# toolchain, -g for DWARF, and --exclude-all-symbols so the functions are
# found by their debug information and not by an export table.
#
# Rebuilding changes the bytes (PE timestamps) but not the code the tests
# depend on; the tests read every address and signature from the DWARF.
set -eu
cd "$(dirname "$0")"
for level in O0 O3; do
    x86_64-w64-mingw32-gcc -"$level" -g -shared -o "cases_$level.dll" cases.c \
        -Wl,--exclude-all-symbols
done
x86_64-w64-mingw32-gcc --version | head -n 1
