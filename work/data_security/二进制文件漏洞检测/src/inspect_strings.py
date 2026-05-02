from __future__ import annotations

import re
import sys

from common import BINARY_DIR


WORDS = [
    "error",
    "return",
    "check",
    "stack",
    "heap",
    "buffer",
    "under",
    "overflow",
    "integer",
    "alloc",
    "read",
    "write",
    "char",
    "int",
    "malloc",
    "free",
    "null",
    "uninit",
    "negative",
    "large",
    "socket",
    "file",
    "source",
    "sink",
    "bad",
    "good",
]


def main() -> None:
    for binary_id in sys.argv[1:]:
        raw = (BINARY_DIR / f"{binary_id}.exe").read_bytes()
        strings = [item.decode("latin1", "ignore") for item in re.findall(rb"[ -~]{4,220}", raw)]
        hits = [item for item in strings if any(word in item.lower() for word in WORDS)]
        print(f"\n== {binary_id} hits={len(hits)} ==")
        for item in hits[:120]:
            print(item)


if __name__ == "__main__":
    main()
