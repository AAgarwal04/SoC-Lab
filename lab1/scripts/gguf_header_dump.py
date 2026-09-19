#!/usr/bin/env python3
"""Minimal pure-Python GGUF header parser (no external deps).
Reads only the metadata KV section (+ as much of the tensor list as is
present in a possibly-truncated file) and prints it. Equivalent to the
metadata Netron shows for a .gguf file, since GGUF has no computation
graph of its own -- just tensors + hyperparameter key/value metadata.
"""
import struct
import sys

GGUF_TYPE_NAMES = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32", 5: "int32",
    6: "float32", 7: "bool", 8: "string", 9: "array", 10: "uint64", 11: "int64", 12: "float64",
}

SCALAR_FMT = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4),
    6: ("f", 4), 7: ("B", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8),
}


class Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read(self, n):
        if self.pos + n > len(self.data):
            raise EOFError(f"need {n} bytes at pos {self.pos}, only {len(self.data)} available")
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        n = self.u64()
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, vtype):
        if vtype == 8:
            return self.string()
        if vtype == 9:
            elem_type = self.u32()
            count = self.u64()
            return [self.value(elem_type) for _ in range(count)]
        fmt, size = SCALAR_FMT[vtype]
        return struct.unpack("<" + fmt, self.read(size))[0]


def parse(path, max_bytes=64 * 1024 * 1024):
    with open(path, "rb") as f:
        data = f.read(max_bytes)
    r = Reader(data)
    magic = r.read(4)
    if magic != b"GGUF":
        print(f"NOT A GGUF FILE (magic={magic!r})")
        return
    version = r.u32()
    tensor_count = r.u64()
    kv_count = r.u64()
    print(f"=== {path} ===")
    print(f"gguf version: {version}  tensor_count: {tensor_count}  kv_count: {kv_count}")
    kv = {}
    try:
        for _ in range(kv_count):
            key = r.string()
            vtype = r.u32()
            val = r.value(vtype)
            kv[key] = val
    except EOFError as e:
        print(f"  (metadata truncated early: {e})")

    interesting_prefixes = (
        "general.", ".architecture", ".block_count", ".embedding_length",
        ".feed_forward_length", ".attention.head_count", ".attention.head_count_kv",
        ".attention.layer_norm", ".attention.key_length", ".attention.value_length",
        ".context_length", ".vocab_size", ".rope", "tokenizer.ggml.model",
    )
    print("\n-- key hyperparameters --")
    for k in sorted(kv.keys()):
        if any(p in k for p in interesting_prefixes) or k.count(".") <= 1:
            v = kv[k]
            if isinstance(v, list):
                v = f"<array len={len(v)}>"
            print(f"  {k} = {v}")

    # tensor infos (best-effort; may be truncated if max_bytes too small)
    tensors = []
    try:
        for _ in range(tensor_count):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ttype = r.u32()
            offset = r.u64()
            tensors.append((name, dims, ttype))
    except EOFError:
        pass

    print(f"\n-- tensor list ({len(tensors)}/{tensor_count} parsed) --")
    if tensors:
        # Print unique name patterns (strip layer index) to show block structure
        seen_patterns = {}
        for name, dims, ttype in tensors:
            import re
            pattern = re.sub(r"\.\d+\.", ".N.", name)
            seen_patterns.setdefault(pattern, []).append((name, dims, ttype))
        for pattern in sorted(seen_patterns.keys()):
            example = seen_patterns[pattern][0]
            print(f"  {pattern}  (e.g. dims={example[1]}, type={example[2]}, x{len(seen_patterns[pattern])})")
    print()


if __name__ == "__main__":
    for path in sys.argv[1:]:
        parse(path)
