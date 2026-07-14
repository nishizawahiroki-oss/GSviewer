#!/usr/bin/env python3
"""Stream a list-property PLY down to a fixed x,y,z,red,green,blue PLY that GSviewer can open."""
import sys, mmap, struct, time

src = sys.argv[1]
dst = sys.argv[2]

with open(src, "rb") as f:
    # --- parse header ---
    header = bytearray()
    while True:
        line = f.readline()
        header += line
        if line.strip() == b"end_header":
            break
    data_start = f.tell()
    lines = header.decode("ascii", "replace").splitlines()
    assert lines[0].strip() == "ply", "not a ply"
    fmt = next(l for l in lines if l.startswith("format"))
    assert "binary_little_endian" in fmt, f"unsupported: {fmt}"
    vcount = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])

    scalar = {  # ply type -> struct size in bytes
        "char":1,"uchar":1,"int8":1,"uint8":1,
        "short":2,"ushort":2,"int16":2,"uint16":2,
        "int":4,"uint":4,"int32":4,"uint32":4,"float":4,"float32":4,
        "double":8,"float64":8,
    }
    # Build a per-property plan for the vertex element.
    props = []  # (kind, info)  kind: 'scalar' or 'list'
    in_vertex = False
    for l in lines:
        t = l.split()
        if t and t[0] == "element":
            in_vertex = (t[1] == "vertex")
            continue
        if in_vertex and t and t[0] == "property":
            if t[1] == "list":
                props.append(("list", (scalar[t[2]], scalar[t[3]], t[4])))
            else:
                props.append(("scalar", (scalar[t[1]], t[2])))

    # Offsets of the fields we keep, assuming they precede any list property.
    off = {}
    cur = 0
    fixed_prefix_ok = True
    for kind, info in props:
        if kind == "list":
            break
        size, name = info
        off[name] = (cur, size)
        cur += size
    for k in ("x","y","z","red","green","blue"):
        if k not in off:
            fixed_prefix_ok = False
    assert fixed_prefix_ok, "x/y/z/rgb are not all in the fixed prefix"
    xo = off["x"][0]; ro = off["red"][0]

    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

# --- write target header ---
out_header = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    f"element vertex {vcount}\n"
    "property float x\nproperty float y\nproperty float z\n"
    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    "end_header\n"
).encode("ascii")

pos = data_start
buf = bytearray()
FLUSH = 1 << 22  # ~4 MB
t0 = time.time()
with open(dst, "wb") as out:
    out.write(out_header)
    for i in range(vcount):
        base = pos
        p = pos
        for kind, info in props:
            if kind == "scalar":
                p += info[0]
            else:
                length_size, item_size, _ = info
                n = mm[p]  # length prefix (uint8 in this file)
                if length_size != 1:
                    n = int.from_bytes(mm[p:p+length_size], "little")
                p += length_size + n * item_size
        # keep xyz (12) + rgb (3)
        buf += mm[base+xo:base+xo+12]
        buf += mm[base+ro:base+ro+3]
        pos = p
        if len(buf) >= FLUSH:
            out.write(buf); buf = bytearray()
    if buf:
        out.write(buf)

mm.close()
print(f"done: {vcount} verts, consumed {pos} bytes, {time.time()-t0:.1f}s")
print(f"out: {dst}")
