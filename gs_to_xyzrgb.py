#!/usr/bin/env python3
"""Convert a full 3DGS PLY (x,y,z + SH + opacity + scale/rot ...) to a fixed
x,y,z,red,green,blue PLY that GSviewer can display.

Color comes from the SH DC term:  c = 0.5 + C0 * f_dc,  C0 = 0.2820947917738781.
Optionally drops near-transparent gaussians:  --min-opacity SIGMOID_ALPHA
(opacity is stored as a logit; alpha = sigmoid(opacity)).
"""
import sys, mmap, struct, math, argparse

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--min-opacity", type=float, default=0.0,
                help="keep gaussians with sigmoid(opacity) >= this (0 = keep all)")
args = ap.parse_args()

C0 = 0.28209479177387814

with open(args.src, "rb") as f:
    header = bytearray()
    while True:
        line = f.readline()
        header += line
        if line.strip() == b"end_header":
            break
    data_start = f.tell()
    lines = header.decode("ascii", "replace").splitlines()
    assert lines[0].strip() == "ply"
    assert "binary_little_endian" in next(l for l in lines if l.startswith("format"))
    vcount = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])

    fsize = {"float":4,"float32":4,"double":8,"float64":8,
             "char":1,"uchar":1,"int8":1,"uint8":1,"short":2,"ushort":2,
             "int16":2,"uint16":2,"int":4,"uint":4,"int32":4,"uint32":4}
    off, cur = {}, 0
    for l in lines:
        t = l.split()
        if t[:1] == ["property"] and t[1] != "list":
            off[t[2]] = cur
            cur += fsize[t[1]]
    stride = cur
    for k in ("x","f_dc_0","f_dc_1","f_dc_2"):
        assert k in off, f"missing {k}"
    xo = off["x"]
    dc = (off["f_dc_0"], off["f_dc_1"], off["f_dc_2"])
    op = off.get("opacity")
    if args.min_opacity > 0 and op is None:
        print("warn: no opacity property; keeping all points"); args.min_opacity = 0.0

    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

f3 = struct.Struct("<3f")
f1 = struct.Struct("<f")

def to_byte(v):
    v = 0.5 + C0 * v
    if v <= 0: return 0
    if v >= 1: return 255
    return int(v * 255 + 0.5)

logit_thresh = None
if args.min_opacity > 0:
    a = min(max(args.min_opacity, 1e-6), 1 - 1e-6)
    logit_thresh = math.log(a / (1 - a))

buf = bytearray()
FLUSH = 1 << 22
kept = 0
bodytmp = "%s.body" % args.dst
with open(bodytmp, "wb") as out:
    base = data_start
    for i in range(vcount):
        if logit_thresh is not None:
            if f1.unpack_from(mm, base + op)[0] < logit_thresh:
                base += stride; continue
        r, g, b = f3.unpack_from(mm, base + dc[0])
        buf += mm[base + xo:base + xo + 12]
        buf += bytes((to_byte(r), to_byte(g), to_byte(b)))
        kept += 1
        base += stride
        if len(buf) >= FLUSH:
            out.write(buf); buf = bytearray()
    if buf:
        out.write(buf)
mm.close()

hdr = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    f"element vertex {kept}\n"
    "property float x\nproperty float y\nproperty float z\n"
    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    "end_header\n"
).encode("ascii")

import os, shutil
with open(args.dst, "wb") as out:
    out.write(hdr)
    with open(bodytmp, "rb") as body:
        shutil.copyfileobj(body, out, length=1 << 22)
os.remove(bodytmp)
print(f"kept {kept}/{vcount} gaussians -> {args.dst}")
