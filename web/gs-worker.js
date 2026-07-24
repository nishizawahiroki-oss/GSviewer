"use strict";

// Streaming parser + depth sorter for full 3DGS PLY files (x/y/z + f_dc + opacity
// + scale + rot).  Packs each gaussian into a 2-texel RGBA32UI row for the EWA
// splatting shader in gs-viewer.js, then serves depth-sort requests over the
// retained positions.

const SH_C0 = 0.28209479177387814;
const TEX_WIDTH = 2048; // 1024 splats per row, 2 texels per splat

const FLOAT_TYPES = new Set(["float", "float32"]);
const PLY_TYPE_SIZES = {
  char: 1, int8: 1, uchar: 1, uint8: 1,
  short: 2, int16: 2, ushort: 2, uint16: 2,
  int: 4, int32: 4, uint: 4, uint32: 4,
  float: 4, float32: 4, double: 8, float64: 8,
};

let vertexCount = 0;
let positions = null; // Float32Array(3n), retained for sorting

const _halfFloat = new Float32Array(1);
const _halfInt = new Int32Array(_halfFloat.buffer);

function floatToHalf(value) {
  _halfFloat[0] = value;
  const f = _halfInt[0];
  const sign = (f >> 31) & 0x0001;
  const exp = (f >> 23) & 0x00ff;
  let frac = f & 0x007fffff;
  let newExp;
  if (exp === 0) {
    newExp = 0;
  } else if (exp < 113) {
    newExp = 0;
    frac |= 0x00800000;
    frac >>= 113 - exp;
    if (frac & 0x01000000) {
      newExp = 1;
      frac = 0;
    }
  } else if (exp < 142) {
    newExp = exp - 112;
  } else {
    newExp = 31;
    frac = 0;
  }
  return (sign << 15) | (newExp << 10) | (frac >> 13);
}

function packHalf2x16(x, y) {
  return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
}

function fail(message) {
  throw new Error(message);
}

function clampByte(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function findHeaderEnd(bytes, length) {
  const marker = [101, 110, 100, 95, 104, 101, 97, 100, 101, 114]; // "end_header"
  outer: for (let i = 0; i <= length - marker.length; i += 1) {
    if (i > 0 && bytes[i - 1] !== 10) continue;
    for (let j = 0; j < marker.length; j += 1) {
      if (bytes[i + j] !== marker[j]) continue outer;
    }
    const end = i + marker.length;
    if (bytes[end] === 10) return end + 1;
    if (bytes[end] === 13 && bytes[end + 1] === 10) return end + 2;
  }
  return -1;
}

function parseHeaderText(text) {
  const lines = text.split(/\r?\n/);
  if (lines[0]?.trim() !== "ply") fail("The selected file is not a PLY.");

  let format = null;
  let currentElement = null;
  let count = null;
  let stride = 0;
  let firstElement = null;
  const byName = {};

  for (const rawLine of lines.slice(1)) {
    const line = rawLine.trim();
    if (!line || line === "end_header") continue;
    const parts = line.split(/\s+/);
    if (parts[0] === "format") {
      format = `${parts[1]} ${parts[2]}`;
    } else if (parts[0] === "element" && parts.length === 3) {
      currentElement = parts[1];
      if (firstElement === null) firstElement = currentElement;
      if (currentElement === "vertex") {
        count = Number.parseInt(parts[2], 10);
        if (!Number.isSafeInteger(count) || count <= 0) fail(`Invalid vertex count: ${line}`);
      }
    } else if (parts[0] === "property" && currentElement === "vertex") {
      if (parts[1] === "list") fail("List properties are not supported.");
      if (parts.length !== 3 || !(parts[1] in PLY_TYPE_SIZES)) fail(`Unsupported property: ${line}`);
      byName[parts[2]] = { type: parts[1], offset: stride };
      stride += PLY_TYPE_SIZES[parts[1]];
    }
  }

  if (format !== "binary_little_endian 1.0") {
    fail("Only binary_little_endian PLY 1.0 is supported.");
  }
  if (count === null) fail("PLY has no vertex element.");
  if (firstElement !== "vertex") {
    fail("The vertex element must be the first PLY element.");
  }

  const requireFloat = (name) => {
    const property = byName[name];
    if (!property) fail(`3DGS PLY is missing the '${name}' property. 点群表示 (/) を使用してください。`);
    if (!FLOAT_TYPES.has(property.type)) fail(`Property '${name}' must be float32.`);
    return property.offset;
  };

  // 2D Gaussian splats (surfels) are flat and store only scale_0/scale_1; the
  // third axis is the disk normal.  scale[2] === -1 flags that missing scale so
  // processRecords can render a near-flat disk instead of failing.
  const scale2 = byName.scale_2;
  const plan = {
    count,
    stride,
    x: requireFloat("x"),
    y: requireFloat("y"),
    z: requireFloat("z"),
    scale: [
      requireFloat("scale_0"),
      requireFloat("scale_1"),
      scale2 && FLOAT_TYPES.has(scale2.type) ? scale2.offset : -1,
    ],
    rot: [requireFloat("rot_0"), requireFloat("rot_1"), requireFloat("rot_2"), requireFloat("rot_3")],
    opacity: byName.opacity && FLOAT_TYPES.has(byName.opacity.type) ? byName.opacity.offset : -1,
    filter3d: byName.filter_3D && FLOAT_TYPES.has(byName.filter_3D.type) ? byName.filter_3D.offset : -1,
    dc: null,
    rgb: null,
  };

  if (byName.f_dc_0 && byName.f_dc_1 && byName.f_dc_2) {
    plan.dc = [requireFloat("f_dc_0"), requireFloat("f_dc_1"), requireFloat("f_dc_2")];
  } else if (byName.red && byName.green && byName.blue
    && [byName.red, byName.green, byName.blue].every((p) => p.type === "uchar" || p.type === "uint8")) {
    plan.rgb = [byName.red.offset, byName.green.offset, byName.blue.offset];
  } else {
    fail("PLY has neither f_dc_0..2 nor uchar RGB color properties.");
  }
  return plan;
}

function emptyBounds() {
  return {
    min: [Infinity, Infinity, Infinity],
    max: [-Infinity, -Infinity, -Infinity],
  };
}

async function parseStream(stream, totalBytes, fileName) {
  const reader = stream.getReader();
  let pending = new Uint8Array(1 << 21);
  let pendingLength = 0;
  let plan = null;
  let texdata = null;
  let texFloat = null;
  let texHeight = 0;
  let index = 0;
  let bytesSeen = 0;
  let sampleStep = 1;
  const samples = [[], [], []];
  const allBounds = emptyBounds();
  let lastProgress = 0;

  const appendChunk = (chunk) => {
    if (pendingLength + chunk.length > pending.length) {
      const grown = new Uint8Array(Math.max(pending.length * 2, pendingLength + chunk.length));
      grown.set(pending.subarray(0, pendingLength));
      pending = grown;
    }
    pending.set(chunk, pendingLength);
    pendingLength += chunk.length;
  };

  const processRecords = () => {
    const stride = plan.stride;
    const available = Math.min(Math.floor(pendingLength / stride), plan.count - index);
    if (available <= 0) return;
    const view = new DataView(pending.buffer, 0, available * stride);
    const hasFilter = plan.filter3d >= 0;
    const hasOpacity = plan.opacity >= 0;
    const useDc = plan.dc !== null;

    for (let record = 0; record < available; record += 1) {
      const base = record * stride;
      let x = view.getFloat32(base + plan.x, true);
      let y = view.getFloat32(base + plan.y, true);
      let z = view.getFloat32(base + plan.z, true);
      let degenerate = false;
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
        // Park non-finite gaussians at the origin with zero alpha so they
        // cannot poison the bounds, the sort, or the shader.
        x = 0;
        y = 0;
        z = 0;
        degenerate = true;
      }
      const offset3 = index * 3;
      positions[offset3] = x;
      positions[offset3 + 1] = y;
      positions[offset3 + 2] = z;

      if (degenerate) {
        const texBaseSkip = index * 8;
        texFloat[texBaseSkip] = 0;
        texFloat[texBaseSkip + 1] = 0;
        texFloat[texBaseSkip + 2] = 0;
        texdata[texBaseSkip + 3] = 0;
        texdata[texBaseSkip + 4] = 0;
        texdata[texBaseSkip + 5] = 0;
        texdata[texBaseSkip + 6] = 0;
        texdata[texBaseSkip + 7] = 0;
        index += 1;
        continue;
      }

      if (x < allBounds.min[0]) allBounds.min[0] = x;
      if (y < allBounds.min[1]) allBounds.min[1] = y;
      if (z < allBounds.min[2]) allBounds.min[2] = z;
      if (x > allBounds.max[0]) allBounds.max[0] = x;
      if (y > allBounds.max[1]) allBounds.max[1] = y;
      if (z > allBounds.max[2]) allBounds.max[2] = z;
      if (index % sampleStep === 0) {
        samples[0].push(x);
        samples[1].push(y);
        samples[2].push(z);
      }

      // Color: SH DC term or direct RGB.
      let red;
      let green;
      let blue;
      if (useDc) {
        red = clampByte((0.5 + SH_C0 * view.getFloat32(base + plan.dc[0], true)) * 255);
        green = clampByte((0.5 + SH_C0 * view.getFloat32(base + plan.dc[1], true)) * 255);
        blue = clampByte((0.5 + SH_C0 * view.getFloat32(base + plan.dc[2], true)) * 255);
      } else {
        red = view.getUint8(base + plan.rgb[0]);
        green = view.getUint8(base + plan.rgb[1]);
        blue = view.getUint8(base + plan.rgb[2]);
      }

      let alpha = hasOpacity
        ? 1 / (1 + Math.exp(-view.getFloat32(base + plan.opacity, true)))
        : 1;

      // Scales are stored as logs.
      let sx = Math.exp(view.getFloat32(base + plan.scale[0], true));
      let sy = Math.exp(view.getFloat32(base + plan.scale[1], true));
      // 2DGS surfels have no scale_2 (plan.scale[2] === -1): render a near-flat
      // disk whose thickness is a tiny fraction of its in-plane extent, so the
      // world covariance stays well-conditioned and the disk reads as flat.
      let sz = plan.scale[2] >= 0
        ? Math.exp(view.getFloat32(base + plan.scale[2], true))
        : Math.min(sx, sy) * 1e-3;

      // Mip-Splatting 3D filter: widen the gaussian and compensate opacity.
      if (hasFilter) {
        const filter = view.getFloat32(base + plan.filter3d, true);
        const filterSq = filter * filter;
        if (filterSq > 0) {
          const sx2 = sx * sx + filterSq;
          const sy2 = sy * sy + filterSq;
          const sz2 = sz * sz + filterSq;
          alpha *= Math.sqrt((sx * sx * sy * sy * sz * sz) / (sx2 * sy2 * sz2));
          sx = Math.sqrt(sx2);
          sy = Math.sqrt(sy2);
          sz = Math.sqrt(sz2);
        }
      }

      // Normalized quaternion (rot_0 = w).
      let qw = view.getFloat32(base + plan.rot[0], true);
      let qx = view.getFloat32(base + plan.rot[1], true);
      let qy = view.getFloat32(base + plan.rot[2], true);
      let qz = view.getFloat32(base + plan.rot[3], true);
      const qlen = Math.hypot(qw, qx, qy, qz) || 1;
      qw /= qlen;
      qx /= qlen;
      qy /= qlen;
      qz /= qlen;

      // M = diag(s) * R  (R here is the transpose of the INRIA build_rotation
      // matrix), so sigma = M^T M = R_inria * S^2 * R_inria^T -- identical to
      // the reference CUDA rasterizer's world-space covariance.
      const m0 = (1 - 2 * (qy * qy + qz * qz)) * sx;
      const m1 = 2 * (qx * qy + qw * qz) * sx;
      const m2 = 2 * (qx * qz - qw * qy) * sx;
      const m3 = 2 * (qx * qy - qw * qz) * sy;
      const m4 = (1 - 2 * (qx * qx + qz * qz)) * sy;
      const m5 = 2 * (qy * qz + qw * qx) * sy;
      const m6 = 2 * (qx * qz + qw * qy) * sz;
      const m7 = 2 * (qy * qz - qw * qx) * sz;
      const m8 = (1 - 2 * (qx * qx + qy * qy)) * sz;

      const sigma0 = m0 * m0 + m3 * m3 + m6 * m6; // xx
      const sigma1 = m0 * m1 + m3 * m4 + m6 * m7; // xy
      const sigma2 = m0 * m2 + m3 * m5 + m6 * m8; // xz
      const sigma3 = m1 * m1 + m4 * m4 + m7 * m7; // yy
      const sigma4 = m1 * m2 + m4 * m5 + m7 * m8; // yz
      const sigma5 = m2 * m2 + m5 * m5 + m8 * m8; // zz

      const texBase = index * 8;
      texFloat[texBase] = x;
      texFloat[texBase + 1] = y;
      texFloat[texBase + 2] = z;
      texdata[texBase + 3] = 0;
      // The quad spans [-2, 2]; scaling sigma by 4 makes exp(-dot(p, p)) in the
      // fragment shader equal the true gaussian falloff exp(-r^2 / (2 sigma)).
      texdata[texBase + 4] = packHalf2x16(4 * sigma0, 4 * sigma1);
      texdata[texBase + 5] = packHalf2x16(4 * sigma2, 4 * sigma3);
      texdata[texBase + 6] = packHalf2x16(4 * sigma4, 4 * sigma5);
      texdata[texBase + 7] =
        ((clampByte(alpha * 255) << 24) | (blue << 16) | (green << 8) | red) >>> 0;

      index += 1;
    }

    const consumed = available * stride;
    pending.copyWithin(0, consumed, pendingLength);
    pendingLength -= consumed;
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    bytesSeen += value.byteLength;
    appendChunk(value);

    if (!plan) {
      const headerEnd = findHeaderEnd(pending, pendingLength);
      if (headerEnd < 0) {
        if (pendingLength > (1 << 20)) fail("PLY header exceeds the 1 MiB safety limit.");
        continue;
      }
      const text = new TextDecoder("ascii").decode(pending.subarray(0, headerEnd));
      plan = parseHeaderText(text);
      vertexCount = plan.count;
      positions = new Float32Array(vertexCount * 3);
      texHeight = Math.ceil((2 * vertexCount) / TEX_WIDTH);
      texdata = new Uint32Array(TEX_WIDTH * texHeight * 4);
      texFloat = new Float32Array(texdata.buffer);
      sampleStep = Math.max(1, Math.ceil(vertexCount / 60000));
      pending.copyWithin(0, headerEnd, pendingLength);
      pendingLength -= headerEnd;
      self.postMessage({ type: "header", vertexCount, fileName });
    }

    processRecords();

    if (index >= plan.count) {
      await reader.cancel().catch(() => {});
      break;
    }
    const now = Date.now();
    if (now - lastProgress > 120) {
      lastProgress = now;
      self.postMessage({
        type: "progress",
        bytes: bytesSeen,
        totalBytes,
        count: index,
        vertexCount: plan?.count ?? 0,
      });
    }
  }

  if (!plan) fail("PLY header is incomplete (missing end_header).");
  if (index < plan.count) {
    fail(`PLY ended early: ${index.toLocaleString()} / ${plan.count.toLocaleString()} gaussians.`);
  }

  for (const axis of samples) axis.sort((a, b) => a - b);
  const quantile = (axis, fraction) => axis[Math.round((axis.length - 1) * fraction)];
  const bounds = {
    all: allBounds,
    robust: {
      min: samples.map((axis) => quantile(axis, 0.01)),
      max: samples.map((axis) => quantile(axis, 0.99)),
    },
  };

  self.postMessage(
    {
      type: "ready",
      texdata: texdata.buffer,
      texWidth: TEX_WIDTH,
      texHeight,
      vertexCount,
      bounds,
      fileName,
    },
    [texdata.buffer],
  );
}

function runSort(viewProj) {
  const count = vertexCount;
  const depths = new Float32Array(count);
  let maxDepth = -Infinity;
  let minDepth = Infinity;
  for (let i = 0; i < count; i += 1) {
    const offset = i * 3;
    const depth = viewProj[2] * positions[offset]
      + viewProj[6] * positions[offset + 1]
      + viewProj[10] * positions[offset + 2];
    depths[i] = depth;
    if (depth > maxDepth) maxDepth = depth;
    if (depth < minDepth) minDepth = depth;
  }

  // 16-bit counting sort, ascending depth = front-to-back for the
  // ONE_MINUS_DST_ALPHA "under" blending in gs-viewer.js.  Bucketing runs in
  // float space so outlier coordinates cannot overflow the depth key.
  const bucketScale = minDepth < maxDepth ? (256 * 256 - 1) / (maxDepth - minDepth) : 0;
  const buckets = new Uint32Array(count);
  const counts = new Uint32Array(256 * 256);
  for (let i = 0; i < count; i += 1) {
    let bucket = ((depths[i] - minDepth) * bucketScale) | 0;
    if (!(bucket >= 0)) bucket = 0; // guards NaN depths from non-finite input
    else if (bucket > 256 * 256 - 1) bucket = 256 * 256 - 1;
    buckets[i] = bucket;
    counts[bucket] += 1;
  }
  const starts = new Uint32Array(256 * 256);
  for (let i = 1; i < 256 * 256; i += 1) starts[i] = starts[i - 1] + counts[i - 1];
  const depthIndex = new Uint32Array(count);
  for (let i = 0; i < count; i += 1) {
    depthIndex[starts[buckets[i]]] = i;
    starts[buckets[i]] += 1;
  }
  return depthIndex;
}

self.onmessage = async (event) => {
  const data = event.data;
  try {
    if (data?.type === "load-url") {
      const response = await fetch(data.url, { cache: "no-store" });
      if (!response.ok) fail(`PLY request failed: HTTP ${response.status}`);
      const total = Number.parseInt(response.headers.get("content-length") || "0", 10);
      if (!response.body) fail("Streaming download is unavailable in this browser.");
      await parseStream(response.body, total, data.fileName || "pointcloud.ply");
    } else if (data?.type === "load-file") {
      await parseStream(data.file.stream(), data.file.size, data.file.name);
    } else if (data?.type === "sort") {
      if (!positions) return;
      const started = Date.now();
      const depthIndex = runSort(data.viewProj);
      self.postMessage(
        { type: "sorted", depthIndex: depthIndex.buffer, sortMs: Date.now() - started },
        [depthIndex.buffer],
      );
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
