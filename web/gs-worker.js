"use strict";

// Streaming parser + depth sorter for Gaussian PLY files. Each Gaussian is
// packed as raw center / activated scale / quaternion data so gs-viewer.js can
// execute the official 2DGS surfel and ArtiFixer 3DGUT projection equations.

const SH_C0 = 0.28209479177387814;
const TEXELS_PER_SPLAT = 3;
const SPLATS_PER_ROW = 682;
const TEX_WIDTH = TEXELS_PER_SPLAT * SPLATS_PER_ROW;

const RENDERER_STANDARD_3DGS = 1;
const RENDERER_2DGS = 2;
const RENDERER_ARTIFIXER_3DGUT = 3;
const RENDERER_LIGHT_PROXY = 4;

const FLOAT_TYPES = new Set(["float", "float32"]);
const PLY_TYPE_SIZES = {
  char: 1, int8: 1, uchar: 1, uint8: 1,
  short: 2, int16: 2, ushort: 2, uint16: 2,
  int: 4, int32: 4, uint: 4, uint32: 4,
  float: 4, float32: 4, double: 8, float64: 8,
};

let vertexCount = 0;
let positions = null; // Float32Array(3n), retained for sorting

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

  const scale2 = byName.scale_2 && FLOAT_TYPES.has(byName.scale_2.type)
    ? byName.scale_2.offset
    : -1;
  const plan = {
    count,
    stride,
    dimension: scale2 >= 0 ? 3 : 2,
    x: requireFloat("x"),
    y: requireFloat("y"),
    z: requireFloat("z"),
    scale: [requireFloat("scale_0"), requireFloat("scale_1"), scale2],
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

function lightProxyCount(lightSources) {
  if (!Array.isArray(lightSources?.fixtures)) return 0;
  return lightSources.fixtures.reduce((sum, fixture) => {
    const count = fixture?.parametric_proxy?.proxy_count;
    return sum + (Number.isInteger(count) && count > 0 ? count : 0);
  }, 0);
}

function quaternionFromRotationMatrix(matrix) {
  const m00 = matrix[0][0];
  const m01 = matrix[0][1];
  const m02 = matrix[0][2];
  const m10 = matrix[1][0];
  const m11 = matrix[1][1];
  const m12 = matrix[1][2];
  const m20 = matrix[2][0];
  const m21 = matrix[2][1];
  const m22 = matrix[2][2];
  const trace = m00 + m11 + m22;
  let qw;
  let qx;
  let qy;
  let qz;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    qw = 0.25 * s;
    qx = (m21 - m12) / s;
    qy = (m02 - m20) / s;
    qz = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    qw = (m21 - m12) / s;
    qx = 0.25 * s;
    qy = (m01 + m10) / s;
    qz = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    qw = (m02 - m20) / s;
    qx = (m01 + m10) / s;
    qy = 0.25 * s;
    qz = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    qw = (m10 - m01) / s;
    qx = (m02 + m20) / s;
    qy = (m12 + m21) / s;
    qz = 0.25 * s;
  }
  return [qw, qx, qy, qz];
}

function appendLightProxies(lightSources, startIndex, texdata, texFloat) {
  const bindings = [];
  let index = startIndex;
  for (const fixture of lightSources?.fixtures || []) {
    const proxy = fixture?.parametric_proxy;
    if (!proxy) continue;
    const count = proxy.proxy_count;
    const position = proxy.position_world;
    const extent = proxy.extent_98_percent_world_units;
    const rotation = proxy.rotation_world_from_proxy;
    const baseColor = proxy.base_emission_srgb_power_1;
    if (!Number.isInteger(count) || count <= 0
      || ![position, extent, baseColor].every((value) => Array.isArray(value) && value.length === 3)
      || !Array.isArray(rotation) || rotation.length !== 3) continue;

    const proxyStart = index;
    const segmentLength = Math.max(0.02, Math.abs(extent[0]) / count);
    const scales = [
      Math.max(0.01, segmentLength / 3.2),
      Math.max(0.01, Math.abs(extent[1]) / 3.2),
      Math.max(0.008, Math.abs(extent[2]) / 3.2),
    ];
    const quaternion = quaternionFromRotationMatrix(rotation);
    const intensity = Number.isFinite(proxy.relative_intensity_0_1)
      ? proxy.relative_intensity_0_1 : 0.5;
    const red = clampByte(baseColor[0] * intensity * 255);
    const green = clampByte(baseColor[1] * intensity * 255);
    const blue = clampByte(baseColor[2] * intensity * 255);
    const proxyPositions = [];

    for (let localIndex = 0; localIndex < count; localIndex += 1) {
      const along = (localIndex - (count - 1) / 2) * segmentLength;
      const x = position[0] + rotation[0][0] * along;
      const y = position[1] + rotation[1][0] * along;
      const z = position[2] + rotation[2][0] * along;
      const offset3 = index * 3;
      positions[offset3] = x;
      positions[offset3 + 1] = y;
      positions[offset3 + 2] = z;
      proxyPositions.push(x, y, z);
      const texBase = index * TEXELS_PER_SPLAT * 4;
      texFloat[texBase] = x;
      texFloat[texBase + 1] = y;
      texFloat[texBase + 2] = z;
      texFloat[texBase + 3] = 1;
      texFloat[texBase + 4] = scales[0];
      texFloat[texBase + 5] = scales[1];
      texFloat[texBase + 6] = scales[2];
      texdata[texBase + 7] =
        ((RENDERER_LIGHT_PROXY << 24) | (blue << 16) | (green << 8) | red) >>> 0;
      texFloat[texBase + 8] = quaternion[0];
      texFloat[texBase + 9] = quaternion[1];
      texFloat[texBase + 10] = quaternion[2];
      texFloat[texBase + 11] = quaternion[3];
      index += 1;
    }
    bindings.push({
      fixtureId: fixture.fixture_id,
      fixtureType: fixture.specification?.fixture_type || "unknown",
      proxyStart,
      proxyCount: count,
      proxyPositions,
      baseColor,
      defaultIntensity: intensity,
    });
  }
  return { bindings, endIndex: index };
}

async function parseStream(stream, totalBytes, fileName, lightSources = null, rendererHint = null) {
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
  let rendererKind = null;
  let rendererCode = 0;

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
    const hasFilter = rendererKind === "3DGS" && plan.filter3d >= 0;
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
        const texBaseSkip = index * TEXELS_PER_SPLAT * 4;
        texdata.fill(0, texBaseSkip, texBaseSkip + TEXELS_PER_SPLAT * 4);
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
      // 2DGS stores only the two tangent-plane scales. The official surfel
      // renderer consumes these directly and never invents a third scale.
      let sz = plan.dimension === 3
        ? Math.exp(view.getFloat32(base + plan.scale[2], true))
        : 0;

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

      const texBase = index * TEXELS_PER_SPLAT * 4;
      texFloat[texBase] = x;
      texFloat[texBase + 1] = y;
      texFloat[texBase + 2] = z;
      texFloat[texBase + 3] = alpha;
      texFloat[texBase + 4] = sx;
      texFloat[texBase + 5] = sy;
      texFloat[texBase + 6] = sz;
      texdata[texBase + 7] =
        ((rendererCode << 24) | (blue << 16) | (green << 8) | red) >>> 0;
      texFloat[texBase + 8] = qw;
      texFloat[texBase + 9] = qx;
      texFloat[texBase + 10] = qy;
      texFloat[texBase + 11] = qz;

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
      const artifixerByName = /artifixer/i.test(fileName);
      rendererKind = plan.dimension === 2
        ? "2DGS"
        : (rendererHint === "ArtiFixer3D" || artifixerByName ? "ArtiFixer3D" : "3DGS");
      rendererCode = rendererKind === "2DGS"
        ? RENDERER_2DGS
        : (rendererKind === "ArtiFixer3D" ? RENDERER_ARTIFIXER_3DGUT : RENDERER_STANDARD_3DGS);
      const generatedLightCount = lightProxyCount(lightSources);
      vertexCount = plan.count + generatedLightCount;
      positions = new Float32Array(vertexCount * 3);
      texHeight = Math.ceil(vertexCount / SPLATS_PER_ROW);
      texdata = new Uint32Array(TEX_WIDTH * texHeight * 4);
      texFloat = new Float32Array(texdata.buffer);
      sampleStep = Math.max(1, Math.ceil(vertexCount / 60000));
      pending.copyWithin(0, headerEnd, pendingLength);
      pendingLength -= headerEnd;
      self.postMessage({
        type: "header",
        vertexCount: plan.count,
        lightProxyCount: generatedLightCount,
        gaussianKind: rendererKind,
        fileName,
      });
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

  const lightResult = appendLightProxies(lightSources, index, texdata, texFloat);
  index = lightResult.endIndex;
  if (index !== vertexCount) {
    fail(`Generated light proxy count mismatch: ${index - plan.count} / ${vertexCount - plan.count}.`);
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
      sceneVertexCount: plan.count,
      gaussianKind: rendererKind,
      lightBindings: lightResult.bindings,
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
      await parseStream(
        response.body,
        total,
        data.fileName || "pointcloud.ply",
        data.lightSources || null,
        data.rendererHint || null,
      );
    } else if (data?.type === "load-file") {
      await parseStream(
        data.file.stream(),
        data.file.size,
        data.file.name,
        data.lightSources || null,
        data.rendererHint || null,
      );
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
