"use strict";

const TYPE_INFO = {
  char: { size: 1, read: (view, offset) => view.getInt8(offset) },
  int8: { size: 1, read: (view, offset) => view.getInt8(offset) },
  uchar: { size: 1, read: (view, offset) => view.getUint8(offset) },
  uint8: { size: 1, read: (view, offset) => view.getUint8(offset) },
  short: { size: 2, read: (view, offset) => view.getInt16(offset, true) },
  int16: { size: 2, read: (view, offset) => view.getInt16(offset, true) },
  ushort: { size: 2, read: (view, offset) => view.getUint16(offset, true) },
  uint16: { size: 2, read: (view, offset) => view.getUint16(offset, true) },
  int: { size: 4, read: (view, offset) => view.getInt32(offset, true) },
  int32: { size: 4, read: (view, offset) => view.getInt32(offset, true) },
  uint: { size: 4, read: (view, offset) => view.getUint32(offset, true) },
  uint32: { size: 4, read: (view, offset) => view.getUint32(offset, true) },
  float: { size: 4, read: (view, offset) => view.getFloat32(offset, true) },
  float32: { size: 4, read: (view, offset) => view.getFloat32(offset, true) },
  double: { size: 8, read: (view, offset) => view.getFloat64(offset, true) },
  float64: { size: 8, read: (view, offset) => view.getFloat64(offset, true) },
};

const CAMERA_RUNS = [
  { name: "back", color: [115, 217, 64] },
  { name: "down", color: [25, 217, 199] },
  { name: "front", color: [26, 140, 255] },
  { name: "left", color: [255, 196, 32] },
  { name: "right", color: [255, 77, 56] },
  { name: "up", color: [190, 90, 255] },
];
const TRAJECTORY_COLOR = [255, 255, 255];
const SH_C0 = 0.28209479177387814;

function fail(message) {
  throw new Error(message);
}

function findHeaderEnd(bytes) {
  const marker = new TextEncoder().encode("end_header");
  outer: for (let i = 0; i <= bytes.length - marker.length; i += 1) {
    if (i > 0 && bytes[i - 1] !== 10) continue;
    for (let j = 0; j < marker.length; j += 1) {
      if (bytes[i + j] !== marker[j]) continue outer;
    }
    const end = i + marker.length;
    if (bytes[end] === 10) return end + 1;
    if (bytes[end] === 13 && bytes[end + 1] === 10) return end + 2;
  }
  fail("PLY header has no end_header marker.");
}

function parseHeader(buffer) {
  const bytes = new Uint8Array(buffer);
  const headerBytes = findHeaderEnd(bytes);
  const text = new TextDecoder("ascii").decode(bytes.subarray(0, headerBytes));
  const lines = text.split(/\r?\n/);
  if (lines[0]?.trim() !== "ply") fail("The selected file is not a PLY.");

  let format = null;
  let currentElement = null;
  let vertexCount = null;
  let recordBytes = 0;
  let cameraElementCount = 0;
  let cameraRecordBytes = 0;
  const properties = [];
  const comments = [];
  const otherElements = [];

  for (const rawLine of lines.slice(1)) {
    const line = rawLine.trim();
    if (!line) continue;
    const parts = line.split(/\s+/);
    if (parts[0] === "format") {
      format = `${parts[1]} ${parts[2]}`;
    } else if (parts[0] === "comment") {
      comments.push(line.slice("comment".length).trim());
    } else if (parts[0] === "element" && parts.length === 3) {
      currentElement = parts[1];
      const count = Number.parseInt(parts[2], 10);
      if (!Number.isSafeInteger(count) || count < 0) fail(`Invalid element count: ${line}`);
      if (currentElement === "vertex") vertexCount = count;
      else if (currentElement === "camera") cameraElementCount = count;
      else if (count > 0) otherElements.push(`${currentElement}=${count}`);
    } else if (parts[0] === "property" && (currentElement === "vertex" || currentElement === "camera")) {
      if (parts[1] === "list") fail("List properties are not supported.");
      if (parts.length !== 3 || !TYPE_INFO[parts[1]]) fail(`Unsupported property: ${line}`);
      if (currentElement === "vertex") {
        properties.push({ name: parts[2], type: parts[1], offset: recordBytes });
        recordBytes += TYPE_INFO[parts[1]].size;
      } else {
        cameraRecordBytes += TYPE_INFO[parts[1]].size;
      }
    }
  }

  if (format !== "binary_little_endian 1.0") {
    fail("Only binary_little_endian PLY 1.0 is supported.");
  }
  if (!Number.isSafeInteger(vertexCount) || vertexCount <= 0) fail("PLY has no vertices.");
  if (otherElements.length) fail(`Additional PLY elements are not supported: ${otherElements.join(", ")}`);
  if (headerBytes + vertexCount * recordBytes + cameraElementCount * cameraRecordBytes !== buffer.byteLength) {
    fail(
      `PLY size does not match its header (${buffer.byteLength.toLocaleString()} bytes received).`,
    );
  }

  const byName = Object.fromEntries(properties.map((property) => [property.name, property]));
  for (const name of ["x", "y", "z"]) {
    if (!byName[name]) fail(`PLY is missing the '${name}' property.`);
  }
  return {
    headerBytes,
    vertexCount,
    recordBytes,
    properties,
    byName,
    comments,
    cameraElementCount,
  };
}

function makePropertyReader(dataView, header, name) {
  const property = header.byName[name];
  if (!property) return null;
  const read = TYPE_INFO[property.type].read;
  return (vertexBase) => read(dataView, vertexBase + property.offset);
}

function colorsEqual(a, b) {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
}

function detectCameraLayout(dataView, header) {
  const pointType = header.byName.point_type;
  if (pointType && ["uchar", "uint8"].includes(pointType.type) && header.cameraElementCount) {
    const typeAt = (index) => {
      const base = header.headerBytes + index * header.recordBytes;
      return dataView.getUint8(base + pointType.offset);
    };
    let index = 0;
    const ranges = [];
    for (const expectedType of [0, 1, 2]) {
      const start = index;
      while (index < header.vertexCount && typeAt(index) === expectedType) index += 1;
      ranges.push({ start, count: index - start });
    }
    const [scene, cameras, trajectory] = ranges;
    if (scene.count && cameras.count && index === header.vertexCount) {
      if (cameras.count % header.cameraElementCount) {
        return {
          detected: false,
          sceneCount: header.vertexCount,
          reason: "Camera point count is inconsistent with the camera element.",
        };
      }
      const samplesPerFrustum = cameras.count / header.cameraElementCount;
      return {
        detected: true,
        sceneCount: scene.count,
        cameraStart: cameras.start,
        cameraCount: cameras.count,
        trajectoryStart: trajectory.start,
        trajectoryCount: trajectory.count,
        viewCount: header.cameraElementCount,
        samplesPerFrustum,
        runs: [{ name: "camera_frustum", color: [255, 255, 255], start: cameras.start, count: cameras.count }],
      };
    }
  }

  const rgbProperties = [header.byName.red, header.byName.green, header.byName.blue];
  if (rgbProperties.some((property) => !property)) {
    return { detected: false, sceneCount: header.vertexCount, reason: "RGB properties are missing." };
  }
  if (rgbProperties.some((property) => !["uchar", "uint8"].includes(property.type))) {
    return {
      detected: false,
      sceneCount: header.vertexCount,
      reason: "Camera markers require uchar RGB properties.",
    };
  }

  const colorAt = (index) => {
    const base = header.headerBytes + index * header.recordBytes;
    return rgbProperties.map((property) => dataView.getUint8(base + property.offset));
  };

  let index = header.vertexCount - 1;
  while (index >= 0 && colorsEqual(colorAt(index), TRAJECTORY_COLOR)) index -= 1;
  const trajectoryStart = index + 1;
  const trajectoryCount = header.vertexCount - trajectoryStart;

  const reverseRuns = [];
  for (let runIndex = CAMERA_RUNS.length - 1; runIndex >= 0; runIndex -= 1) {
    const expected = CAMERA_RUNS[runIndex];
    if (index < 0 || !colorsEqual(colorAt(index), expected.color)) {
      return {
        detected: false,
        sceneCount: header.vertexCount,
        reason: `Expected trailing '${expected.name}' camera run was not found.`,
      };
    }
    const runEnd = index + 1;
    while (index >= 0 && colorsEqual(colorAt(index), expected.color)) index -= 1;
    const start = index + 1;
    reverseRuns.push({ ...expected, start, count: runEnd - start });
  }

  const runs = reverseRuns.reverse();
  const runCount = runs[0].count;
  if (runs.some((run) => run.count !== runCount)) {
    return {
      detected: false,
      sceneCount: header.vertexCount,
      reason: "The six camera runs have different lengths.",
    };
  }
  if (runCount % 128 !== 0) {
    return {
      detected: false,
      sceneCount: header.vertexCount,
      reason: "Camera marker count is not divisible by 128 samples.",
    };
  }
  const viewCount = runCount / 128;
  if (trajectoryCount !== Math.max(0, viewCount - 1) * 8) {
    return {
      detected: false,
      sceneCount: header.vertexCount,
      reason: "Camera marker and trajectory counts are inconsistent.",
    };
  }
  return {
    detected: true,
    sceneCount: runs[0].start,
    cameraStart: runs[0].start,
    cameraCount: trajectoryStart - runs[0].start,
    trajectoryStart,
    trajectoryCount,
    viewCount,
    samplesPerFrustum: 128,
    runs,
  };
}

function clampByte(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function makeColorReader(dataView, header) {
  const direct = ["red", "green", "blue"].map((name) => ({
    property: header.byName[name],
    read: makePropertyReader(dataView, header, name),
  }));
  if (direct.every((item) => item.read)) {
    return (base) => direct.map((item) => {
      const value = item.read(base);
      const isFloat = ["float", "float32", "double", "float64"].includes(item.property.type);
      return clampByte(isFloat && value >= 0 && value <= 1 ? value * 255 : value);
    });
  }

  const shReaders = ["f_dc_0", "f_dc_1", "f_dc_2"].map((name) =>
    makePropertyReader(dataView, header, name),
  );
  if (shReaders.every(Boolean)) {
    return (base) => shReaders.map((read) => clampByte((0.5 + SH_C0 * read(base)) * 255));
  }
  return () => [220, 225, 235];
}

function emptyBounds() {
  return {
    min: [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY],
    max: [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY],
  };
}

function addToBounds(bounds, x, y, z) {
  if (x < bounds.min[0]) bounds.min[0] = x;
  if (y < bounds.min[1]) bounds.min[1] = y;
  if (z < bounds.min[2]) bounds.min[2] = z;
  if (x > bounds.max[0]) bounds.max[0] = x;
  if (y > bounds.max[1]) bounds.max[1] = y;
  if (z > bounds.max[2]) bounds.max[2] = z;
}

function boundsForRange(positions, start, count) {
  if (!count) return null;
  const bounds = emptyBounds();
  const end = start + count;
  for (let index = start; index < end; index += 1) {
    const offset = index * 3;
    addToBounds(bounds, positions[offset], positions[offset + 1], positions[offset + 2]);
  }
  return bounds;
}

function robustBounds(positions, count) {
  if (!count) return null;
  const maxSamples = 50000;
  const step = Math.max(1, Math.ceil(count / maxSamples));
  const axes = [[], [], []];
  for (let index = 0; index < count; index += step) {
    const offset = index * 3;
    axes[0].push(positions[offset]);
    axes[1].push(positions[offset + 1]);
    axes[2].push(positions[offset + 2]);
  }
  for (const axis of axes) axis.sort((a, b) => a - b);
  const quantile = (axis, fraction) => axis[Math.round((axis.length - 1) * fraction)];
  return {
    min: axes.map((axis) => quantile(axis, 0.01)),
    max: axes.map((axis) => quantile(axis, 0.99)),
  };
}

function parsePly(buffer, fileName) {
  const dataView = new DataView(buffer);
  const header = parseHeader(buffer);
  const layout = detectCameraLayout(dataView, header);
  const readX = makePropertyReader(dataView, header, "x");
  const readY = makePropertyReader(dataView, header, "y");
  const readZ = makePropertyReader(dataView, header, "z");
  const readColor = makeColorReader(dataView, header);
  const positions = new Float32Array(header.vertexCount * 3);
  const colors = new Uint8Array(header.vertexCount * 3);
  const allBounds = emptyBounds();
  const updateInterval = Math.max(50000, Math.floor(header.vertexCount / 60));

  for (let index = 0; index < header.vertexCount; index += 1) {
    const base = header.headerBytes + index * header.recordBytes;
    const x = readX(base);
    const y = readY(base);
    const z = readZ(base);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      fail(`Vertex ${index.toLocaleString()} contains a non-finite position.`);
    }
    const offset = index * 3;
    positions[offset] = x;
    positions[offset + 1] = y;
    positions[offset + 2] = z;
    const color = readColor(base);
    colors[offset] = color[0];
    colors[offset + 1] = color[1];
    colors[offset + 2] = color[2];
    addToBounds(allBounds, x, y, z);

    if (index % updateInterval === 0) {
      self.postMessage({ type: "progress", phase: "parse", value: index / header.vertexCount });
    }
  }

  const sceneCount = layout.sceneCount;
  const bounds = {
    all: allBounds,
    scene: boundsForRange(positions, 0, sceneCount),
    sceneRobust: robustBounds(positions, sceneCount),
    cameras: layout.detected
      ? boundsForRange(positions, layout.cameraStart, layout.cameraCount)
      : null,
    trajectory: layout.detected
      ? boundsForRange(positions, layout.trajectoryStart, layout.trajectoryCount)
      : null,
  };

  return {
    positions,
    colors,
    metadata: {
      fileName,
      fileBytes: buffer.byteLength,
      vertexCount: header.vertexCount,
      recordBytes: header.recordBytes,
      comments: header.comments,
      layout,
      bounds,
    },
  };
}

self.onmessage = (event) => {
  if (event.data?.type !== "parse" || !(event.data.buffer instanceof ArrayBuffer)) return;
  try {
    const result = parsePly(event.data.buffer, event.data.fileName || "pointcloud.ply");
    self.postMessage(
      {
        type: "result",
        positions: result.positions.buffer,
        colors: result.colors.buffer,
        metadata: result.metadata,
      },
      [result.positions.buffer, result.colors.buffer],
    );
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
