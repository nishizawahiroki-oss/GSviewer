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
  const elements = [];
  const properties = [];
  const cameraProperties = [];
  const comments = [];

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
      elements.push({ name: currentElement, count, recordBytes: 0, variableLength: false });
      if (currentElement === "vertex") vertexCount = count;
      else if (currentElement === "camera") cameraElementCount = count;
    } else if (parts[0] === "property") {
      const element = elements[elements.length - 1];
      if (!element) fail(`Property appears before an element: ${line}`);
      if (parts[1] === "list") {
        if (
          parts.length !== 5
          || !TYPE_INFO[parts[2]]
          || !TYPE_INFO[parts[3]]
          || element.name !== "face"
        ) {
          fail(`Unsupported list property: ${line}`);
        }
        element.variableLength = true;
        continue;
      }
      if (parts.length !== 3 || !TYPE_INFO[parts[1]]) fail(`Unsupported property: ${line}`);
      element.recordBytes += TYPE_INFO[parts[1]].size;
      if (currentElement === "vertex") {
        properties.push({ name: parts[2], type: parts[1], offset: recordBytes });
        recordBytes += TYPE_INFO[parts[1]].size;
      } else if (currentElement === "camera") {
        cameraProperties.push({ name: parts[2], type: parts[1], offset: cameraRecordBytes });
        cameraRecordBytes += TYPE_INFO[parts[1]].size;
      }
    }
  }

  if (format !== "binary_little_endian 1.0") {
    fail("Only binary_little_endian PLY 1.0 is supported.");
  }
  if (!Number.isSafeInteger(vertexCount) || vertexCount <= 0) fail("PLY has no vertices.");
  const otherElements = elements.filter(
    (element) => !["vertex", "camera", "face"].includes(element.name) && element.count,
  );
  if (otherElements.length) {
    fail(
      `Additional PLY elements are not supported: ${otherElements
        .map((element) => `${element.name}=${element.count}`)
        .join(", ")}`,
    );
  }
  const variableElements = elements.filter((element) => element.variableLength);
  if (variableElements.some((element) => element.name !== "face")) {
    fail("Variable-length PLY elements other than face are not supported.");
  }
  if (variableElements.length && cameraElementCount) {
    fail("PLY files containing both face lists and camera elements are not supported.");
  }
  const fixedPayloadBytes = elements
    .filter((element) => !element.variableLength)
    .reduce((total, element) => total + element.count * element.recordBytes, 0);
  const minimumSize = headerBytes + fixedPayloadBytes;
  if (
    variableElements.length
      ? buffer.byteLength < minimumSize
      : buffer.byteLength !== minimumSize
  ) {
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
    cameraRecordBytes,
    cameraProperties,
    faceCount: elements.find((element) => element.name === "face")?.count || 0,
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

const FRUSTUM_FACE_COLORS = [
  [26, 140, 255], // face 0: front (blue)
  [255, 77, 56], // face 1: right (red)
  [115, 217, 64], // face 2: back (green)
  [255, 196, 32], // face 3: left (yellow)
  [190, 90, 255], // face 4: up (purple)
  [25, 217, 199], // face 5: down (cyan)
];

const FRUSTUM_REQUIRED = [
  "center_x", "center_y", "center_z",
  "world_from_camera_r00", "world_from_camera_r01", "world_from_camera_r02",
  "world_from_camera_r10", "world_from_camera_r11", "world_from_camera_r12",
  "world_from_camera_r20", "world_from_camera_r21", "world_from_camera_r22",
  "fx", "fy", "cx", "cy", "width", "height",
];

// Rebuild exact camera frusta as solid line segments from the PLY's camera
// element (COLMAP convention: camera +X right, +Y down, +Z forward).
function buildCameraFrustums(dataView, header) {
  if (!header.cameraElementCount || !header.cameraRecordBytes) return null;
  const byName = Object.fromEntries(
    header.cameraProperties.map((property) => [property.name, property]),
  );
  if (FRUSTUM_REQUIRED.some((name) => !byName[name])) return null;

  const count = header.cameraElementCount;
  const elementBase = header.headerBytes + header.vertexCount * header.recordBytes;
  const readers = Object.fromEntries(
    Object.entries(byName).map(([name, property]) => [
      name,
      (cameraIndex) => TYPE_INFO[property.type].read(
        dataView,
        elementBase + cameraIndex * header.cameraRecordBytes + property.offset,
      ),
    ]),
  );

  const centers = new Float64Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    centers[index * 3] = readers.center_x(index);
    centers[index * 3 + 1] = readers.center_y(index);
    centers[index * 3 + 2] = readers.center_z(index);
  }

  // Frustum depth: median spacing between consecutive distinct camera centers
  // (multi-face rigs repeat the same center, so skip near-zero steps).
  const spacings = [];
  for (let index = 1; index < count; index += 1) {
    const distance = Math.hypot(
      centers[index * 3] - centers[(index - 1) * 3],
      centers[index * 3 + 1] - centers[(index - 1) * 3 + 1],
      centers[index * 3 + 2] - centers[(index - 1) * 3 + 2],
    );
    if (distance > 1e-9) spacings.push(distance);
  }
  spacings.sort((a, b) => a - b);
  let depth = spacings.length ? spacings[Math.floor(spacings.length / 2)] * 1.2 : 0;
  if (!(depth > 0)) {
    let minX = Infinity; let minY = Infinity; let minZ = Infinity;
    let maxX = -Infinity; let maxY = -Infinity; let maxZ = -Infinity;
    for (let index = 0; index < count; index += 1) {
      minX = Math.min(minX, centers[index * 3]); maxX = Math.max(maxX, centers[index * 3]);
      minY = Math.min(minY, centers[index * 3 + 1]); maxY = Math.max(maxY, centers[index * 3 + 1]);
      minZ = Math.min(minZ, centers[index * 3 + 2]); maxZ = Math.max(maxZ, centers[index * 3 + 2]);
    }
    depth = Math.max(1e-3, Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) * 0.05);
  }

  // 8 solid segments per camera: 4 center->corner rays + 4 far-plane edges.
  const VERTICES_PER_CAMERA = 16;
  const positions = new Float32Array(count * VERTICES_PER_CAMERA * 3);
  const colors = new Uint8Array(count * VERTICES_PER_CAMERA * 3);
  const faceReader = byName.face_id ? readers.face_id : null;
  const corner = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];

  for (let index = 0; index < count; index += 1) {
    const cx = readers.cx(index);
    const cy = readers.cy(index);
    const fx = readers.fx(index);
    const fy = readers.fy(index);
    const width = readers.width(index);
    const height = readers.height(index);
    const rotation = [
      readers.world_from_camera_r00(index), readers.world_from_camera_r01(index), readers.world_from_camera_r02(index),
      readers.world_from_camera_r10(index), readers.world_from_camera_r11(index), readers.world_from_camera_r12(index),
      readers.world_from_camera_r20(index), readers.world_from_camera_r21(index), readers.world_from_camera_r22(index),
    ];
    const centerX = centers[index * 3];
    const centerY = centers[index * 3 + 1];
    const centerZ = centers[index * 3 + 2];

    const pixelCorners = [[0, 0], [width, 0], [width, height], [0, height]];
    for (let cornerIndex = 0; cornerIndex < 4; cornerIndex += 1) {
      const [u, v] = pixelCorners[cornerIndex];
      const dirX = ((u - cx) / fx) * depth;
      const dirY = ((v - cy) / fy) * depth;
      const dirZ = depth;
      corner[cornerIndex][0] = centerX + rotation[0] * dirX + rotation[1] * dirY + rotation[2] * dirZ;
      corner[cornerIndex][1] = centerY + rotation[3] * dirX + rotation[4] * dirY + rotation[5] * dirZ;
      corner[cornerIndex][2] = centerZ + rotation[6] * dirX + rotation[7] * dirY + rotation[8] * dirZ;
    }

    const color = faceReader
      ? FRUSTUM_FACE_COLORS[faceReader(index) % FRUSTUM_FACE_COLORS.length]
      : [255, 255, 255];
    const segments = [
      [centerX, centerY, centerZ], corner[0],
      [centerX, centerY, centerZ], corner[1],
      [centerX, centerY, centerZ], corner[2],
      [centerX, centerY, centerZ], corner[3],
      corner[0], corner[1],
      corner[1], corner[2],
      corner[2], corner[3],
      corner[3], corner[0],
    ];
    const vertexBase = index * VERTICES_PER_CAMERA;
    for (let vertexIndex = 0; vertexIndex < VERTICES_PER_CAMERA; vertexIndex += 1) {
      const outOffset = (vertexBase + vertexIndex) * 3;
      positions[outOffset] = segments[vertexIndex][0];
      positions[outOffset + 1] = segments[vertexIndex][1];
      positions[outOffset + 2] = segments[vertexIndex][2];
      colors[outOffset] = color[0];
      colors[outOffset + 1] = color[1];
      colors[outOffset + 2] = color[2];
    }
  }

  return { positions, colors, count, verticesPerCamera: VERTICES_PER_CAMERA };
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

  const frustums = buildCameraFrustums(dataView, header);

  return {
    positions,
    colors,
    frustums,
    metadata: {
      fileName,
      fileBytes: buffer.byteLength,
      vertexCount: header.vertexCount,
      recordBytes: header.recordBytes,
      mesh: {
        available: header.faceCount > 0,
        faceCount: header.faceCount,
      },
      comments: header.comments,
      layout,
      bounds,
      frustum: frustums
        ? { count: frustums.count, verticesPerCamera: frustums.verticesPerCamera }
        : null,
    },
  };
}

self.onmessage = (event) => {
  if (event.data?.type !== "parse" || !(event.data.buffer instanceof ArrayBuffer)) return;
  try {
    const result = parsePly(event.data.buffer, event.data.fileName || "pointcloud.ply");
    const transfers = [result.positions.buffer, result.colors.buffer];
    if (result.frustums) {
      transfers.push(result.frustums.positions.buffer, result.frustums.colors.buffer);
    }
    self.postMessage(
      {
        type: "result",
        positions: result.positions.buffer,
        colors: result.colors.buffer,
        frustumPositions: result.frustums ? result.frustums.positions.buffer : null,
        frustumColors: result.frustums ? result.frustums.colors.buffer : null,
        metadata: result.metadata,
      },
      transfers,
    );
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
