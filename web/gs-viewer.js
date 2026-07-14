"use strict";

// WebGL2 gaussian-splat renderer: anisotropic EWA splats with front-to-back
// "under" alpha blending.  Parsing and depth sorting run in gs-worker.js.

const canvas = document.querySelector("#viewport");
const ui = Object.fromEntries(
  [
    "file-name",
    "file-input",
    "splat-count",
    "file-size",
    "sort-time",
    "fps",
    "background",
    "splat-scale",
    "splat-scale-value",
    "alpha-threshold",
    "alpha-threshold-value",
    "flip-z",
    "fit-scene",
    "gpu-info",
    "loading",
    "progress-label",
    "progress-bar",
    "progress-detail",
    "error-overlay",
    "error-box",
  ].map((id) => [id, document.querySelector(`#${id}`)]),
);

const gl = canvas.getContext("webgl2", {
  alpha: true,
  antialias: false,
  depth: false,
  powerPreference: "high-performance",
  preserveDrawingBuffer: false,
});

if (!gl) {
  showError("WebGL2 を利用できません。GPU アクセラレーションを有効にした Chrome / Safari / Firefox を使用してください。");
  throw new Error("WebGL2 is unavailable.");
}

const VERTEX_SHADER = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;

uniform usampler2D uTexture;
uniform mat4 uProjection;
uniform mat4 uView;
uniform vec2 uFocal;
uniform vec2 uViewport;
uniform float uSplatScale;
uniform float uAlphaThreshold;

layout(location = 0) in vec2 aPosition;
layout(location = 1) in int aIndex;

out vec4 vColor;
out vec2 vPosition;

void main() {
  uvec4 cen = texelFetch(uTexture, ivec2((uint(aIndex) & 0x3ffu) << 1, uint(aIndex) >> 10), 0);
  vec4 cam = uView * vec4(uintBitsToFloat(cen.xyz), 1.0);
  vec4 pos2d = uProjection * cam;

  float clip = 1.2 * pos2d.w;
  if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }

  uvec4 cov = texelFetch(uTexture, ivec2(((uint(aIndex) & 0x3ffu) << 1) | 1u, uint(aIndex) >> 10), 0);
  if (float(cov.w >> 24) / 255.0 < uAlphaThreshold) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }
  vec2 u1 = unpackHalf2x16(cov.x);
  vec2 u2 = unpackHalf2x16(cov.y);
  vec2 u3 = unpackHalf2x16(cov.z);
  mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);

  mat3 J = mat3(
    uFocal.x / cam.z, 0.0, -(uFocal.x * cam.x) / (cam.z * cam.z),
    0.0, -uFocal.y / cam.z, (uFocal.y * cam.y) / (cam.z * cam.z),
    0.0, 0.0, 0.0
  );
  mat3 T = transpose(mat3(uView)) * J;
  mat3 cov2d = transpose(T) * Vrk * T;

  // Mip-Splatting screen-space low-pass: dilate by 0.1 px^2 and compensate
  // the opacity by the determinant ratio (this checkpoint stores filter_3D,
  // so it was trained with this 2D kernel).
  float detBefore = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1];
  cov2d[0][0] += 0.1;
  cov2d[1][1] += 0.1;
  float detAfter = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1];
  float alphaCompensation = sqrt(max(0.0, detBefore / detAfter));

  float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
  float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
  float lambda1 = mid + radius;
  float lambda2 = mid - radius;
  if (lambda2 < 0.0) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }
  vec2 eigenDirection = vec2(cov2d[0][1], lambda1 - cov2d[0][0]);
  vec2 diagonalVector = dot(eigenDirection, eigenDirection) < 1.0e-24
    ? vec2(1.0, 0.0)
    : normalize(eigenDirection);
  vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector * uSplatScale;
  vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x) * uSplatScale;

  vColor = clamp(pos2d.z / pos2d.w + 1.0, 0.0, 1.0)
    * vec4(float(cov.w & 0xffu), float((cov.w >> 8) & 0xffu), float((cov.w >> 16) & 0xffu), float(cov.w >> 24) * alphaCompensation) / 255.0;
  vPosition = aPosition;

  vec2 vCenter = pos2d.xy / pos2d.w;
  gl_Position = vec4(
    vCenter + aPosition.x * majorAxis / uViewport + aPosition.y * minorAxis / uViewport,
    0.0,
    1.0
  );
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec4 vColor;
in vec2 vPosition;
out vec4 fragColor;

void main() {
  float A = -dot(vPosition, vPosition);
  if (A < -4.0) discard;
  float B = exp(A) * vColor.a;
  fragColor = vec4(B * vColor.rgb, B);
}
`;

function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) || "Unknown shader compilation error";
    gl.deleteShader(shader);
    throw new Error(message);
  }
  return shader;
}

function createProgram() {
  const vertex = compileShader(gl.VERTEX_SHADER, VERTEX_SHADER);
  const fragment = compileShader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
  const result = gl.createProgram();
  gl.attachShader(result, vertex);
  gl.attachShader(result, fragment);
  gl.linkProgram(result);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(result, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(result) || "Unknown shader link error";
    gl.deleteProgram(result);
    throw new Error(message);
  }
  return result;
}

const program = createProgram();
const uniforms = {
  texture: gl.getUniformLocation(program, "uTexture"),
  projection: gl.getUniformLocation(program, "uProjection"),
  view: gl.getUniformLocation(program, "uView"),
  focal: gl.getUniformLocation(program, "uFocal"),
  viewport: gl.getUniformLocation(program, "uViewport"),
  splatScale: gl.getUniformLocation(program, "uSplatScale"),
  alphaThreshold: gl.getUniformLocation(program, "uAlphaThreshold"),
};

gl.useProgram(program);
gl.disable(gl.DEPTH_TEST);
gl.enable(gl.BLEND);
gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

const quadBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

const indexBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
gl.enableVertexAttribArray(1);
gl.vertexAttribIPointer(1, 1, gl.INT, 0, 0);
gl.vertexAttribDivisor(1, 1);

const splatTexture = gl.createTexture();
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, splatTexture);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.uniform1i(uniforms.texture, 0);

let worker = null;
let loadGeneration = 0;
let splatCount = 0;
let sortedCount = 0;
let bounds = null;
let sortInFlight = false;
let lastSortRow = null;
let contextLost = false;
let progressHideTimer = null;
let frameCount = 0;
let fpsWindowStart = performance.now();

const orbit = {
  target: [0, 0, 0],
  distance: 10,
  yaw: 0.72,
  pitch: 0.32,
  fullRadius: 100,
};

canvas.addEventListener("webglcontextlost", (event) => {
  event.preventDefault();
  contextLost = true;
  worker?.terminate();
  worker = null;
  showError("GPU コンテキストがリセットされました。復旧後にビューアーを自動再読み込みします。");
});
canvas.addEventListener("webglcontextrestored", () => {
  window.location.reload();
});

function gpuDescription() {
  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = debug
    ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL)
    : gl.getParameter(gl.RENDERER);
  let family = "GPU";
  if (/apple|metal/i.test(renderer)) family = "Mac GPU";
  else if (/nvidia/i.test(renderer)) family = "NVIDIA GPU";
  else if (/amd|radeon/i.test(renderer)) family = "AMD GPU";
  else if (/intel/i.test(renderer)) family = "Intel GPU";
  return `${family} · WebGL2 · ${renderer}`;
}

ui["gpu-info"].textContent = gpuDescription();

function showError(message) {
  ui["loading"]?.classList.add("hidden");
  ui["error-box"].textContent = message;
  ui["error-overlay"].classList.remove("hidden");
}

function clearError() {
  ui["error-overlay"].classList.add("hidden");
}

function setProgress(value, label, detail) {
  if (progressHideTimer !== null) {
    window.clearTimeout(progressHideTimer);
    progressHideTimer = null;
  }
  ui["loading"].classList.remove("hidden");
  ui["progress-bar"].style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
  if (label) ui["progress-label"].textContent = label;
  if (detail) ui["progress-detail"].textContent = detail;
}

function finishProgress() {
  if (progressHideTimer !== null) window.clearTimeout(progressHideTimer);
  const completedGeneration = loadGeneration;
  ui["progress-bar"].style.width = "100%";
  progressHideTimer = window.setTimeout(() => {
    if (completedGeneration === loadGeneration) ui["loading"].classList.add("hidden");
    progressHideTimer = null;
  }, 180);
}

function formatCount(value) {
  return Number.isFinite(value) ? value.toLocaleString("ja-JP") : "—";
}

function formatBytes(value) {
  if (!Number.isFinite(value) || value <= 0) return "";
  const units = ["B", "KiB", "MiB", "GiB"];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toFixed(unit > 1 ? 1 : 0)} ${units[unit]}`;
}

function vec3Subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vec3Cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function vec3Dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function vec3Normalize(value) {
  const length = Math.hypot(value[0], value[1], value[2]) || 1;
  return [value[0] / length, value[1] / length, value[2] / length];
}

// COLMAP-style view matrix: camera +X right, +Y down, +Z forward.  Built as a
// GL lookAt and then flipping the Y and Z rows.
function lookAtCV(eye, center, up) {
  const z = vec3Normalize(vec3Subtract(eye, center));
  const x = vec3Normalize(vec3Cross(up, z));
  const y = vec3Cross(z, x);
  return new Float32Array([
    x[0], -y[0], -z[0], 0,
    x[1], -y[1], -z[1], 0,
    x[2], -y[2], -z[2], 0,
    -vec3Dot(x, eye), vec3Dot(y, eye), vec3Dot(z, eye), 1,
  ]);
}

// Direct3D-style projection for the +Z-forward view space: NDC z in [0, 1],
// w = view-space depth, y flipped back to GL's upward NDC axis.
function perspectiveCV(focalX, focalY, width, height, near, far) {
  return new Float32Array([
    (2 * focalX) / width, 0, 0, 0,
    0, -(2 * focalY) / height, 0, 0,
    0, 0, far / (far - near), 1,
    0, 0, -(far * near) / (far - near), 0,
  ]);
}

function multiplyMatrices(a, b) {
  const output = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let index = 0; index < 4; index += 1) {
        sum += a[index * 4 + row] * b[column * 4 + index];
      }
      output[column * 4 + row] = sum;
    }
  }
  return output;
}

function cameraOffset() {
  const cosPitch = Math.cos(orbit.pitch);
  return [
    orbit.distance * cosPitch * Math.sin(orbit.yaw),
    orbit.distance * Math.sin(orbit.pitch),
    orbit.distance * cosPitch * Math.cos(orbit.yaw),
  ];
}

function boundsRadius(box) {
  if (!box) return 1;
  return Math.max(
    1e-4,
    Math.hypot(box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]) / 2,
  );
}

// The scene mirrors about z=0 when Z-flip is on; orbit.target lives in the
// displayed (possibly flipped) world.
function zSign() {
  return ui["flip-z"].checked ? -1 : 1;
}

function fitBounds(box) {
  if (!box) return;
  orbit.target = [0, 1, 2].map((axis) => (box.min[axis] + box.max[axis]) / 2);
  orbit.target[2] *= zSign();
  const radius = boundsRadius(box);
  const aspect = Math.max(0.25, canvas.clientWidth / Math.max(1, canvas.clientHeight));
  const fovY = Math.PI / 4;
  const fovX = 2 * Math.atan(Math.tan(fovY / 2) * aspect);
  orbit.distance = (radius / Math.sin(Math.min(fovX, fovY) / 2)) * 1.08;
  orbit.yaw = 0.72;
  orbit.pitch = 0.32;
}

function resizeCanvas() {
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(canvas.clientWidth * pixelRatio));
  const height = Math.max(1, Math.round(canvas.clientHeight * pixelRatio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    gl.viewport(0, 0, width, height);
  }
}

function maybeRequestSort(viewProj) {
  if (!worker || sortInFlight || !splatCount) return;
  const row = [viewProj[2], viewProj[6], viewProj[10]];
  if (lastSortRow) {
    const lengthProduct =
      Math.hypot(row[0], row[1], row[2]) * Math.hypot(lastSortRow[0], lastSortRow[1], lastSortRow[2]);
    const dot = vec3Dot(row, lastSortRow) / Math.max(1e-12, lengthProduct);
    if (Math.abs(dot - 1) < 0.001) return;
  }
  sortInFlight = true;
  lastSortRow = row;
  worker.postMessage({ type: "sort", viewProj: Array.from(viewProj) });
}

function render() {
  requestAnimationFrame(render);
  if (contextLost || gl.isContextLost()) return;
  resizeCanvas();

  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (!splatCount) return;

  const focal = (0.5 * canvas.height) / Math.tan(Math.PI / 8);
  const near = Math.max(1e-5, orbit.distance * 0.005);
  const far = orbit.distance * 100 + orbit.fullRadius * 10;
  const projection = perspectiveCV(focal, focal, canvas.width, canvas.height, near, far);
  const eyeOffset = cameraOffset();
  const eye = [0, 1, 2].map((axis) => orbit.target[axis] + eyeOffset[axis]);
  const view = lookAtCV(eye, orbit.target, [0, 1, 0]);
  if (zSign() < 0) {
    // view * diag(1, 1, -1, 1): mirrors world positions AND covariances.
    view[8] = -view[8];
    view[9] = -view[9];
    view[10] = -view[10];
  }
  const viewProj = multiplyMatrices(projection, view);
  maybeRequestSort(viewProj);
  if (!sortedCount) return;

  gl.uniformMatrix4fv(uniforms.projection, false, projection);
  gl.uniformMatrix4fv(uniforms.view, false, view);
  gl.uniform2f(uniforms.focal, focal, focal);
  gl.uniform2f(uniforms.viewport, canvas.width, canvas.height);
  gl.uniform1f(uniforms.splatScale, Number.parseFloat(ui["splat-scale"].value));
  gl.uniform1f(uniforms.alphaThreshold, Number.parseFloat(ui["alpha-threshold"].value));

  gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, sortedCount);

  frameCount += 1;
  const now = performance.now();
  if (now - fpsWindowStart >= 500) {
    ui["fps"].textContent = `${Math.round((frameCount * 1000) / (now - fpsWindowStart))}`;
    frameCount = 0;
    fpsWindowStart = now;
  }
}

function attachWorker(newWorker, generation, fileBytes) {
  newWorker.onmessage = (event) => {
    if (generation !== loadGeneration) {
      newWorker.terminate();
      return;
    }
    const data = event.data;
    if (data.type === "header") {
      ui["splat-count"].textContent = formatCount(data.vertexCount);
    } else if (data.type === "progress") {
      const ratio = data.totalBytes ? data.bytes / data.totalBytes : 0;
      setProgress(
        ratio * 0.97,
        "3DGS PLY を解析しています",
        `${formatBytes(data.bytes)} / ${formatBytes(data.totalBytes)} · ${formatCount(data.count)} ガウシアン`,
      );
    } else if (data.type === "ready") {
      try {
        uploadTexture(data);
        finishProgress();
      } catch (error) {
        showError(error instanceof Error ? error.message : String(error));
      }
    } else if (data.type === "sorted") {
      const depthIndex = new Uint32Array(data.depthIndex);
      gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, depthIndex, gl.DYNAMIC_DRAW);
      sortedCount = depthIndex.length;
      sortInFlight = false;
      ui["sort-time"].textContent = `${data.sortMs} ms`;
    } else if (data.type === "error") {
      showError(data.message);
      newWorker.terminate();
      if (worker === newWorker) {
        worker = null;
        sortInFlight = false;
      }
    }
  };
  newWorker.onerror = (event) => {
    if (generation === loadGeneration) showError(`3DGS worker error: ${event.message}`);
    newWorker.terminate();
    if (worker === newWorker) {
      worker = null;
      sortInFlight = false;
    }
  };
  if (fileBytes) ui["file-size"].textContent = formatBytes(fileBytes);
}

function uploadTexture(data) {
  const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  if (data.texWidth > maxTextureSize || data.texHeight > maxTextureSize) {
    throw new Error(
      `このデータはテクスチャ上限を超えています (${data.texWidth}x${data.texHeight} > ${maxTextureSize}).`,
    );
  }
  gl.bindTexture(gl.TEXTURE_2D, splatTexture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA32UI,
    data.texWidth,
    data.texHeight,
    0,
    gl.RGBA_INTEGER,
    gl.UNSIGNED_INT,
    new Uint32Array(data.texdata),
  );
  const uploadError = gl.getError();
  if (uploadError !== gl.NO_ERROR) {
    throw new Error(`Splat texture upload failed (WebGL error 0x${uploadError.toString(16)}).`);
  }

  splatCount = data.vertexCount;
  sortedCount = 0;
  bounds = data.bounds;
  orbit.fullRadius = boundsRadius(bounds.all);
  fitBounds(bounds.robust || bounds.all);
  lastSortRow = null;
  sortInFlight = false;
  ui["splat-count"].textContent = formatCount(splatCount);
  ui["file-name"].textContent = data.fileName;
}

function startLoad(message, fileBytes, transfer) {
  const generation = ++loadGeneration;
  if (worker) worker.terminate();
  clearError();
  splatCount = 0;
  sortedCount = 0;
  sortInFlight = false;
  lastSortRow = null;
  worker = new Worker("/gs-worker.js");
  attachWorker(worker, generation, fileBytes);
  setProgress(0.02, "3DGS PLY を読み込んでいます", "接続中…");
  worker.postMessage(message, transfer || []);
}

async function loadServerFile() {
  const entryGeneration = loadGeneration;
  let fileBytes = 0;
  let fileName = "pointcloud.ply";
  try {
    const response = await fetch("/metadata.json", { cache: "no-store" });
    if (response.ok) {
      const summary = await response.json();
      fileBytes = summary.fileBytes || 0;
      fileName = summary.fileName || fileName;
    }
  } catch {
    // metadata is optional
  }
  if (entryGeneration !== loadGeneration) return; // a local-file load won the race
  ui["file-name"].textContent = fileName;
  startLoad({ type: "load-url", url: "/pointcloud.ply", fileName }, fileBytes);
}

function loadLocalFile(file) {
  ui["file-name"].textContent = file.name;
  startLoad({ type: "load-file", file }, file.size);
}

ui["background"].addEventListener("input", () => {
  canvas.style.background = ui["background"].value;
});
ui["splat-scale"].addEventListener("input", () => {
  ui["splat-scale-value"].textContent = `${Number.parseFloat(ui["splat-scale"].value).toFixed(2)}×`;
});
ui["alpha-threshold"].addEventListener("input", () => {
  ui["alpha-threshold-value"].textContent = Number.parseFloat(ui["alpha-threshold"].value).toFixed(2);
});
ui["flip-z"].addEventListener("input", () => {
  // Keep the camera on the object: convert the stored target into the new
  // displayed world.
  orbit.target[2] = -orbit.target[2];
});
ui["fit-scene"].addEventListener("click", () => {
  if (bounds) fitBounds(bounds.robust || bounds.all);
});
ui["file-input"].addEventListener("change", () => {
  const [file] = ui["file-input"].files;
  if (file) loadLocalFile(file);
  ui["file-input"].value = ""; // allow re-selecting the same file
});

const pointer = { active: false, id: null, x: 0, y: 0, mode: "orbit" };
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("pointerdown", (event) => {
  pointer.active = true;
  pointer.id = event.pointerId;
  pointer.x = event.clientX;
  pointer.y = event.clientY;
  pointer.mode = event.button === 2 || event.shiftKey ? "pan" : "orbit";
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!pointer.active || event.pointerId !== pointer.id) return;
  const dx = event.clientX - pointer.x;
  const dy = event.clientY - pointer.y;
  pointer.x = event.clientX;
  pointer.y = event.clientY;
  if (pointer.mode === "orbit") {
    orbit.yaw -= dx * 0.006;
    orbit.pitch = Math.max(-1.54, Math.min(1.54, orbit.pitch + dy * 0.006));
  } else {
    const eye = vec3Normalize(cameraOffset());
    const forward = [-eye[0], -eye[1], -eye[2]];
    const right = vec3Normalize(vec3Cross(forward, [0, 1, 0]));
    const up = vec3Normalize(vec3Cross(right, forward));
    const scale = (2 * orbit.distance * Math.tan(Math.PI / 8)) / Math.max(1, canvas.clientHeight);
    for (let axis = 0; axis < 3; axis += 1) {
      orbit.target[axis] += right[axis] * -dx * scale + up[axis] * dy * scale;
    }
  }
});
const finishPointer = (event) => {
  if (event.pointerId === pointer.id) {
    pointer.active = false;
    pointer.id = null;
  }
};
canvas.addEventListener("pointerup", finishPointer);
canvas.addEventListener("pointercancel", finishPointer);
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  orbit.distance *= Math.exp(Math.max(-120, Math.min(120, event.deltaY)) * 0.0015);
  orbit.distance = Math.max(1e-4, Math.min(orbit.fullRadius * 1000, orbit.distance));
}, { passive: false });
canvas.addEventListener("dblclick", () => {
  if (bounds) fitBounds(bounds.robust || bounds.all);
});

canvas.style.background = ui["background"].value;
requestAnimationFrame(render);
loadServerFile();
