"use strict";

const canvas = document.querySelector("#viewport");
const ui = Object.fromEntries(
  [
    "file-name",
    "file-input",
    "scene-count",
    "camera-count",
    "camera-point-count",
    "trajectory-count",
    "show-scene",
    "show-cameras",
    "show-trajectory",
    "camera-xray",
    "scene-size",
    "scene-size-value",
    "camera-size",
    "camera-size-value",
    "background",
    "camera-index",
    "camera-index-value",
    "camera-selection-section",
    "fit-scene",
    "fit-cameras",
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
  alpha: false,
  antialias: false,
  depth: true,
  powerPreference: "high-performance",
  preserveDrawingBuffer: false,
});

if (!gl) {
  showError("WebGL2 を利用できません。GPU アクセラレーションを有効にした Chrome / Safari / Firefox を使用してください。");
  throw new Error("WebGL2 is unavailable.");
}

const VERTEX_SHADER = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aColor;

uniform mat4 uViewProjection;
uniform vec3 uOrigin;
uniform float uPointSize;
uniform float uBrightness;

out vec3 vColor;

void main() {
  gl_Position = uViewProjection * vec4(aPosition - uOrigin, 1.0);
  gl_PointSize = uPointSize;
  vColor = aColor * uBrightness;
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec3 vColor;
uniform float uRoundPoint;
out vec4 outColor;

void main() {
  if (uRoundPoint > 0.5) {
    vec2 point = gl_PointCoord * 2.0 - 1.0;
    float radiusSquared = dot(point, point);
    if (radiusSquared > 0.94) discard;
  }
  outColor = vec4(vColor, 1.0);
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
  viewProjection: gl.getUniformLocation(program, "uViewProjection"),
  origin: gl.getUniformLocation(program, "uOrigin"),
  pointSize: gl.getUniformLocation(program, "uPointSize"),
  brightness: gl.getUniformLocation(program, "uBrightness"),
  roundPoint: gl.getUniformLocation(program, "uRoundPoint"),
};

let positionBuffer = null;
let colorBuffer = null;
let metadata = null;
let activeWorker = null;
let loadGeneration = 0;
let renderPending = false;
let lastFitTarget = "scene";
let progressHideTimer = null;
let contextLost = false;

canvas.addEventListener("webglcontextlost", (event) => {
  event.preventDefault();
  contextLost = true;
  activeWorker?.terminate();
  activeWorker = null;
  showError("GPU コンテキストがリセットされました。復旧後にビューアーを自動再読み込みします。");
});
canvas.addEventListener("webglcontextrestored", () => {
  window.location.reload();
});

const orbit = {
  target: [0, 0, 0],
  distance: 10,
  yaw: 0.72,
  pitch: 0.32,
  fullRadius: 100,
};

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
  if (!Number.isFinite(value)) return "";
  const units = ["B", "KiB", "MiB", "GiB"];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toFixed(unit > 1 ? 1 : 0)} ${units[unit]}`;
}

function readHexColor(value) {
  return [
    Number.parseInt(value.slice(1, 3), 16) / 255,
    Number.parseInt(value.slice(3, 5), 16) / 255,
    Number.parseInt(value.slice(5, 7), 16) / 255,
  ];
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

function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  const rangeInv = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (near + far) * rangeInv, -1,
    0, 0, near * far * 2 * rangeInv, 0,
  ]);
}

function lookAt(eye, center, up) {
  const z = vec3Normalize(vec3Subtract(eye, center));
  const x = vec3Normalize(vec3Cross(up, z));
  const y = vec3Cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -vec3Dot(x, eye), -vec3Dot(y, eye), -vec3Dot(z, eye), 1,
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

function boundsRadius(bounds) {
  if (!bounds) return 1;
  return Math.max(
    1e-4,
    Math.hypot(
      bounds.max[0] - bounds.min[0],
      bounds.max[1] - bounds.min[1],
      bounds.max[2] - bounds.min[2],
    ) / 2,
  );
}

function fitBounds(bounds, targetName) {
  if (!bounds) return;
  orbit.target = [0, 1, 2].map((axis) => (bounds.min[axis] + bounds.max[axis]) / 2);
  const radius = boundsRadius(bounds);
  const aspect = Math.max(0.25, canvas.clientWidth / Math.max(1, canvas.clientHeight));
  const fovY = Math.PI / 4;
  const fovX = 2 * Math.atan(Math.tan(fovY / 2) * aspect);
  orbit.distance = (radius / Math.sin(Math.min(fovX, fovY) / 2)) * 1.08;
  orbit.yaw = 0.72;
  orbit.pitch = 0.32;
  lastFitTarget = targetName;
  requestRender();
}

function requestRender() {
  if (renderPending) return;
  renderPending = true;
  requestAnimationFrame(render);
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

function bindGeometry() {
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.UNSIGNED_BYTE, true, 0, 0);
}

function drawPoints(start, count, size) {
  if (count <= 0) return;
  gl.uniform1f(uniforms.pointSize, size * Math.min(window.devicePixelRatio || 1, 2));
  gl.uniform1f(uniforms.roundPoint, 1);
  gl.drawArrays(gl.POINTS, start, count);
}

function drawLineStrip(start, count) {
  if (count <= 1) return;
  gl.uniform1f(uniforms.roundPoint, 0);
  gl.drawArrays(gl.LINE_STRIP, start, count);
}

function render() {
  renderPending = false;
  if (contextLost || gl.isContextLost()) return;
  resizeCanvas();
  const background = readHexColor(ui.background.value);
  gl.clearColor(background[0], background[1], background[2], 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  if (!metadata || !positionBuffer || !colorBuffer) return;

  const aspect = canvas.width / Math.max(1, canvas.height);
  const eye = cameraOffset();
  const view = lookAt(eye, [0, 0, 0], [0, 1, 0]);
  const near = Math.max(0.0005, orbit.distance * 0.0001);
  const far = Math.max(orbit.distance * 10, orbit.distance + orbit.fullRadius * 5);
  const projection = perspective(Math.PI / 4, aspect, near, far);

  gl.useProgram(program);
  gl.uniformMatrix4fv(uniforms.viewProjection, false, multiplyMatrices(projection, view));
  gl.uniform3fv(uniforms.origin, orbit.target);
  gl.uniform1f(uniforms.brightness, 1);
  bindGeometry();
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  gl.disable(gl.BLEND);

  const layout = metadata.layout;
  if (ui["show-scene"].checked) {
    drawPoints(0, layout.sceneCount, Number.parseFloat(ui["scene-size"].value));
  }

  const hasCameraDrawing = layout.detected
    && (ui["show-cameras"].checked || ui["show-trajectory"].checked);
  if (hasCameraDrawing && ui["camera-xray"].checked) gl.disable(gl.DEPTH_TEST);

  if (layout.detected && ui["show-cameras"].checked) {
    const selected = Number.parseInt(ui["camera-index"].value, 10);
    if (selected < 0) {
      drawPoints(layout.cameraStart, layout.cameraCount, Number.parseFloat(ui["camera-size"].value));
    } else {
      for (const run of layout.runs) {
        drawPoints(
          run.start + selected * layout.samplesPerFrustum,
          layout.samplesPerFrustum,
          Number.parseFloat(ui["camera-size"].value),
        );
      }
    }
  }

  if (layout.detected && ui["show-trajectory"].checked) {
    drawLineStrip(layout.trajectoryStart, layout.trajectoryCount);
    drawPoints(
      layout.trajectoryStart,
      layout.trajectoryCount,
      Math.max(1.5, Number.parseFloat(ui["camera-size"].value) * 0.65),
    );
  }
  gl.enable(gl.DEPTH_TEST);
}

function uploadGeometry(positionsBuffer, colorsBuffer, nextMetadata) {
  setProgress(0.96, "GPU に転送しています", "点群バッファを作成中…");
  const nextPositionBuffer = gl.createBuffer();
  const nextColorBuffer = gl.createBuffer();
  if (!nextPositionBuffer || !nextColorBuffer) {
    if (nextPositionBuffer) gl.deleteBuffer(nextPositionBuffer);
    if (nextColorBuffer) gl.deleteBuffer(nextColorBuffer);
    throw new Error("GPU buffer allocation failed. The WebGL context may be unavailable.");
  }

  try {
    gl.bindBuffer(gl.ARRAY_BUFFER, nextPositionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positionsBuffer), gl.STATIC_DRAW);
    let uploadError = gl.getError();
    if (uploadError !== gl.NO_ERROR) {
      throw new Error(`Position upload failed (WebGL error 0x${uploadError.toString(16)}).`);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, nextColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(colorsBuffer), gl.STATIC_DRAW);
    uploadError = gl.getError();
    if (uploadError !== gl.NO_ERROR) {
      throw new Error(`Color upload failed (WebGL error 0x${uploadError.toString(16)}).`);
    }
  } catch (error) {
    gl.deleteBuffer(nextPositionBuffer);
    gl.deleteBuffer(nextColorBuffer);
    throw error;
  }

  const previousPositionBuffer = positionBuffer;
  const previousColorBuffer = colorBuffer;
  positionBuffer = nextPositionBuffer;
  colorBuffer = nextColorBuffer;
  metadata = nextMetadata;
  if (previousPositionBuffer) gl.deleteBuffer(previousPositionBuffer);
  if (previousColorBuffer) gl.deleteBuffer(previousColorBuffer);
  orbit.fullRadius = boundsRadius(metadata.bounds.all);
  updateDataUi();
  fitBounds(metadata.bounds.sceneRobust || metadata.bounds.scene || metadata.bounds.all, "scene");
  finishProgress();
}

function updateDataUi() {
  const layout = metadata.layout;
  ui["file-name"].textContent = `${metadata.fileName} · ${formatBytes(metadata.fileBytes)}`;
  ui["scene-count"].textContent = formatCount(layout.sceneCount);
  ui["camera-count"].textContent = layout.detected ? formatCount(layout.viewCount) : "未検出";
  ui["camera-point-count"].textContent = layout.detected ? formatCount(layout.cameraCount) : "—";
  ui["trajectory-count"].textContent = layout.detected ? formatCount(layout.trajectoryCount) : "—";
  ui["camera-index"].max = layout.detected ? String(layout.viewCount - 1) : "-1";
  ui["camera-index"].value = "-1";
  ui["camera-index"].disabled = !layout.detected;
  ui["fit-cameras"].disabled = !layout.detected;
  ui["camera-selection-section"].classList.toggle("hidden", !layout.detected);
  updateCameraIndexLabel();
}

function updateCameraIndexLabel() {
  if (!metadata?.layout.detected) {
    ui["camera-index-value"].textContent = "—";
    return;
  }
  const index = Number.parseInt(ui["camera-index"].value, 10);
  ui["camera-index-value"].textContent = index < 0
    ? "すべて"
    : `${index + 1} / ${metadata.layout.viewCount}`;
}

async function fetchArrayBufferWithProgress(url, generation) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`PLY request failed: HTTP ${response.status}`);
  const total = Number.parseInt(response.headers.get("content-length") || "0", 10);
  if (!response.body || !total) return response.arrayBuffer();

  const output = new Uint8Array(total);
  const reader = response.body.getReader();
  let offset = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (generation !== loadGeneration) {
      await reader.cancel();
      throw new Error("Loading superseded by another file.");
    }
    if (done) break;
    output.set(value, offset);
    offset += value.byteLength;
    setProgress(
      (offset / total) * 0.48,
      "PLY を読み込んでいます",
      `${formatBytes(offset)} / ${formatBytes(total)}`,
    );
  }
  if (offset !== total) throw new Error(`PLY download ended early (${offset} / ${total} bytes).`);
  return output.buffer;
}

function parseInWorker(buffer, fileName, generation) {
  if (activeWorker) activeWorker.terminate();
  const worker = new Worker("/ply-worker.js");
  activeWorker = worker;
  worker.onmessage = (event) => {
    if (generation !== loadGeneration) {
      worker.terminate();
      if (activeWorker === worker) activeWorker = null;
      return;
    }
    if (event.data.type === "progress") {
      setProgress(
        0.5 + event.data.value * 0.43,
        "PLY を解析しています",
        `${Math.round(event.data.value * 100)}% · 座標と RGB を展開中`,
      );
    } else if (event.data.type === "result") {
      try {
        uploadGeometry(event.data.positions, event.data.colors, event.data.metadata);
        worker.terminate();
        if (activeWorker === worker) activeWorker = null;
      } catch (error) {
        worker.terminate();
        if (activeWorker === worker) activeWorker = null;
        showError(error instanceof Error ? error.message : String(error));
      }
    } else if (event.data.type === "error") {
      showError(event.data.message);
      worker.terminate();
      if (activeWorker === worker) activeWorker = null;
    }
  };
  worker.onerror = (event) => {
    if (generation === loadGeneration) showError(`PLY worker error: ${event.message}`);
    worker.terminate();
    if (activeWorker === worker) activeWorker = null;
  };
  setProgress(0.5, "PLY を解析しています", `${formatBytes(buffer.byteLength)} を処理します`);
  worker.postMessage({ type: "parse", buffer, fileName }, [buffer]);
}

async function loadServerFile() {
  const generation = ++loadGeneration;
  if (activeWorker) {
    activeWorker.terminate();
    activeWorker = null;
  }
  clearError();
  try {
    const summaryResponse = await fetch("/metadata.json", { cache: "no-store" });
    const summary = summaryResponse.ok ? await summaryResponse.json() : null;
    if (summary?.fileName) ui["file-name"].textContent = summary.fileName;
    const buffer = await fetchArrayBufferWithProgress("/pointcloud.ply", generation);
    if (generation !== loadGeneration) return;
    parseInWorker(buffer, summary?.fileName || "pointcloud.ply", generation);
  } catch (error) {
    if (generation === loadGeneration && !String(error).includes("superseded")) {
      showError(error instanceof Error ? error.message : String(error));
    }
  }
}

async function loadLocalFile(file) {
  const generation = ++loadGeneration;
  if (activeWorker) {
    activeWorker.terminate();
    activeWorker = null;
  }
  clearError();
  ui["file-name"].textContent = file.name;
  try {
    setProgress(0.12, "ローカル PLY を読み込んでいます", formatBytes(file.size));
    const buffer = await file.arrayBuffer();
    if (generation !== loadGeneration) return;
    parseInWorker(buffer, file.name, generation);
  } catch (error) {
    if (generation === loadGeneration) showError(error instanceof Error ? error.message : String(error));
  }
}

for (const id of ["show-scene", "show-cameras", "show-trajectory", "camera-xray", "background"]) {
  ui[id].addEventListener("input", requestRender);
}
ui["scene-size"].addEventListener("input", () => {
  ui["scene-size-value"].textContent = `${Number.parseFloat(ui["scene-size"].value).toFixed(1)} px`;
  requestRender();
});
ui["camera-size"].addEventListener("input", () => {
  ui["camera-size-value"].textContent = `${Number.parseFloat(ui["camera-size"].value).toFixed(1)} px`;
  requestRender();
});
ui["camera-index"].addEventListener("input", () => {
  updateCameraIndexLabel();
  requestRender();
});
ui["fit-scene"].addEventListener("click", () => {
  if (metadata) fitBounds(metadata.bounds.sceneRobust || metadata.bounds.scene, "scene");
});
ui["fit-cameras"].addEventListener("click", () => {
  if (metadata?.layout.detected) fitBounds(metadata.bounds.cameras, "cameras");
});
ui["file-input"].addEventListener("change", () => {
  const [file] = ui["file-input"].files;
  if (file) loadLocalFile(file);
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
  requestRender();
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
  requestRender();
}, { passive: false });
canvas.addEventListener("dblclick", () => {
  if (!metadata) return;
  const bounds = lastFitTarget === "cameras"
    ? metadata.bounds.cameras
    : metadata.bounds.sceneRobust || metadata.bounds.scene;
  fitBounds(bounds, lastFitTarget);
});

window.addEventListener("resize", requestRender);
requestRender();
loadServerFile();
