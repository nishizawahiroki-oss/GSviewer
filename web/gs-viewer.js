"use strict";

// WebGL2 Gaussian renderer with front-to-back "under" alpha blending.
// This file is a modified browser/GLSL adaptation; it does not contain or run
// the upstream CUDA / OptiX backend.
//
// 2DGS-derived portions:
// Copyright (C) 2023, Inria
// GRAPHDECO research group, https://team.inria.fr/graphdeco
// All rights reserved. Licensed for non-commercial research/evaluation under
// licenses/Gaussian-Splatting-License.md.
//
// ArtiFixer3D-derived portions:
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// The 2DGS path adapts hbb1/diff-surfel-rasterization forward.cu equations
// 8-10 (commit e0ed020), and the ArtiFixer3D path adapts the 3DGUT projection
// and density response from nv-tlabs/3DGRUT-ArtiFixer (commit 62e1038).

const canvas = document.querySelector("#viewport");
const ui = Object.fromEntries(
  [
    "file-name",
    "file-input",
    "splat-count",
    "light-count",
    "light-proxy-count",
    "file-size",
    "sort-time",
    "fps",
    "background",
    "splat-scale",
    "splat-scale-value",
    "alpha-threshold",
    "alpha-threshold-value",
    "lights-enabled",
    "master-intensity",
    "master-intensity-value",
    "fixture-select",
    "fixture-enabled",
    "fixture-intensity",
    "fixture-intensity-value",
    "fixture-color",
    "selection-marker",
    "flip-x",
    "flip-y",
    "flip-z",
    "axis-gizmo",
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

flat out uint vRenderer;
flat out vec4 vColor;
flat out vec3 vCenter;
flat out vec3 vScale;
flat out vec3 vRotation0;
flat out vec3 vRotation1;
flat out vec3 vRotation2;
flat out vec3 vTu;
flat out vec3 vTv;
flat out vec3 vTw;
flat out vec2 vSurfelCenter;
out vec2 vPosition;

const uint STANDARD_3DGS = 1u;
const uint OFFICIAL_2DGS = 2u;
const uint ARTIFIXER_3DGUT = 3u;
const uint LIGHT_PROXY = 4u;
const uint SPLATS_PER_ROW = 682u;

ivec2 splatTexel(uint index, uint offset) {
  uint row = index / SPLATS_PER_ROW;
  uint column = (index - row * SPLATS_PER_ROW) * 3u + offset;
  return ivec2(int(column), int(row));
}

mat3 quaternionToRotation(vec4 quaternion) {
  vec4 q = quaternion / max(length(quaternion), 1.0e-12);
  float w = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;
  return mat3(
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
    2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
    2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)
  );
}

vec2 projectPixel(vec3 point) {
  vec3 camera = (uView * vec4(point, 1.0)).xyz;
  if (camera.z <= 0.0) return vec2(0.0);
  return vec2(
    uFocal.x * camera.x / camera.z + 0.5 * uViewport.x,
    -uFocal.y * camera.y / camera.z + 0.5 * uViewport.y
  );
}

void hideSplat() {
  gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
}

void renderStandard3D(vec3 center, vec3 scale, mat3 rotation, vec4 clipCenter) {
  mat3 scaledRotation = rotation * mat3(
    scale.x, 0.0, 0.0,
    0.0, scale.y, 0.0,
    0.0, 0.0, scale.z
  );
  mat3 covariance = scaledRotation * transpose(scaledRotation);
  vec3 cam = (uView * vec4(center, 1.0)).xyz;
  mat3 J = mat3(
    uFocal.x / cam.z, 0.0, -(uFocal.x * cam.x) / (cam.z * cam.z),
    0.0, -uFocal.y / cam.z, (uFocal.y * cam.y) / (cam.z * cam.z),
    0.0, 0.0, 0.0
  );
  mat3 T = transpose(mat3(uView)) * J;
  mat3 cov2d = 4.0 * transpose(T) * covariance * T;

  float detBefore = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1];
  cov2d[0][0] += 0.1;
  cov2d[1][1] += 0.1;
  float detAfter = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1];
  vColor.a *= sqrt(max(0.0, detBefore / detAfter));

  float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
  float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
  float lambda1 = mid + radius;
  float lambda2 = mid - radius;
  if (lambda2 < 0.0) {
    hideSplat();
    return;
  }
  vec2 eigenDirection = vec2(cov2d[0][1], lambda1 - cov2d[0][0]);
  vec2 diagonalVector = dot(eigenDirection, eigenDirection) < 1.0e-24
    ? vec2(1.0, 0.0)
    : normalize(eigenDirection);
  vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
  vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);
  vPosition = aPosition;
  vec2 projectedCenter = clipCenter.xy / clipCenter.w;
  gl_Position = vec4(
    projectedCenter + aPosition.x * majorAxis / uViewport + aPosition.y * minorAxis / uViewport,
    0.0,
    1.0
  );
}

void renderOfficial2DGS(vec3 center, vec3 scale, mat3 rotation) {
  mat4 worldToClip = uProjection * uView;
  vec4 clipU = worldToClip * vec4(rotation[0] * scale.x, 0.0);
  vec4 clipV = worldToClip * vec4(rotation[1] * scale.y, 0.0);
  vec4 clipC = worldToClip * vec4(center, 1.0);
  vec2 pixelOffset = (uViewport - 1.0) * 0.5;
  vTu = vec3(
    0.5 * uViewport.x * clipU.x + pixelOffset.x * clipU.w,
    0.5 * uViewport.x * clipV.x + pixelOffset.x * clipV.w,
    0.5 * uViewport.x * clipC.x + pixelOffset.x * clipC.w
  );
  vTv = vec3(
    0.5 * uViewport.y * clipU.y + pixelOffset.y * clipU.w,
    0.5 * uViewport.y * clipV.y + pixelOffset.y * clipV.w,
    0.5 * uViewport.y * clipC.y + pixelOffset.y * clipC.w
  );
  vTw = vec3(clipU.w, clipV.w, clipC.w);

  vec3 cutoff = vec3(9.0, 9.0, -1.0);
  float denominator = dot(cutoff, vTw * vTw);
  if (abs(denominator) < 1.0e-10) {
    hideSplat();
    return;
  }
  vec3 f = cutoff / denominator;
  vSurfelCenter = vec2(dot(f, vTu * vTw), dot(f, vTv * vTw));
  vec2 extentSq = vSurfelCenter * vSurfelCenter
    - vec2(dot(f, vTu * vTu), dot(f, vTv * vTv));
  vec2 conicExtent = sqrt(max(vec2(1.0e-4), extentSq));
  float radius = max(max(conicExtent.x, conicExtent.y), 3.0 * 0.707106);
  vec2 extent = vec2(radius);
  vec2 centerNdc = 2.0 * ((vSurfelCenter + 0.5) / uViewport) - 1.0;
  gl_Position = vec4(centerNdc + aPosition * extent / uViewport, 0.0, 1.0);
  vPosition = aPosition;
}

void renderArtiFixer3DGUT(vec3 center, vec3 scale, mat3 rotation) {
  // Official ArtiFixer3D uses 3DGUT with alpha=1, beta=2, kappa=0.
  // Lambda is therefore zero and the six off-center sigma points have 1/6 weight.
  const float delta = 1.7320508075688772;
  vec2 sigma0 = projectPixel(center);
  vec2 sigma1 = projectPixel(center + delta * scale.x * rotation[0]);
  vec2 sigma2 = projectPixel(center + delta * scale.y * rotation[1]);
  vec2 sigma3 = projectPixel(center + delta * scale.z * rotation[2]);
  vec2 sigma4 = projectPixel(center - delta * scale.x * rotation[0]);
  vec2 sigma5 = projectPixel(center - delta * scale.y * rotation[1]);
  vec2 sigma6 = projectPixel(center - delta * scale.z * rotation[2]);
  vec2 projectedCenter = (sigma1 + sigma2 + sigma3 + sigma4 + sigma5 + sigma6) / 6.0;

  vec2 d0 = sigma0 - projectedCenter;
  vec2 d1 = sigma1 - projectedCenter;
  vec2 d2 = sigma2 - projectedCenter;
  vec2 d3 = sigma3 - projectedCenter;
  vec2 d4 = sigma4 - projectedCenter;
  vec2 d5 = sigma5 - projectedCenter;
  vec2 d6 = sigma6 - projectedCenter;
  vec3 covariance = 2.0 * vec3(d0.x * d0.x, d0.x * d0.y, d0.y * d0.y);
  covariance += (vec3(d1.x * d1.x, d1.x * d1.y, d1.y * d1.y)
    + vec3(d2.x * d2.x, d2.x * d2.y, d2.y * d2.y)
    + vec3(d3.x * d3.x, d3.x * d3.y, d3.y * d3.y)
    + vec3(d4.x * d4.x, d4.x * d4.y, d4.y * d4.y)
    + vec3(d5.x * d5.x, d5.x * d5.y, d5.y * d5.y)
    + vec3(d6.x * d6.x, d6.x * d6.y, d6.y * d6.y)) / 6.0;

  vec3 dilated = covariance + vec3(0.3, 0.0, 0.3);
  float determinant = dilated.x * dilated.z - dilated.y * dilated.y;
  float originalDeterminant = covariance.x * covariance.z - covariance.y * covariance.y;
  if (!(determinant > 0.0)) {
    hideSplat();
    return;
  }
  float projectedOpacity = vColor.a * sqrt(max(0.000025, originalDeterminant / determinant));
  if (projectedOpacity < 1.0 / 255.0) {
    hideSplat();
    return;
  }
  float extentFactor = min(3.33, sqrt(2.0 * log(projectedOpacity * 255.0)));
  float mid = 0.5 * (dilated.x + dilated.z);
  float lambda = mid + sqrt(max(0.01, mid * mid - determinant));
  float radius = extentFactor * sqrt(lambda);
  vec2 extent = min(extentFactor * sqrt(vec2(dilated.x, dilated.z)), vec2(radius));
  vec2 centerNdc = 2.0 * (projectedCenter / uViewport) - 1.0;
  gl_Position = vec4(centerNdc + aPosition * extent / uViewport, 0.0, 1.0);
  vPosition = aPosition;
}

void main() {
  uint index = uint(aIndex);
  uvec4 rawCenter = texelFetch(uTexture, splatTexel(index, 0u), 0);
  uvec4 rawScaleColor = texelFetch(uTexture, splatTexel(index, 1u), 0);
  uvec4 rawRotation = texelFetch(uTexture, splatTexel(index, 2u), 0);
  uint packedColorRenderer = rawScaleColor.w;
  vRenderer = packedColorRenderer >> 24;
  vCenter = uintBitsToFloat(rawCenter.xyz);
  vScale = uintBitsToFloat(rawScaleColor.xyz) * uSplatScale;
  vec4 quaternion = uintBitsToFloat(rawRotation);
  mat3 rotation = quaternionToRotation(quaternion);
  vRotation0 = rotation[0];
  vRotation1 = rotation[1];
  vRotation2 = rotation[2];
  vColor = vec4(
    float(packedColorRenderer & 0xffu) / 255.0,
    float((packedColorRenderer >> 8) & 0xffu) / 255.0,
    float((packedColorRenderer >> 16) & 0xffu) / 255.0,
    uintBitsToFloat(rawCenter.w)
  );

  vec4 cameraCenter = uView * vec4(vCenter, 1.0);
  vec4 clipCenter = uProjection * cameraCenter;
  float minimumDepth = (vRenderer == ARTIFIXER_3DGUT || vRenderer == OFFICIAL_2DGS)
    ? 0.2 : 1.0e-5;
  if (cameraCenter.z < minimumDepth || vColor.a < uAlphaThreshold) {
    hideSplat();
    return;
  }
  float clip = 1.2 * clipCenter.w;
  if (clipCenter.z < -clip || clipCenter.x < -clip || clipCenter.x > clip
    || clipCenter.y < -clip || clipCenter.y > clip) {
    hideSplat();
    return;
  }

  if (vRenderer == OFFICIAL_2DGS) {
    renderOfficial2DGS(vCenter, vScale, rotation);
  } else if (vRenderer == ARTIFIXER_3DGUT) {
    renderArtiFixer3DGUT(vCenter, vScale, rotation);
  } else if (vRenderer == STANDARD_3DGS || vRenderer == LIGHT_PROXY) {
    renderStandard3D(vCenter, vScale, rotation, clipCenter);
  } else {
    hideSplat();
  }
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
precision highp int;

uniform mat4 uView;
uniform vec3 uCameraPosition;
uniform vec2 uFocal;
uniform vec2 uViewport;

flat in uint vRenderer;
flat in vec4 vColor;
flat in vec3 vCenter;
flat in vec3 vScale;
flat in vec3 vRotation0;
flat in vec3 vRotation1;
flat in vec3 vRotation2;
flat in vec3 vTu;
flat in vec3 vTv;
flat in vec3 vTw;
flat in vec2 vSurfelCenter;
in vec2 vPosition;
out vec4 fragColor;

const uint STANDARD_3DGS = 1u;
const uint OFFICIAL_2DGS = 2u;
const uint ARTIFIXER_3DGUT = 3u;
const uint LIGHT_PROXY = 4u;

float official2DGSAlpha() {
  vec2 pixel = gl_FragCoord.xy - 0.5;
  vec3 k = pixel.x * vTw - vTu;
  vec3 l = pixel.y * vTw - vTv;
  vec3 intersection = cross(k, l);
  if (abs(intersection.z) < 1.0e-10) return 0.0;
  vec2 surfel = intersection.xy / intersection.z;
  float rho3d = dot(surfel, surfel);
  vec2 centerDelta = vSurfelCenter - pixel;
  float rho2d = 2.0 * dot(centerDelta, centerDelta);
  float rho = min(rho3d, rho2d);
  return min(0.99, vColor.a * exp(-0.5 * rho));
}

float artiFixer3DGUTAlpha() {
  vec3 cameraDirection = normalize(vec3(
    (gl_FragCoord.x - 0.5 * uViewport.x) / uFocal.x,
    -(gl_FragCoord.y - 0.5 * uViewport.y) / uFocal.y,
    1.0
  ));
  vec3 rayDirection = normalize(transpose(mat3(uView)) * cameraDirection);
  mat3 rotation = mat3(vRotation0, vRotation1, vRotation2);
  vec3 inverseScale = 1.0 / max(vScale, vec3(1.0e-10));
  vec3 localOrigin = inverseScale * (transpose(rotation) * (uCameraPosition - vCenter));
  vec3 localDirection = normalize(inverseScale * (transpose(rotation) * rayDirection));
  vec3 closestVector = cross(localDirection, localOrigin);
  float squaredDistance = dot(closestVector, closestVector);
  float response = exp(-0.5 * squaredDistance);
  if (response <= 0.0113) return 0.0;
  return min(0.99, response * vColor.a);
}

void main() {
  float alpha;
  if (vRenderer == OFFICIAL_2DGS) {
    alpha = official2DGSAlpha();
  } else if (vRenderer == ARTIFIXER_3DGUT) {
    alpha = artiFixer3DGUTAlpha();
  } else {
    float power = -dot(vPosition, vPosition);
    if (power < -4.0) discard;
    alpha = min(0.99, exp(power) * vColor.a);
  }
  if (alpha < 1.0 / 255.0) discard;
  fragColor = vec4(alpha * vColor.rgb, alpha);
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
  cameraPosition: gl.getUniformLocation(program, "uCameraPosition"),
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
let textureData = null;
let lightBindings = [];
let lightStates = new Map();
let latestViewProjection = null;

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

function colorToHex(color) {
  return `#${color.map((value) => Math.max(0, Math.min(255, Math.round(value * 255)))
    .toString(16).padStart(2, "0")).join("")}`;
}

function hexToColor(value) {
  return [1, 3, 5].map((offset) => Number.parseInt(value.slice(offset, offset + 2), 16) / 255);
}

function setLightControlAvailability(available) {
  for (const id of [
    "lights-enabled",
    "master-intensity",
    "fixture-select",
    "fixture-enabled",
    "fixture-intensity",
    "fixture-color",
  ]) ui[id].disabled = !available;
}

function selectedLightState() {
  return lightStates.get(ui["fixture-select"].value) || null;
}

function syncSelectedLightControls() {
  const state = selectedLightState();
  if (!state) {
    ui["fixture-intensity-value"].textContent = "—";
    return;
  }
  ui["fixture-enabled"].checked = state.enabled;
  ui["fixture-intensity"].value = String(state.intensity);
  ui["fixture-intensity-value"].textContent = `${state.intensity.toFixed(2)}×`;
  ui["fixture-color"].value = colorToHex(state.color);
}

function projectLightPosition(position) {
  if (!latestViewProjection) return null;
  const matrix = latestViewProjection;
  const clipX = matrix[0] * position[0] + matrix[4] * position[1]
    + matrix[8] * position[2] + matrix[12];
  const clipY = matrix[1] * position[0] + matrix[5] * position[1]
    + matrix[9] * position[2] + matrix[13];
  const clipZ = matrix[2] * position[0] + matrix[6] * position[1]
    + matrix[10] * position[2] + matrix[14];
  const clipW = matrix[3] * position[0] + matrix[7] * position[1]
    + matrix[11] * position[2] + matrix[15];
  if (!(clipW > 0) || clipZ < 0 || clipZ > clipW) return null;
  const bounds = canvas.getBoundingClientRect();
  return {
    x: bounds.left + (clipX / clipW * 0.5 + 0.5) * bounds.width,
    y: bounds.top + (0.5 - clipY / clipW * 0.5) * bounds.height,
    depth: clipZ / clipW,
  };
}

function projectedBindingPoints(binding) {
  const points = [];
  const positions = binding.proxyPositions || [];
  for (let offset = 0; offset < positions.length; offset += 3) {
    const projected = projectLightPosition(positions.slice(offset, offset + 3));
    if (projected) points.push(projected);
  }
  return points;
}

function findLightAt(clientX, clientY, radius = 24) {
  if (!ui["lights-enabled"].checked || Number.parseFloat(ui["master-intensity"].value) <= 0) {
    return null;
  }
  let best = null;
  for (const binding of lightBindings) {
    const state = lightStates.get(binding.fixtureId);
    if (!state?.enabled || state.intensity <= 0) continue;
    for (const point of projectedBindingPoints(binding)) {
      const distance = Math.hypot(point.x - clientX, point.y - clientY);
      if (distance > radius) continue;
      if (!best || distance < best.distance - 1 || (
        Math.abs(distance - best.distance) <= 1 && point.depth < best.depth
      )) best = { binding, distance, depth: point.depth };
    }
  }
  return best?.binding || null;
}

function updateSelectionMarker() {
  const binding = lightBindings.find(
    (item) => item.fixtureId === ui["fixture-select"].value,
  );
  const points = binding ? projectedBindingPoints(binding) : [];
  if (!points.length) {
    ui["selection-marker"].style.display = "none";
    return;
  }
  const xValues = points.map((point) => point.x);
  const yValues = points.map((point) => point.y);
  const padding = 10;
  const left = Math.min(...xValues) - padding;
  const top = Math.min(...yValues) - padding;
  const width = Math.max(22, Math.max(...xValues) - Math.min(...xValues) + padding * 2);
  const height = Math.max(22, Math.max(...yValues) - Math.min(...yValues) + padding * 2);
  Object.assign(ui["selection-marker"].style, {
    display: "block",
    left: `${left}px`,
    top: `${top}px`,
    width: `${width}px`,
    height: `${height}px`,
  });
}

function selectLight(binding) {
  if (!binding) return false;
  ui["fixture-select"].value = binding.fixtureId;
  syncSelectedLightControls();
  updateSelectionMarker();
  return true;
}

function packedLightColor(state) {
  const enabled = ui["lights-enabled"].checked && state.enabled;
  const intensity = enabled
    ? state.intensity * Number.parseFloat(ui["master-intensity"].value) : 0;
  const channels = state.color.map((value) => Math.max(0, Math.min(255, Math.round(value * intensity * 255))));
  return ((4 << 24) | (channels[2] << 16) | (channels[1] << 8) | channels[0]) >>> 0;
}

function updateLightTexture(binding) {
  if (!binding) return;
  const state = lightStates.get(binding.fixtureId);
  if (!state || !textureData) return;
  const packedColor = packedLightColor(state);
  const visible = ui["lights-enabled"].checked && state.enabled
    && state.intensity * Number.parseFloat(ui["master-intensity"].value) > 0.0001;
  const textureFloats = new Float32Array(textureData.buffer);
  gl.bindTexture(gl.TEXTURE_2D, splatTexture);
  for (let index = binding.proxyStart; index < binding.proxyStart + binding.proxyCount; index += 1) {
    const base = index * 12;
    textureFloats[base + 3] = visible ? 1 : 0;
    textureData[base + 7] = packedColor;
    const x = (index % 682) * 3;
    const y = Math.floor(index / 682);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      x,
      y,
      1,
      1,
      gl.RGBA_INTEGER,
      gl.UNSIGNED_INT,
      textureData.subarray(base, base + 4),
    );
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      x + 1,
      y,
      1,
      1,
      gl.RGBA_INTEGER,
      gl.UNSIGNED_INT,
      textureData.subarray(base + 4, base + 8),
    );
  }
}

function updateAllLightTextures() {
  for (const binding of lightBindings) updateLightTexture(binding);
}

function initializeLightControls(bindings) {
  lightBindings = bindings || [];
  lightStates = new Map();
  ui["fixture-select"].replaceChildren();
  for (const binding of lightBindings) {
    lightStates.set(binding.fixtureId, {
      enabled: true,
      intensity: binding.defaultIntensity,
      color: binding.baseColor.map((value) => Math.max(0, Math.min(1, value))),
    });
    const option = document.createElement("option");
    option.value = binding.fixtureId;
    option.textContent = `${binding.fixtureId} · ${binding.fixtureType}`;
    ui["fixture-select"].append(option);
  }
  ui["light-count"].textContent = formatCount(lightBindings.length);
  ui["light-proxy-count"].textContent = formatCount(
    lightBindings.reduce((sum, binding) => sum + binding.proxyCount, 0),
  );
  setLightControlAvailability(lightBindings.length > 0);
  syncSelectedLightControls();
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

// The scene mirrors about the flipped planes; orbit.target lives in the
// displayed (possibly flipped) world.
function flipSigns() {
  return [
    ui["flip-x"].checked ? -1 : 1,
    ui["flip-y"].checked ? -1 : 1,
    ui["flip-z"].checked ? -1 : 1,
  ];
}

function fitBounds(box) {
  if (!box) return;
  const signs = flipSigns();
  orbit.target = [0, 1, 2].map((axis) => signs[axis] * (box.min[axis] + box.max[axis]) / 2);
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
  // view * diag(sx, sy, sz, 1): mirrors world positions AND covariances.
  const signs = flipSigns();
  for (let axis = 0; axis < 3; axis += 1) {
    if (signs[axis] < 0) {
      view[axis * 4] = -view[axis * 4];
      view[axis * 4 + 1] = -view[axis * 4 + 1];
      view[axis * 4 + 2] = -view[axis * 4 + 2];
    }
  }
  drawAxisGizmo(view);
  const viewProj = multiplyMatrices(projection, view);
  latestViewProjection = viewProj;
  updateSelectionMarker();
  maybeRequestSort(viewProj);
  if (!sortedCount) return;

  gl.uniformMatrix4fv(uniforms.projection, false, projection);
  gl.uniformMatrix4fv(uniforms.view, false, view);
  gl.uniform3f(
    uniforms.cameraPosition,
    signs[0] * eye[0],
    signs[1] * eye[1],
    signs[2] * eye[2],
  );
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
      ui["light-proxy-count"].textContent = formatCount(data.lightProxyCount);
      ui["file-name"].textContent = `${data.fileName} · ${data.gaussianKind}`;
    } else if (data.type === "progress") {
      const ratio = data.totalBytes ? data.bytes / data.totalBytes : 0;
      setProgress(
        ratio * 0.97,
        "Gaussian PLY を解析しています",
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
  textureData = new Uint32Array(data.texdata);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA32UI,
    data.texWidth,
    data.texHeight,
    0,
    gl.RGBA_INTEGER,
    gl.UNSIGNED_INT,
    textureData,
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
  ui["splat-count"].textContent = formatCount(data.sceneVertexCount || splatCount);
  ui["file-name"].textContent = `${data.fileName} · ${data.gaussianKind}`;
  initializeLightControls(data.lightBindings);
}

function startLoad(message, fileBytes, transfer) {
  const generation = ++loadGeneration;
  if (worker) worker.terminate();
  clearError();
  splatCount = 0;
  sortedCount = 0;
  sortInFlight = false;
  lastSortRow = null;
  textureData = null;
  latestViewProjection = null;
  initializeLightControls([]);
  worker = new Worker("/gs-worker.js");
  attachWorker(worker, generation, fileBytes);
  setProgress(0.02, "Gaussian PLY を読み込んでいます", "接続中…");
  worker.postMessage(message, transfer || []);
}

async function loadServerFile() {
  const entryGeneration = loadGeneration;
  let fileBytes = 0;
  let fileName = "pointcloud.ply";
  let lightSources = null;
  let rendererHint = null;
  try {
    const response = await fetch("/metadata.json", { cache: "no-store" });
    if (response.ok) {
      const summary = await response.json();
      fileBytes = summary.fileBytes || 0;
      fileName = summary.fileName || fileName;
      rendererHint = summary.gaussianKind || null;
      if (summary.lightPackage?.available) {
        const registryResponse = await fetch(summary.lightPackage.registryUrl, { cache: "no-store" });
        if (!registryResponse.ok) throw new Error(`光源 JSON の取得に失敗しました: HTTP ${registryResponse.status}`);
        lightSources = await registryResponse.json();
      }
    }
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
    return;
  }
  if (entryGeneration !== loadGeneration) return; // a local-file load won the race
  ui["file-name"].textContent = fileName;
  startLoad({
    type: "load-url",
    url: "/pointcloud.ply",
    fileName,
    lightSources,
    rendererHint,
  }, fileBytes);
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
ui["lights-enabled"].addEventListener("input", updateAllLightTextures);
ui["master-intensity"].addEventListener("input", () => {
  ui["master-intensity-value"].textContent = `${Number.parseFloat(ui["master-intensity"].value).toFixed(2)}×`;
  updateAllLightTextures();
});
ui["fixture-select"].addEventListener("change", syncSelectedLightControls);
ui["fixture-enabled"].addEventListener("input", () => {
  const state = selectedLightState();
  if (!state) return;
  state.enabled = ui["fixture-enabled"].checked;
  updateLightTexture(lightBindings.find((binding) => binding.fixtureId === ui["fixture-select"].value));
});
ui["fixture-intensity"].addEventListener("input", () => {
  const state = selectedLightState();
  if (!state) return;
  state.intensity = Number.parseFloat(ui["fixture-intensity"].value);
  ui["fixture-intensity-value"].textContent = `${state.intensity.toFixed(2)}×`;
  updateLightTexture(lightBindings.find((binding) => binding.fixtureId === ui["fixture-select"].value));
});
ui["fixture-color"].addEventListener("input", () => {
  const state = selectedLightState();
  if (!state) return;
  state.color = hexToColor(ui["fixture-color"].value);
  updateLightTexture(lightBindings.find((binding) => binding.fixtureId === ui["fixture-select"].value));
});
for (const [axis, id] of [[0, "flip-x"], [1, "flip-y"], [2, "flip-z"]]) {
  ui[id].addEventListener("input", () => {
    // Keep the camera on the object: convert the stored target into the new
    // displayed world.
    orbit.target[axis] = -orbit.target[axis];
  });
}

// Corner gizmo: shows where the data's +X/+Y/+Z axes point on screen,
// including the current axis flips (view columns already carry the signs).
const gizmoContext = ui["axis-gizmo"].getContext("2d");

function drawAxisGizmo(view) {
  const context = gizmoContext;
  if (!context) return;
  const size = ui["axis-gizmo"].width;
  const center = size / 2;
  const reach = size * 0.36;
  context.clearRect(0, 0, size, size);

  const axes = [
    { label: "X", color: "#ff5a52", camera: [view[0], view[1], view[2]] },
    { label: "Y", color: "#62dda1", camera: [view[4], view[5], view[6]] },
    { label: "Z", color: "#65b8ff", camera: [view[8], view[9], view[10]] },
  ];
  // Camera space is +X right, +Y down, +Z forward, so screen = (x, y) directly;
  // draw far-pointing axes first so near ones overlay them.
  axes.sort((a, b) => b.camera[2] - a.camera[2]);

  for (const axis of axes) {
    const tipX = center + axis.camera[0] * reach;
    const tipY = center + axis.camera[1] * reach;
    const towardViewer = axis.camera[2] <= 0;
    context.globalAlpha = towardViewer ? 1 : 0.45;
    context.strokeStyle = axis.color;
    context.fillStyle = axis.color;
    context.lineWidth = 4;
    context.beginPath();
    context.moveTo(center, center);
    context.lineTo(tipX, tipY);
    context.stroke();
    context.beginPath();
    context.arc(tipX, tipY, 10, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "#10131a";
    context.font = "bold 13px ui-sans-serif, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(axis.label, tipX, tipY + 0.5);
  }
  context.globalAlpha = 1;
}
ui["fit-scene"].addEventListener("click", () => {
  if (bounds) fitBounds(bounds.robust || bounds.all);
});
ui["file-input"].addEventListener("change", () => {
  const [file] = ui["file-input"].files;
  if (file) loadLocalFile(file);
  ui["file-input"].value = ""; // allow re-selecting the same file
});

const pointer = {
  active: false,
  id: null,
  x: 0,
  y: 0,
  startX: 0,
  startY: 0,
  moved: false,
  button: 0,
  mode: "orbit",
};
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("pointerdown", (event) => {
  pointer.active = true;
  pointer.id = event.pointerId;
  pointer.x = event.clientX;
  pointer.y = event.clientY;
  pointer.startX = event.clientX;
  pointer.startY = event.clientY;
  pointer.moved = false;
  pointer.button = event.button;
  pointer.mode = event.button === 2 || event.shiftKey ? "pan" : "orbit";
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!pointer.active || event.pointerId !== pointer.id) {
    canvas.style.cursor = findLightAt(event.clientX, event.clientY, 18) ? "pointer" : "default";
    return;
  }
  const dx = event.clientX - pointer.x;
  const dy = event.clientY - pointer.y;
  if (Math.hypot(event.clientX - pointer.startX, event.clientY - pointer.startY) > 4) {
    pointer.moved = true;
  }
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
    if (!pointer.moved && pointer.button === 0 && pointer.mode === "orbit") {
      selectLight(findLightAt(event.clientX, event.clientY));
    }
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
canvas.addEventListener("dblclick", (event) => {
  if (selectLight(findLightAt(event.clientX, event.clientY))) return;
  if (bounds) fitBounds(bounds.robust || bounds.all);
});

canvas.style.background = ui["background"].value;
requestAnimationFrame(render);
loadServerFile();
