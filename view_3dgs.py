#!/usr/bin/env python3
"""Serve a dependency-free WebGL2 viewer for a PLY or light package.

The target PLY stores the 3DGS centers first and appends sampled training-camera
frustums plus a trajectory.  The browser performs the heavy vertex decoding and
GPU upload; mesh PLY faces are validated and their vertices are shown by the
point viewer.  This module validates the file and serves it from localhost.
"""

from __future__ import annotations

import argparse
import json
import mmap
import mimetypes
import re
import sys
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO
from urllib.parse import unquote, urlsplit


DEFAULT_PLY = Path("/Users/hiroki/point_cloud_iteration_100000_with_train_views.ply")
WEB_ROOT = Path(__file__).resolve().with_name("web")
MAX_HEADER_BYTES = 1024 * 1024
PACKAGE_PLY_NAME = "gaussian_without_light_sources.ply"
MESH_PACKAGE_PLY_NAME = "mesh_ours_2pivots_post.ply"
PACKAGE_PLY_NAMES = (PACKAGE_PLY_NAME, MESH_PACKAGE_PLY_NAME)
LIGHT_SOURCES_NAME = "light_sources.json"

PLY_TYPE_SIZES = {
    "char": 1,
    "int8": 1,
    "uchar": 1,
    "uint8": 1,
    "short": 2,
    "int16": 2,
    "ushort": 2,
    "uint16": 2,
    "int": 4,
    "int32": 4,
    "uint": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}

CAMERA_RUNS = (
    ("back", (115, 217, 64)),
    ("down", (25, 217, 199)),
    ("front", (26, 140, 255)),
    ("left", (255, 196, 32)),
    ("right", (255, 77, 56)),
    ("up", (190, 90, 255)),
)
TRAJECTORY_COLOR = (255, 255, 255)


class PlyError(ValueError):
    """Raised when the input is not a supported PLY."""


class LightPackageError(ValueError):
    """Raised when a relighting package is missing or inconsistent."""


@dataclass(frozen=True)
class PlyProperty:
    name: str
    data_type: str
    offset: int

    @property
    def size(self) -> int:
        return PLY_TYPE_SIZES[self.data_type]


@dataclass(frozen=True)
class PlyElement:
    name: str
    count: int
    record_bytes: int
    variable_length: bool = False


@dataclass(frozen=True)
class PlyHeader:
    path: Path
    header_bytes: int
    vertex_count: int
    record_bytes: int
    properties: tuple[PlyProperty, ...]
    comments: tuple[str, ...]
    elements: tuple[PlyElement, ...]

    def property(self, name: str) -> PlyProperty | None:
        return next((item for item in self.properties if item.name == name), None)

    def element(self, name: str) -> PlyElement | None:
        return next((item for item in self.elements if item.name == name), None)


@dataclass(frozen=True)
class CameraRun:
    name: str
    color: tuple[int, int, int]
    start: int
    count: int


@dataclass(frozen=True)
class CameraLayout:
    detected: bool
    scene_count: int
    trajectory_start: int
    trajectory_count: int
    view_count: int | None
    samples_per_frustum: int | None
    runs: tuple[CameraRun, ...]
    reason: str | None = None

    @classmethod
    def not_detected(cls, vertex_count: int, reason: str) -> "CameraLayout":
        return cls(
            detected=False,
            scene_count=vertex_count,
            trajectory_start=vertex_count,
            trajectory_count=0,
            view_count=None,
            samples_per_frustum=None,
            runs=(),
            reason=reason,
        )


def _read_header_bytes(stream: BinaryIO) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        line = stream.readline()
        if not line:
            raise PlyError("PLY header is incomplete (missing end_header).")
        chunks.append(line)
        total += len(line)
        if total > MAX_HEADER_BYTES:
            raise PlyError("PLY header exceeds the 1 MiB safety limit.")
        if line.rstrip(b"\r\n") == b"end_header":
            return b"".join(chunks)


def read_ply_header(path: Path) -> PlyHeader:
    path = path.expanduser().resolve()
    try:
        with path.open("rb") as stream:
            raw_header = _read_header_bytes(stream)
    except OSError as exc:
        raise PlyError(f"Cannot open PLY: {path}: {exc}") from exc

    try:
        lines = raw_header.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise PlyError("PLY header must be ASCII.") from exc

    if not lines or lines[0].strip() != "ply":
        raise PlyError("The selected file does not start with a PLY header.")
    if "format binary_little_endian 1.0" not in (line.strip() for line in lines):
        raise PlyError("Only binary_little_endian PLY 1.0 is supported.")

    elements: list[dict[str, object]] = []
    current_element: dict[str, object] | None = None
    vertex_properties: list[PlyProperty] = []
    comments: list[str] = []

    for line in lines[1:]:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "comment":
            comments.append(line.partition("comment")[2].strip())
        elif parts[0] == "element" and len(parts) == 3:
            try:
                count = int(parts[2])
            except ValueError as exc:
                raise PlyError(f"Invalid element count: {line}") from exc
            if count < 0:
                raise PlyError(f"Invalid negative element count: {line}")
            if parts[1] == "vertex" and any(item["name"] == "vertex" for item in elements):
                    raise PlyError("Multiple vertex elements are not supported.")
            current_element = {
                "name": parts[1],
                "count": count,
                "record_bytes": 0,
                "variable_length": False,
            }
            elements.append(current_element)
        elif parts[0] == "property":
            if current_element is None:
                raise PlyError(f"Property appears before an element: {line}")
            if len(parts) >= 2 and parts[1] == "list":
                if (
                    len(parts) != 5
                    or parts[2] not in PLY_TYPE_SIZES
                    or parts[3] not in PLY_TYPE_SIZES
                    or current_element["name"] != "face"
                ):
                    raise PlyError(f"Unsupported list property: {line}")
                current_element["variable_length"] = True
                continue
            if len(parts) != 3 or parts[1] not in PLY_TYPE_SIZES:
                raise PlyError(f"Unsupported property: {line}")
            offset = int(current_element["record_bytes"])
            if current_element["name"] == "vertex":
                vertex_properties.append(PlyProperty(parts[2], parts[1], offset))
            current_element["record_bytes"] = offset + PLY_TYPE_SIZES[parts[1]]

    vertex_element = next((item for item in elements if item["name"] == "vertex"), None)
    if vertex_element is None:
        raise PlyError("PLY has no vertex element.")
    vertex_count = int(vertex_element["count"])
    vertex_record_bytes = int(vertex_element["record_bytes"])
    if vertex_count <= 0:
        raise PlyError("PLY vertex element is empty.")
    if not elements or elements[0]["name"] != "vertex":
        raise PlyError("The vertex element must be the first PLY element.")
    unsupported = [
        item
        for item in elements
        if item["name"] not in {"vertex", "camera", "face"} and item["count"]
    ]
    if unsupported:
        details = ", ".join(f"{item['name']}={item['count']}" for item in unsupported)
        raise PlyError(f"Additional populated PLY elements are not supported: {details}")

    variable_elements = [item for item in elements if item["variable_length"]]
    invalid_variable_elements = [
        item for item in variable_elements if item["name"] != "face"
    ]
    if invalid_variable_elements:
        details = ", ".join(str(item["name"]) for item in invalid_variable_elements)
        raise PlyError(f"Variable-length PLY elements are not supported: {details}")

    names = {item.name for item in vertex_properties}
    missing = {"x", "y", "z"} - names
    if missing:
        raise PlyError(f"PLY is missing position properties: {', '.join(sorted(missing))}")

    fixed_payload_size = sum(
        int(item["count"]) * int(item["record_bytes"])
        for item in elements
        if not item["variable_length"]
    )
    expected_size = len(raw_header) + fixed_payload_size
    actual_size = path.stat().st_size
    if variable_elements and actual_size < expected_size:
        raise PlyError(
            f"PLY is shorter than the fixed payload declared by its header: "
            f"expected at least {expected_size:,} bytes, found {actual_size:,} bytes."
        )
    if not variable_elements and actual_size != expected_size:
        raise PlyError(
            f"PLY size mismatch: expected {expected_size:,} bytes from the header, "
            f"found {actual_size:,} bytes."
        )

    return PlyHeader(
        path=path,
        header_bytes=len(raw_header),
        vertex_count=vertex_count,
        record_bytes=vertex_record_bytes,
        properties=tuple(vertex_properties),
        comments=tuple(comments),
        elements=tuple(
            PlyElement(
                str(item["name"]),
                int(item["count"]),
                int(item["record_bytes"]),
                bool(item["variable_length"]),
            )
            for item in elements
        ),
    )


def detect_camera_layout(header: PlyHeader) -> CameraLayout:
    point_type = header.property("point_type")
    camera_element = header.element("camera")
    if point_type and point_type.data_type in {"uchar", "uint8"} and camera_element:
        with header.path.open("rb") as stream:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
                def type_at(index: int) -> int:
                    return data[header.header_bytes + index * header.record_bytes + point_type.offset]

                index = 0
                ranges: list[tuple[int, int]] = []
                for expected_type in (0, 1, 2):
                    start = index
                    while index < header.vertex_count and type_at(index) == expected_type:
                        index += 1
                    ranges.append((start, index - start))

        scene_start, scene_count = ranges[0]
        camera_start, camera_count = ranges[1]
        trajectory_start, trajectory_count = ranges[2]
        if scene_start == 0 and scene_count and camera_count and index == header.vertex_count:
            if camera_element.count <= 0 or camera_count % camera_element.count:
                return CameraLayout.not_detected(
                    header.vertex_count,
                    "Camera point count is inconsistent with the camera element.",
                )
            samples_per_frustum = camera_count // camera_element.count
            return CameraLayout(
                detected=True,
                scene_count=scene_count,
                trajectory_start=trajectory_start,
                trajectory_count=trajectory_count,
                view_count=camera_element.count,
                samples_per_frustum=samples_per_frustum,
                runs=(CameraRun("camera_frustum", (255, 255, 255), camera_start, camera_count),),
            )

    color_properties = tuple(header.property(name) for name in ("red", "green", "blue"))
    if any(item is None for item in color_properties):
        return CameraLayout.not_detected(header.vertex_count, "RGB properties are missing.")
    if any(item.data_type not in {"uchar", "uint8"} for item in color_properties if item):
        return CameraLayout.not_detected(
            header.vertex_count, "Camera markers require uchar RGB properties."
        )

    red, green, blue = color_properties
    assert red is not None and green is not None and blue is not None

    with header.path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            def color_at(index: int) -> tuple[int, int, int]:
                base = header.header_bytes + index * header.record_bytes
                return data[base + red.offset], data[base + green.offset], data[base + blue.offset]

            index = header.vertex_count - 1
            while index >= 0 and color_at(index) == TRAJECTORY_COLOR:
                index -= 1
            trajectory_start = index + 1
            trajectory_count = header.vertex_count - trajectory_start

            reverse_runs: list[CameraRun] = []
            for name, expected_color in reversed(CAMERA_RUNS):
                if index < 0 or color_at(index) != expected_color:
                    return CameraLayout.not_detected(
                        header.vertex_count,
                        f"Expected trailing camera run '{name}' was not found.",
                    )
                run_end = index + 1
                while index >= 0 and color_at(index) == expected_color:
                    index -= 1
                run_start = index + 1
                reverse_runs.append(
                    CameraRun(name, expected_color, run_start, run_end - run_start)
                )

    runs = tuple(reversed(reverse_runs))
    run_counts = {run.count for run in runs}
    if len(run_counts) != 1:
        return CameraLayout.not_detected(
            header.vertex_count, "The six camera marker runs have different lengths."
        )

    run_count = runs[0].count
    if run_count % 128 != 0:
        return CameraLayout.not_detected(
            header.vertex_count,
            "Camera marker count is not divisible by 128 samples.",
        )
    samples_per_frustum = 128
    view_count = run_count // samples_per_frustum
    if trajectory_count != max(0, view_count - 1) * 8:
        return CameraLayout.not_detected(
            header.vertex_count,
            "Camera marker runs and trajectory length are inconsistent.",
        )

    return CameraLayout(
        detected=True,
        scene_count=runs[0].start,
        trajectory_start=trajectory_start,
        trajectory_count=trajectory_count,
        view_count=view_count,
        samples_per_frustum=samples_per_frustum,
        runs=runs,
    )


GAUSSIAN_COMMON_PROPERTIES = (
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1",
    "rot_0", "rot_1", "rot_2", "rot_3",
)


def gaussian_splat_kind(header: PlyHeader) -> str | None:
    """Return the supported Gaussian representation stored by the PLY."""
    float_names = {
        item.name for item in header.properties if item.data_type in {"float", "float32"}
    }
    if not all(name in float_names for name in GAUSSIAN_COMMON_PROPERTIES):
        return None
    if "scale_2" not in float_names:
        return "2DGS"
    if "artifixer" in header.path.stem.lower():
        return "ArtiFixer3D"
    return "3DGS"


def is_gaussian_splat(header: PlyHeader) -> bool:
    return gaussian_splat_kind(header) is not None


def resolve_input_path(input_path: Path) -> tuple[Path, Path | None]:
    """Resolve either a standalone PLY or a relighting package directory."""
    resolved = input_path.expanduser().resolve()
    if not resolved.is_dir():
        return resolved, None

    ply_candidates = [resolved / name for name in PACKAGE_PLY_NAMES]
    ply_path = next((path for path in ply_candidates if path.is_file()), None)
    if ply_path is None:
        discovered = sorted(path for path in resolved.glob("*.ply") if path.is_file())
        if len(discovered) == 1:
            ply_path = discovered[0]
        else:
            expected = ", ".join(PACKAGE_PLY_NAMES)
            if discovered:
                expected += f"; found {', '.join(path.name for path in discovered)}"
            raise LightPackageError(
                f"Light package directory has no unambiguous PLY "
                f"({expected}; {resolved})"
            )

    light_sources_path = resolved / LIGHT_SOURCES_NAME
    if not light_sources_path.is_file():
        raise LightPackageError(
            f"Light package directory is missing: {LIGHT_SOURCES_NAME} ({resolved})"
        )
    return ply_path, light_sources_path


def read_light_sources(path: Path, *, scene_vertex_count: int) -> dict[str, object]:
    """Read and minimally validate the light registry consumed by the browser."""
    try:
        with path.open("r", encoding="utf-8") as stream:
            registry = json.load(stream)
    except OSError as exc:
        raise LightPackageError(f"Cannot open light registry: {path}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LightPackageError(f"Invalid light registry JSON: {path}: {exc}") from exc

    if not isinstance(registry, dict) or registry.get("schema_version") != 1:
        raise LightPackageError("light_sources.json must use schema_version 1.")

    fixtures = registry.get("fixtures")
    if isinstance(fixtures, list) and fixtures:
        fixture_ids: set[str] = set()
        proxy_count = 0
        for index, fixture in enumerate(fixtures):
            if not isinstance(fixture, dict):
                raise LightPackageError(f"fixtures[{index}] must be an object.")
            fixture_id = fixture.get("fixture_id")
            proxy = fixture.get("parametric_proxy")
            if not isinstance(fixture_id, str) or not fixture_id or fixture_id in fixture_ids:
                raise LightPackageError(f"fixtures[{index}] has an invalid or duplicate fixture_id.")
            fixture_ids.add(fixture_id)
            if not isinstance(proxy, dict):
                raise LightPackageError(f"{fixture_id} is missing parametric_proxy.")
            count = proxy.get("proxy_count")
            position = proxy.get("position_world")
            rotation = proxy.get("rotation_world_from_proxy")
            extent = proxy.get("extent_98_percent_world_units")
            color = proxy.get("base_emission_srgb_power_1")
            if not isinstance(count, int) or count <= 0:
                raise LightPackageError(f"{fixture_id} has an invalid proxy_count.")
            if not all(
                isinstance(value, list) and len(value) == 3
                for value in (position, extent, color)
            ):
                raise LightPackageError(f"{fixture_id} has invalid proxy position, extent, or color.")
            if not (
                isinstance(rotation, list)
                and len(rotation) == 3
                and all(isinstance(row, list) and len(row) == 3 for row in rotation)
            ):
                raise LightPackageError(f"{fixture_id} has an invalid proxy rotation matrix.")
            proxy_count += count

        counts = registry.get("counts")
        if isinstance(counts, dict):
            declared_scene_count = counts.get("background_ply_vertex_count")
            if declared_scene_count is not None and declared_scene_count != scene_vertex_count:
                raise LightPackageError(
                    "Light registry background_ply_vertex_count does not match "
                    f"the PLY: {declared_scene_count} != {scene_vertex_count}."
                )
            declared_proxy_count = counts.get("parametric_proxy_gaussian_count")
            if declared_proxy_count is not None and declared_proxy_count != proxy_count:
                raise LightPackageError(
                    "Light registry parametric_proxy_gaussian_count does not match fixtures: "
                    f"{declared_proxy_count} != {proxy_count}."
                )
        return registry

    emitters = registry.get("emitters")
    if isinstance(emitters, list) and emitters:
        emitter_ids: set[str] = set()
        sample_count = 0
        for index, emitter in enumerate(emitters):
            if not isinstance(emitter, dict):
                raise LightPackageError(f"emitters[{index}] must be an object.")
            emitter_id = emitter.get("emitter_id")
            if not isinstance(emitter_id, str) or not emitter_id or emitter_id in emitter_ids:
                raise LightPackageError(f"emitters[{index}] has an invalid or duplicate emitter_id.")
            emitter_ids.add(emitter_id)
            samples = emitter.get("samples")
            if not isinstance(samples, list):
                raise LightPackageError(f"{emitter_id} must contain a samples array.")
            if emitter.get("enabled") is False:
                if samples:
                    raise LightPackageError(
                        f"Disabled emitter {emitter_id} must not contain samples."
                    )
                continue
            for field in ("center_world", "normal_world", "relative_flux_rgb_initial"):
                value = emitter.get(field)
                if not isinstance(value, list) or len(value) != 3:
                    raise LightPackageError(f"{emitter_id} has an invalid {field}.")
            if not isinstance(emitter.get("relative_area_world2"), (int, float)):
                raise LightPackageError(f"{emitter_id} has an invalid relative_area_world2.")
            if not isinstance(samples, list) or not samples:
                raise LightPackageError(f"{emitter_id} must contain a non-empty samples array.")
            for sample_index, sample in enumerate(samples):
                if not isinstance(sample, dict):
                    raise LightPackageError(f"{emitter_id}.samples[{sample_index}] must be an object.")
                position = sample.get("position_world")
                normal = sample.get("normal_world")
                weight = sample.get("area_weight_world2")
                if not isinstance(position, list) or len(position) != 3:
                    raise LightPackageError(
                        f"{emitter_id}.samples[{sample_index}] has an invalid position_world."
                    )
                if not isinstance(normal, list) or len(normal) != 3:
                    raise LightPackageError(
                        f"{emitter_id}.samples[{sample_index}] has an invalid normal_world."
                    )
                if not isinstance(weight, (int, float)) or weight < 0:
                    raise LightPackageError(
                        f"{emitter_id}.samples[{sample_index}] has an invalid area_weight_world2."
                    )
            sample_count += len(samples)

        counts = registry.get("counts")
        if isinstance(counts, dict):
            declared_emitter_count = counts.get("emitter_count")
            if declared_emitter_count is not None and declared_emitter_count != len(emitters):
                raise LightPackageError(
                    "Light registry emitter_count does not match emitters: "
                    f"{declared_emitter_count} != {len(emitters)}."
                )
            declared_sample_count = counts.get("sample_count")
            if declared_sample_count is not None and declared_sample_count != sample_count:
                raise LightPackageError(
                    "Light registry sample_count does not match emitters: "
                    f"{declared_sample_count} != {sample_count}."
                )
        return registry

    raise LightPackageError(
        "light_sources.json must contain a non-empty fixtures or emitters array."
    )


def build_summary(
    header: PlyHeader,
    layout: CameraLayout,
    light_sources: dict[str, object] | None = None,
) -> dict[str, object]:
    gaussian_kind = gaussian_splat_kind(header)
    summary: dict[str, object] = {
        "file": str(header.path),
        "fileName": header.path.name,
        "fileBytes": header.path.stat().st_size,
        "format": "binary_little_endian 1.0",
        "headerBytes": header.header_bytes,
        "vertexCount": header.vertex_count,
        "recordBytes": header.record_bytes,
        "gaussianSplat": gaussian_kind is not None,
        "gaussianKind": gaussian_kind,
        "mesh": {
            "available": bool(header.element("face")),
            "faceCount": header.element("face").count if header.element("face") else 0,
        },
        "properties": [
            {"name": item.name, "type": item.data_type, "offset": item.offset}
            for item in header.properties
        ],
        "comments": list(header.comments),
        "cameraLayout": {
            "detected": layout.detected,
            "sceneCount": layout.scene_count,
            "trajectoryStart": layout.trajectory_start,
            "trajectoryCount": layout.trajectory_count,
            "viewCount": layout.view_count,
            "samplesPerFrustum": layout.samples_per_frustum,
            "reason": layout.reason,
            "runs": [
                {
                    "name": run.name,
                    "color": list(run.color),
                    "start": run.start,
                    "count": run.count,
                }
                for run in layout.runs
            ],
        },
    }
    if light_sources is not None:
        fixtures = light_sources.get("fixtures")
        emitters = light_sources.get("emitters")
        if isinstance(fixtures, list) and fixtures:
            summary["lightPackage"] = {
                "available": True,
                "registryKind": "parametric_proxy_fixtures",
                "schemaVersion": light_sources.get("schema_version"),
                "status": light_sources.get("status"),
                "fixtureCount": len(fixtures),
                "proxyCount": sum(
                    int(fixture["parametric_proxy"]["proxy_count"])
                    for fixture in fixtures
                    if isinstance(fixture, dict)
                ),
                "affectsSceneSurfaceLighting": False,
                "registryUrl": "/light-sources.json",
            }
        elif isinstance(emitters, list) and emitters:
            counts = light_sources.get("counts")
            summary["lightPackage"] = {
                "available": True,
                "registryKind": "finite_area_emitters",
                "schemaVersion": light_sources.get("schema_version"),
                "status": light_sources.get("scale_status"),
                "emitterCount": len(emitters),
                "sampleCount": sum(
                    len(emitter.get("samples", []))
                    for emitter in emitters
                    if isinstance(emitter, dict)
                ),
                "sourceFixtureCount": (
                    counts.get("source_fixture_count")
                    if isinstance(counts, dict)
                    else None
                ),
                "affectsSceneSurfaceLighting": False,
                "registryUrl": "/light-sources.json",
            }
        else:
            raise LightPackageError("Validated light registry has no supported light array.")
    else:
        summary["lightPackage"] = {"available": False}
    return summary


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        summary: dict[str, object],
        light_sources_path: Path | None = None,
    ):
        self.summary = summary
        self.ply_path = Path(str(summary["file"]))
        self.light_sources_path = light_sources_path
        super().__init__(address, ViewerRequestHandler)


class ViewerRequestHandler(BaseHTTPRequestHandler):
    server: ViewerServer
    protocol_version = "HTTP/1.1"

    STATIC_FILES = {
        "/": "index.html",
        "/index.html": "index.html",
        "/viewer.js": "viewer.js",
        "/ply-worker.js": "ply-worker.js",
        "/gs": "gs.html",
        "/gs.html": "gs.html",
        "/gs-viewer.js": "gs-viewer.js",
        "/gs-worker.js": "gs-worker.js",
    }

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib callback name
        self._handle_request(send_body=False)

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        self._handle_request(send_body=True)

    def _handle_request(self, *, send_body: bool) -> None:
        path = unquote(urlsplit(self.path).path)
        if path == "/metadata.json":
            payload = json.dumps(self.server.summary, ensure_ascii=False).encode("utf-8")
            self._send_bytes(payload, "application/json; charset=utf-8", send_body)
            return
        if path == "/pointcloud.ply":
            self._send_ply(send_body)
            return
        if path == "/light-sources.json" and self.server.light_sources_path is not None:
            payload = self.server.light_sources_path.read_bytes()
            self._send_bytes(payload, "application/json; charset=utf-8", send_body)
            return
        if path in self.STATIC_FILES:
            static_path = WEB_ROOT / self.STATIC_FILES[path]
            if not static_path.is_file():
                self.send_error(500, f"Viewer asset is missing: {static_path.name}")
                return
            content_type = mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
            self._send_bytes(static_path.read_bytes(), content_type, send_body)
            return
        self.send_error(404, "Not found")

    def _send_bytes(self, payload: bytes, content_type: str, send_body: bool) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if send_body:
            self.wfile.write(payload)

    def _send_ply(self, send_body: bool) -> None:
        size = self.server.ply_path.stat().st_size
        start, end = 0, size - 1
        status = 200
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self._send_range_not_satisfiable(size)
                return
            first, last = match.groups()
            if not first and not last:
                self._send_range_not_satisfiable(size)
                return
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            elif last:
                suffix = int(last)
                if suffix <= 0:
                    self._send_range_not_satisfiable(size)
                    return
                start = max(0, size - suffix)
            if start < 0 or start >= size or end < start:
                self._send_range_not_satisfiable(size)
                return
            status = 206

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not send_body:
            return

        with self.server.ply_path.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _send_range_not_satisfiable(self, size: int) -> None:
        self.send_response(416)
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, message_format: str, *args: object) -> None:
        if getattr(self.server, "quiet", False):
            return
        super().log_message(message_format, *args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    def valid_port(value: str) -> int:
        try:
            port = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("port must be an integer") from exc
        if not 0 <= port <= 65535:
            raise argparse.ArgumentTypeError("port must be between 0 and 65535")
        return port

    parser = argparse.ArgumentParser(
        description="Display a 3DGS PLY or light package with WebGL2."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_PLY,
        help=(
            "binary little-endian PLY, or a directory containing "
            f"one of {', '.join(PACKAGE_PLY_NAMES)} and {LIGHT_SOURCES_NAME} "
            f"(default: {DEFAULT_PLY})"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="listen address (default: 127.0.0.1)")
    parser.add_argument("--port", type=valid_port, default=8000, help="listen port; 0 chooses a free port")
    parser.add_argument("--no-browser", action="store_true", help="do not open the default browser")
    parser.add_argument("--inspect-only", action="store_true", help="validate and print metadata without serving")
    parser.add_argument("--json", action="store_true", help="print metadata as JSON")
    parser.add_argument("--quiet", action="store_true", help="suppress HTTP request logs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        ply_path, light_sources_path = resolve_input_path(args.input)
        header = read_ply_header(ply_path)
        layout = detect_camera_layout(header)
        if (
            light_sources_path is not None
            and gaussian_splat_kind(header) is None
            and header.element("face") is None
        ):
            raise LightPackageError(
                "A light package PLY must contain a supported Gaussian representation "
                "or a face mesh."
            )
        light_sources = (
            read_light_sources(light_sources_path, scene_vertex_count=header.vertex_count)
            if light_sources_path is not None
            else None
        )
    except (PlyError, LightPackageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = build_summary(header, layout, light_sources)
    if args.json or args.inspect_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        camera_text = (
            f", cameras={layout.view_count:,}, trajectory={layout.trajectory_count:,} points"
            if layout.detected and layout.view_count is not None
            else ", embedded camera layout not detected"
        )
        format_name = gaussian_splat_kind(header)
        if format_name is None and header.element("face"):
            format_name = "mesh"
        format_text = (
            f", format={format_name}"
            if format_name is not None
            else ""
        )
        print(
            f"PLY: {header.path}\n"
            f"Vertices: {header.vertex_count:,} ({header.path.stat().st_size / 1024 / 1024:.1f} MiB)"
            f"{format_text}{camera_text}"
        )
        if light_sources is not None:
            package = summary["lightPackage"]
            assert isinstance(package, dict)
            if package.get("registryKind") == "finite_area_emitters":
                print(
                    f"Lights: {package['emitterCount']:,} emitters, "
                    f"{package['sampleCount']:,} area samples"
                )
            else:
                print(
                    f"Lights: {package['fixtureCount']:,} fixtures, "
                    f"{package['proxyCount']:,} emissive proxies"
                )

    if args.inspect_only:
        return 0

    if not WEB_ROOT.is_dir():
        print(f"error: viewer assets not found: {WEB_ROOT}", file=sys.stderr)
        return 2

    try:
        server = ViewerServer((args.host, args.port), summary, light_sources_path)
    except (OSError, OverflowError) as exc:
        print(f"error: cannot start server on {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 2
    server.quiet = args.quiet

    actual_port = server.server_address[1]
    browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    base_url = f"http://{browser_host}:{actual_port}/"
    if is_gaussian_splat(header):
        url = f"{base_url}gs"
        print(f"Gaussian renderer: {url}\nPoint viewer: {base_url}\nStop: Ctrl+C")
    else:
        url = base_url
        print(f"Viewer: {url}\nStop: Ctrl+C")
    if not args.no_browser:
        threading.Timer(0.35, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
