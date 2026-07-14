from __future__ import annotations

import contextlib
import io
import json
import struct
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from view_3dgs import (
    CAMERA_RUNS,
    TRAJECTORY_COLOR,
    PlyError,
    ViewerServer,
    build_summary,
    detect_camera_layout,
    parse_args,
    read_ply_header,
)


RECORD = struct.Struct("<fffBBB")


def write_test_ply(
    path: Path,
    *,
    scene_colors: list[tuple[int, int, int]],
    camera_views: int = 0,
    samples_per_frustum: int = 128,
) -> None:
    records: list[tuple[float, float, float, int, int, int]] = []
    for index, color in enumerate(scene_colors):
        records.append((float(index), float(index + 1), float(-index), *color))

    if camera_views:
        for run_index, (_, color) in enumerate(CAMERA_RUNS):
            for view_index in range(camera_views):
                for sample in range(samples_per_frustum):
                    records.append(
                        (
                            float(view_index),
                            float(run_index),
                            sample / max(1, samples_per_frustum - 1),
                            *color,
                        )
                    )
        for segment in range(camera_views - 1):
            for sample in range(8):
                records.append((segment + sample / 7.0, 0.0, 0.0, *TRAJECTORY_COLOR))

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment synthetic test data\n"
        f"element vertex {len(records)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        for record in records:
            stream.write(RECORD.pack(*record))


class PlyInspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "test.ply"

    def test_detects_camera_suffix_without_misclassifying_white_scene_points(self) -> None:
        scene_colors = [(255, 255, 255), (30, 40, 50), (255, 255, 255), (12, 34, 56)]
        write_test_ply(self.path, scene_colors=scene_colors, camera_views=3)

        header = read_ply_header(self.path)
        layout = detect_camera_layout(header)

        self.assertEqual(header.record_bytes, 15)
        self.assertTrue(layout.detected)
        self.assertEqual(layout.scene_count, len(scene_colors))
        self.assertEqual(layout.view_count, 3)
        self.assertEqual(layout.trajectory_count, 16)
        self.assertEqual([run.name for run in layout.runs], [name for name, _ in CAMERA_RUNS])
        self.assertTrue(all(run.count == 3 * 128 for run in layout.runs))

    def test_plain_rgb_ply_remains_a_scene(self) -> None:
        scene_colors = [(255, 255, 255), (10, 20, 30)]
        write_test_ply(self.path, scene_colors=scene_colors)

        header = read_ply_header(self.path)
        layout = detect_camera_layout(header)

        self.assertFalse(layout.detected)
        self.assertEqual(layout.scene_count, 2)
        self.assertEqual(layout.trajectory_count, 0)

    def test_rejects_camera_runs_that_are_not_128_samples_per_view(self) -> None:
        write_test_ply(
            self.path,
            scene_colors=[(10, 20, 30)],
            camera_views=2,
            samples_per_frustum=129,
        )

        layout = detect_camera_layout(read_ply_header(self.path))

        self.assertFalse(layout.detected)
        self.assertIn("128", layout.reason or "")

    def test_detects_one_camera_without_a_trajectory(self) -> None:
        write_test_ply(
            self.path,
            scene_colors=[(10, 20, 30)],
            camera_views=1,
        )

        layout = detect_camera_layout(read_ply_header(self.path))

        self.assertTrue(layout.detected)
        self.assertEqual(layout.view_count, 1)
        self.assertEqual(layout.trajectory_count, 0)

    def test_rejects_payload_size_mismatch(self) -> None:
        write_test_ply(self.path, scene_colors=[(10, 20, 30)])
        with self.path.open("ab") as stream:
            stream.write(b"unexpected")
        with self.assertRaisesRegex(PlyError, "size mismatch"):
            read_ply_header(self.path)

    def test_rejects_invalid_port_before_server_start(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--port", "-1"])


class HttpServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "served.ply"
        write_test_ply(self.path, scene_colors=[(10, 20, 30)], camera_views=2)
        header = read_ply_header(self.path)
        self.summary = build_summary(header, detect_camera_layout(header))
        self.server = ViewerServer(("127.0.0.1", 0), self.summary)
        self.server.quiet = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def _stop_server(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_serves_metadata_and_byte_ranges(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/metadata.json") as response:
            metadata = json.load(response)
        self.assertEqual(metadata["fileName"], self.path.name)
        self.assertEqual(metadata["cameraLayout"]["viewCount"], 2)

        request = urllib.request.Request(
            f"{self.base_url}/pointcloud.ply",
            headers={"Range": "bytes=0-3"},
        )
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b"ply\n")

        clamped_request = urllib.request.Request(
            f"{self.base_url}/pointcloud.ply",
            headers={"Range": "bytes=0-999999999"},
        )
        with urllib.request.urlopen(clamped_request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(len(response.read()), self.path.stat().st_size)

        invalid_request = urllib.request.Request(
            f"{self.base_url}/pointcloud.ply",
            headers={"Range": "bytes=-"},
        )
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(invalid_request)
        try:
            self.assertEqual(raised.exception.code, 416)
            self.assertEqual(
                raised.exception.headers["Content-Range"],
                f"bytes */{self.path.stat().st_size}",
            )
        finally:
            raised.exception.close()

    def test_serves_viewer_assets(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/") as response:
            html = response.read().decode("utf-8")
        self.assertIn("3DGS + Training Cameras", html)


if __name__ == "__main__":
    unittest.main()
