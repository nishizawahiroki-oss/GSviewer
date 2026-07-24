# Third-party notices

`web/gs-viewer.js` contains WebGL/GLSL adaptations of forward-rendering
algorithms from the projects below. The surrounding browser viewer, streaming
PLY parser, UI, and server are implementations in this repository.

## 2D Gaussian Splatting / diff-surfel-rasterization

- Upstream: <https://github.com/hbb1/2d-gaussian-splatting>
- Upstream revision used by the supplied dataset: `335ad612f2e783a4e57b9cbc4d1e167bd599fc98`
- Rasterizer submodule: <https://github.com/hbb1/diff-surfel-rasterization>
- Rasterizer revision adapted here: `e0ed0207b3e0669960cfad70852200a4a5847f61`
- Adapted portions: tangent-plane transform, bounding support, and equations
  8–10 of the forward surfel intersection / low-pass response.
- License: Gaussian-Splatting License, included in
  [`licenses/Gaussian-Splatting-License.md`](licenses/Gaussian-Splatting-License.md).
  Its use is limited to non-commercial research and evaluation under that
  license.

Copyright notices retained from the upstream source:

> Copyright (C) 2023, Inria — GRAPHDECO research group. All rights reserved.

## ArtiFixer / 3DGRUT-ArtiFixer

- Upstream: <https://github.com/nv-tlabs/ArtiFixer>
- Upstream revision used by the supplied dataset: `a392c4dfe17459ef9952407accdb9fcdcdddba98`
- Renderer submodule: <https://github.com/nv-tlabs/3DGRUT-ArtiFixer>
- Renderer revision adapted here: `62e1038b74b2edc01440fd4ddf5f080109b6faba`
- Adapted portions: 3DGUT Unscented Transform projection, covariance dilation,
  Mip opacity compensation, tight rectangular support, and Gaussian-local
  ray-density response.
- License: Apache License 2.0, included in
  [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt).

Copyright notices retained from the upstream source:

> Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

## Implementation boundary

The upstream CUDA / OptiX runtimes are not bundled or executed by this browser
viewer. Their forward-rendering mathematics are adapted to WebGL2 GLSL. The
viewer currently evaluates only the SH DC color stored in `f_dc_0..2`; upstream
view-dependent higher-order SH evaluation is not included.
