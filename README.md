# pyquantization

Python bindings for **Quad Mesh Quantization Without a T-Mesh** (Coudert-Osmont et al., 2024).

Takes a seamless UV parameterization on a triangle mesh and produces an integer-grid-aligned (quantized) parameterization suitable for quad mesh extraction.

## Installation

```bash
pip install pyquantization
```

Or from source:

```bash
pip install .
```

## Usage

```python
import numpy as np
import pyquantization

# vertices: (n_verts, 3) float64
# triangles: (n_tris, 3) int32
# uv_per_corner: (n_uvs, 2) float64
# uv_triangles: (n_tris, 3) int32
# feature_edges: (n_feat, 2) int32

out_verts, out_faces, out_uvs, out_uv_tris, out_feats = pyquantization.quantize_mesh(
    vertices, triangles, uv_per_corner, uv_triangles, feature_edges,
    scale=-1.0,       # -1.0 for automatic
    scale_auto=1.0,   # multiplier for auto scale
    mode="reembed"    # "reembed" | "imprint" | "decimate"
)
```

## Output modes

- **reembed** (recommended): Re-embeds quantized UVs on the original fine mesh via least-squares optimization. Produces the highest quality output.
- **imprint**: Imprints quantized UVs from the decimated mesh onto the original fine mesh via geometric clipping.
- **decimate**: Returns the decimated mesh directly with quantized UVs.

## Reference

Coudert-Osmont, Y., Desobry, D., Heistermann, M., Bommes, D., Ray, N., Sokolov, D. (2024). "Quad Mesh Quantization Without a T-Mesh." *Computer Graphics Forum*.

Original C++ code from [etcorman/RectangularSurfaceParameterization](https://github.com/etcorman/RectangularSurfaceParameterization/tree/main/QuantizationYoann).

If you use this in scientific work, please cite:

```bibtex
@inproceedings{coudert2024quad,
  title={Quad Mesh Quantization Without a T-Mesh},
  author={Coudert-Osmont, Yoann and Desobry, David and Heistermann, Martin
          and Bommes, David and Ray, Nicolas and Sokolov, Dmitry},
  booktitle={Computer Graphics Forum},
  volume={43},
  number={1},
  year={2024}
}
```

## License

AGPL-3.0-or-later (matching the original C++ code).
