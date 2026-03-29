import numpy as np
import pytest


@pytest.fixture
def quad_mesh():
    """Minimal valid mesh: two triangles forming a unit square."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    uv_per_corner = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    ], dtype=np.float64)
    uv_triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    feature_edges = np.empty((0, 2), dtype=np.int32)
    return vertices, triangles, uv_per_corner, uv_triangles, feature_edges


@pytest.fixture
def grid_mesh():
    """8x8 bumpy grid mesh (128 triangles) with feature edges."""
    n = 8
    xs = np.linspace(0, 3, n + 1)
    ys = np.linspace(0, 3, n + 1)
    xx, yy = np.meshgrid(xs, ys)
    zz = 0.3 * np.sin(xx * 2) * np.cos(yy * 2)
    vertices = np.column_stack([
        xx.ravel(), yy.ravel(), zz.ravel(),
    ]).astype(np.float64)

    triangles = []
    for i in range(n):
        for j in range(n):
            v00 = i * (n + 1) + j
            v10 = i * (n + 1) + j + 1
            v01 = (i + 1) * (n + 1) + j
            v11 = (i + 1) * (n + 1) + j + 1
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
    triangles = np.array(triangles, dtype=np.int32)

    uv_per_corner = vertices[:, :2].copy() * 2.0
    uv_triangles = triangles.copy()

    feat = []
    mid = n // 2
    for j in range(n):
        feat.append([mid * (n + 1) + j, mid * (n + 1) + j + 1])
        feat.append([j * (n + 1) + mid, (j + 1) * (n + 1) + mid])
    feature_edges = np.array(feat, dtype=np.int32)

    return vertices, triangles, uv_per_corner, uv_triangles, feature_edges
