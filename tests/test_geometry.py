"""Self-check for napari_dare3d._api geometry + coordinate mapping.

Ties our quaternion->axis conversion and (m,t,x,y,z) -> napari (t,z,y,x) reversal
to DARE3D's OWN reference drawing (``angles3d.get_points_from_quat``). No napari,
no models, no GPU. Run:

    python tests/test_geometry.py        # standalone
    pytest tests/test_geometry.py        # collected with the suite
"""
import numpy as np

from napari_dare3d._api import _napari_point, _prediction_to_detection, to_layer_data
from dare3d.data.components.angles3d import get_points_from_quat


def _random_unit_quat(rng):
    # Avoid w ~ +-1 (angle ~ 0) where the axis is undefined in the repo encoding.
    while True:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        if 0.05 < abs(q[0]) < 0.99:
            return q


def _same_points(a, b, atol=1e-4):
    """True if point-lists a and b are equal as unordered sets within atol.

    (Repo goes quaternion -> arccos/sin -> axis, we normalise the imaginary part
    directly; mathematically identical but float-different by ~1e-7, so compare
    by nearest match rather than exact buckets.)"""
    b = [np.asarray(p, float) for p in b]
    used = [False] * len(b)
    for pa in (np.asarray(p, float) for p in a):
        for j, pb in enumerate(b):
            if not used[j] and np.linalg.norm(pa - pb) < atol:
                used[j] = True
                break
        else:
            return False
    return all(used)


def test_point_reversal():
    # (m, t, x, y, z) -> napari (t, z, y, x)
    assert _napari_point((3, 7, 10, 20, 30)) == (7.0, 30.0, 20.0, 10.0)


def test_axis_and_segment_match_repo():
    rng = np.random.default_rng(0)
    for _ in range(200):
        quat = _random_unit_quat(rng)
        center_xyz = rng.uniform(5, 95, size=3)
        length = rng.uniform(2, 40)

        # Reference (internal x,y,z). get_points_from_quat mutates its quat arg
        # (compute_axis_angle normalises in place) -> pass a copy.
        p1, p2 = get_points_from_quat(quat.copy(), center_xyz.copy(), 0.5 * length)

        det = _prediction_to_detection(
            {"center": (0, 4, *center_xyz), "length": np.array([length]), "rotation": quat.copy()}
        )
        (_pts, _kw, _lt), (axis_pts, _akw, alt) = to_layer_data([det])
        assert alt == "points"            # axes are sampled points (not a Vectors layer)
        lo, hi = axis_pts[0], axis_pts[-1]   # extreme samples, napari (t, z, y, x)

        # integer time on every sample; spatial part reversed back to (x, y, z)
        assert np.allclose(axis_pts[:, 0], 4.0)
        mine = [lo[1:][::-1], hi[1:][::-1]]
        assert _same_points(mine, [p1, p2]), f"\nmine={mine}\nrepo={[p1, p2]}"

        # endpoints span `length` and are centred on the detection
        assert abs(np.linalg.norm(hi[1:] - lo[1:]) - length) < 1e-6
        assert np.allclose((0.5 * (lo[1:] + hi[1:]))[::-1], center_xyz, atol=1e-6)

        # all samples are collinear and inside the segment (max step < 1 voxel in 3D)
        steps = np.linalg.norm(np.diff(axis_pts[:, 1:], axis=0), axis=1)
        assert steps.max() <= 1.0 + 1e-9


def test_centers_only_layer():
    dets = [{"center_internal": (0, 1, 2, 3, 4), "center_napari": (1.0, 4.0, 3.0, 2.0)}]
    layers = to_layer_data(dets)
    assert len(layers) == 1 and layers[0][2] == "points"
    assert layers[0][0].shape == (1, 4)


def test_decoded_raw_endpoints_drive_napari_axis():
    prediction = {
        "center": (0, 4, 10, 20, 30),
        "center_raw": (0, 4, 10, 20, 30),
        "length": np.array([99.0]),
        "length_raw_voxels": 4.0,
        "length_regression_voxels": 12.0,
        "length_physical_um": 12.0,
        "rotation": np.array([0.5, 0.5, 0.5, 0.5]),
        "axis_raw_xyz": np.array([1.0, 0.0, 0.0]),
        "endpoints_raw_xyz": np.array([[8.0, 20.0, 30.0], [12.0, 20.0, 30.0]]),
        "preprocessing_mode": "training_consistent",
    }

    detection = _prediction_to_detection(prediction)
    layers = to_layer_data([detection])
    axis_points = layers[1][0]

    assert detection["length"] == 4.0
    assert detection["length_regression_voxels"] == 12.0
    assert detection["preprocessing_mode"] == "training_consistent"
    assert np.allclose(axis_points[0], (4.0, 30.0, 20.0, 8.0))
    assert np.allclose(axis_points[-1], (4.0, 30.0, 20.0, 12.0))

if __name__ == "__main__":
    test_point_reversal()
    test_axis_and_segment_match_repo()
    test_centers_only_layer()
    print("test_geometry: ALL OK")
