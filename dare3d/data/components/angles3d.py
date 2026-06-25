from scipy.spatial.transform import Rotation as R
from enum import Enum
import numpy as np
import torch


class ANGLE_REPRESENTATION(Enum):
    AXIS_ANGLE = 6
    EULER = 1
    ROT_MAT_9D_GS = 2
    ROT_MAT_6D = 3
    QUATERNION = 4
    ROT_MAT_9D_SVD = 5

    def from_string(value):
        if value == "axis_angle":
            return ANGLE_REPRESENTATION.AXIS_ANGLE
        if value == "euler":
            return ANGLE_REPRESENTATION.EULER
        # if value == "rotation_matrix_GS":
        #     return ANGLE_REPRESENTATION.ROT_MAT_9D_GS
        if value == "rotation_matrix_SVD":
            return ANGLE_REPRESENTATION.ROT_MAT_9D_SVD
        if value == "quaternion":
            return ANGLE_REPRESENTATION.QUATERNION
        if value == "rotation_matrix_6d":
            return ANGLE_REPRESENTATION.ROT_MAT_6D
        raise ValueError(f"Unknown angle representation {value}")


def quaternion_to_representation(quaternion, representation):
    rotation = R.from_quat(wxyz_to_xyzw(quaternion))
    if representation == ANGLE_REPRESENTATION.QUATERNION:
        return quaternion
    # Rotation matrix
    if (
        representation == ANGLE_REPRESENTATION.ROT_MAT_9D_GS
        or representation == ANGLE_REPRESENTATION.ROT_MAT_9D_SVD
        or representation == ANGLE_REPRESENTATION.ROT_MAT_6D
    ):
        rotation = rotation.as_matrix()
        return np.reshape(rotation, (9))  # Flatten matrix
    if representation == ANGLE_REPRESENTATION.EULER:
        return rotation.as_euler("xyz")
    if representation == ANGLE_REPRESENTATION.AXIS_ANGLE:
        value = rotation.as_rotvec()
        return value
    raise ValueError(f"Unsupported representation : {representation}")


def get_quaternion(point1, point2):
    def sort_points_clockwise(points, reference_point):
        # Calculate vectors from reference point to the points
        vectors = points - reference_point

        # Calculate angles of points with respect to the reference point
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        # Sort the points based on angles in clockwise order
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]

        return sorted_points

    center = (point1 + point2) / 2
    point1, point2 = sort_points_clockwise(np.array([point1, point2]), center)
    quat = compute_quaternion(point1, point2)
    return quat

def apply_quaternion(quaternion, vector):
    from scipy.spatial.transform import Rotation as R
    q = R.from_quat(quaternion)
    return q.apply(vectors=vector)

def compute_quaternion(point1, point2):
    vector = point2 - point1
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        # No rotation, return identity quaternion
        return np.array([1, 0, 0, 0])
    axis = vector / norm

    angle = np.arccos(np.dot([1, 0, 0], axis))
    half_angle = angle / 2
    quat = np.array(
        [
            np.cos(half_angle),
            np.sin(half_angle) * axis[0],
            np.sin(half_angle) * axis[1],
            np.sin(half_angle) * axis[2],
        ]
    )

    return quat.astype(np.float32)


def swap_first_and_last(omat):
    mat = np.copy(omat)
    if len(mat.shape) == 1:
        mat[[0, -1]] = mat[[-1, 0]]
    if len(mat.shape) == 2:
        mat[:, [0, -1]] = mat[:, [-1, 0]]
    return mat


def wxyz_to_xyzw(quaternion):
    return swap_first_and_last(quaternion)


def xyzw_to_wxyz(quaternion):
    return swap_first_and_last(quaternion)


def draw_line_in_matrix(matrix, start, end, num_points, val=1):
    # Generate points along the line using linear interpolation
    t = np.linspace(0, 1, num_points)
    line_points = (1 - t)[:, np.newaxis] * start + t[:, np.newaxis] * end

    if np.any(np.isnan(start)) or np.any(np.isnan(end)):
        return matrix

    for point in line_points:
        x, y, z = point
        x_idx = min(max(0, int(x)), matrix.shape[0] - 1)
        y_idx = min(max(0, int(y)), matrix.shape[1] - 1)
        z_idx = min(max(0, int(z)), matrix.shape[2] - 1)
        matrix[x_idx, y_idx, z_idx] = val
    return matrix


def representation_to_quaternion(rotation, representation, post=True):
    scipy_rotation = None
    if representation == ANGLE_REPRESENTATION.QUATERNION:
        rotation /= np.linalg.norm(rotation)
        return rotation

    if representation == ANGLE_REPRESENTATION.ROT_MAT_6D:
        if len(rotation.shape) == 3:
            rotation = np.expand_dims(rotation, axis=0)
        if rotation.shape[-1] == 6:
            rotation = rot6_to_rot9(rotation)
        else:
            rotation = np.reshape(rotation, (-1, 3, 3))
        scipy_rotation = R.from_matrix(np.copy(rotation[0]))
    if representation == ANGLE_REPRESENTATION.ROT_MAT_9D_GS:
        if len(rotation.shape) == 3:
            rotation = np.expand_dims(rotation, axis=0)
        if post:
            rotation = gram_schmidt_orthogonalization(rotation)
        else:
            rotation = np.reshape(rotation, (-1, 3, 3))
        scipy_rotation = R.from_matrix(np.copy(rotation[0]))
    if representation == ANGLE_REPRESENTATION.ROT_MAT_9D_SVD:
        if len(rotation.shape) == 3:
            rotation = np.expand_dims(rotation, axis=0)
        if post:
            if isinstance(rotation, np.ndarray):
                rotation = torch.tensor(rotation)
            rotation = symmetric_orthogonalization(rotation)
            rotation = rotation.detach().cpu().numpy()
        else:
            rotation = np.reshape(rotation, (-1, 3, 3))
        scipy_rotation = R.from_matrix(np.copy(rotation[0]))
    if representation == ANGLE_REPRESENTATION.AXIS_ANGLE:
        # rotation = np.reshape(rotation, (3, 3))
        scipy_rotation = R.from_rotvec(rotation)
    if representation == ANGLE_REPRESENTATION.EULER:
        scipy_rotation = R.from_euler("xyz", rotation)

    if scipy_rotation is None:
        raise ValueError(f"Unsupported representation : {representation}")

    scipy_rotation_len = 1
    try:
        scipy_rotation_len = len(scipy_rotation)
    except TypeError:
        pass

    if scipy_rotation_len > 1:
        combined_rotation = scipy_rotation[0]
        for i in range(1, scipy_rotation_len):
            combined_rotation = combined_rotation * scipy_rotation[i]
        scipy_rotation = combined_rotation

    quat = scipy_rotation.as_quat()
    quat /= np.linalg.norm(quat)
    return xyzw_to_wxyz(quat)


def rot6_to_rot9(x):
    # R6d = [r1|r2]
    # R = [R·1|R·2|R·3]

    m = torch.reshape(x, [-1, 6])
    # Shape (batch, 3)
    r1 = m[:, 0:3]
    r2 = m[:, 3:6]

    # φ is the normalization operation
    # R·1 = φ(r1)
    R1 = torch.nn.functional.normalize(r1, p=2, dim=-1)

    # R·3 = φ(R·1 × r2)
    R3 = torch.cross(R1, r2)
    R3 = torch.nn.functional.normalize(R3, p=2, dim=-1)

    # R·2 = R·3 × R·1
    R2 = torch.cross(R3, R1)

    R1 = torch.reshape(R1, [-1, 3, 1])
    R2 = torch.reshape(R2, [-1, 3, 1])
    R3 = torch.reshape(R3, [-1, 3, 1])
    R = torch.concat([R1, R2, R3], dim=-1)

    return R


def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu


def gram_schmidt_orthogonalization(x):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, device="cpu")
    m = torch.reshape(x, (-1, 3, 3))
    r = gram_schmidt(m)
    return r


def symmetric_orthogonalization(x):
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

    from https://github.com/amakadia/svd_for_pose

    x: should have shape [batch_size, 9]

    Returns a [batch_size, 3, 3] tensor, where each inner 3x3 matrix is in SO(3).
    """
    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r


def compute_axis_angle(quaternion):
    # Normalize the quaternion
    quaternion /= np.linalg.norm(quaternion)

    # Compute the angle
    angle = 2 * np.arccos(quaternion[0])

    # Compute the axis
    axis = quaternion[1:] / np.sin(angle / 2)

    return axis, angle


def inverse_quaternion(quaternion):
    inverted_quaternion = np.copy(quaternion)
    inverted_quaternion[1:] = -inverted_quaternion[1:]
    return inverted_quaternion.astype(np.float32)    

def compute_point_from_quaternion(quaternion, origin, length):
    axis, _ = compute_axis_angle(quaternion)
    return origin + axis * length


def get_points_from_quat(quaternion, center, length):
    p1 = compute_point_from_quaternion(quaternion, center, length)
    p2 = compute_point_from_quaternion(inverse_quaternion(quaternion), center, length)
    return p1, p2
