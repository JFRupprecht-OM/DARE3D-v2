from dare3d.data.components.angles3d import rot6_to_rot9, symmetric_orthogonalization
import numpy as np
import torch
import torch.nn.functional as F

def double_angle_len_loss(len_loss, angle_loss):
    def loss(head1_pred, head2_pred, head1_true, head2_true):
        pass
    return loss

def angle_len_loss(len_loss, angle_loss):
    def loss(head_pred, head_true):
        angle_loss_value = angle_loss(head_true["angle"], head_pred["angle"])
        len_loss_value = len_loss(head_true["len"], head_pred["len"])
        return angle_loss_value, len_loss_value
    return loss

def svd_loss(alpha=100):
    def angle_loss(y_true, y_pred):
        y_pred = symmetric_orthogonalization(y_pred)
        y_pred = torch.reshape(y_pred, (-1, 9))
        # return torch.mean(alpha * abs(y_true - y_pred))
        return geodesic_from_rot_matrix(y_true, y_pred)

    return angle_loss


def length_loss_wrapper(crop_size):
    def abs_mean(y_true, y_pred):
        # Compute absolute difference
        length_diff = torch.abs(y_true - y_pred)

        # Compute the abs diff batch mean
        return crop_size * torch.mean(length_diff)

    return abs_mean


def rot9_to_rot6(x):
    x = torch.reshape(x, [-1, 3, 3])
    r6 = torch.concat([x[..., 0], x[..., 1]], axis=-1)
    r6 = torch.reshape(r6, [-1, 6])
    return r6


def rot6_loss(alpha=100):
    def angle_loss(y_true, y_pred):
        y_pred = rot6_to_rot9(y_pred)
        y_pred = torch.reshape(y_pred, (-1, 9))
        return alpha * abs(y_true - y_pred)
        # return geodesic_from_rot_matrix(y_true, y_pred)

    return angle_loss


def rot_matrix_to_quat(y_true, y_pred):
    y_pred = torch.reshape(y_pred, (-1, 3, 3))
    y_true = torch.reshape(y_true, (-1, 3, 3))

    y_pred = matrix_to_quaternion(y_pred)
    y_true = matrix_to_quaternion(y_true)
    return y_true, y_pred


def geodesic_from_rot_matrix(y_true, y_pred):
    y_true, y_pred = rot_matrix_to_quat(y_true, y_pred)
    return angle_loss(y_true, y_pred)


def angle_loss(y_true, y_pred):
    # Φ3(q1, q2) = arccos(|q1 · q2|)
    # or 2 arccos for values in range [0; pi]
    # Normalize the true and predicted quaternions
    y_true_norm = F.normalize(y_true, dim=-1, p=2)
    y_pred_norm = F.normalize(y_pred, dim=-1, p=2)

    # Compute the dot product between true and predicted quaternions
    dot_product = torch.sum(y_true_norm * y_pred_norm, dim=-1)

    # distance = 1 - tf.abs(dot_product)
    dot_product = torch.clip(dot_product, -1.0 + 1e-4, 1.0 - 1e-4)
    distance = 2.0 * torch.acos(torch.abs(dot_product))
    distance = 180 * distance / np.pi
    return torch.mean(distance)


def angle_metric():
    def angle_loss_test(y_true, y_pred):
        # Φ3(q1, q2) = arccos(|q1 · q2|)
        # or 2 arccos for values in range [0; pi]
        # Normalize the true and predicted quaternions
        y_true_norm = torch.linalg.norm(y_true, dim=-1, ord=2)
        y_pred_norm = torch.linalg.norm(y_pred, dim=-1, ord=2)

        # Compute the dot product between true and predicted quaternions
        dot_product = torch.dot(y_true_norm, y_pred_norm)

        # distance = 1 - tf.abs(dot_product)
        dot_product = torch.clip(dot_product, -1.0 + 1e-4, 1.0 - 1e-4)
        distance = 2.0 * torch.acos(torch.abs(dot_product))
        distance = 180 * distance / np.pi
        return torch.mean(distance)

    return angle_loss_test

def quaternion_error(y_true, y_pred):
    y_true_norm = y_true / np.linalg.norm(y_true, ord=2)
    y_pred_norm = y_pred / np.linalg.norm(y_pred, ord=2)

    # Compute the dot product between true and predicted quaternions
    dot_product = np.dot(y_true_norm, y_pred_norm)

    dot_product = np.clip(dot_product, -1.0 + 1e-4, 1.0 - 1e-4)
    distance = 2.0 * np.arccos(np.abs(dot_product))
    distance = 180 * distance / np.pi
    return distance


def rot6_metric():
    def _angle_loss_test(y_true, y_pred):
        y_pred = rot6_to_rot9(y_pred)
        y_true, y_pred = rot_matrix_to_quat(y_true, y_pred)
        return angle_loss(y_true, y_pred)

    return _angle_loss_test


def rot9_svd_metric():
    def _angle_loss_test(y_true, y_pred):
        y_pred = symmetric_orthogonalization(y_pred)
        return geodesic_from_rot_matrix(y_true, y_pred)

    return _angle_loss_test


def quaternion_loss():
    return angle_loss

## Taken from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)