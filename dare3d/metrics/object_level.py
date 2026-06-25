import os

from typing import Tuple, List
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from tqdm import tqdm
from skimage import io
from numba import njit
from numba import types
from numba.typed import Dict

int_type = types.int64
float_type = types.float64[:]


def connected_components(
    label_matrix: np.ndarray, return_N: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Returns a tuple containing the labeled matrix and the number of connected components.

    Parameters:
    label_matrix (np.ndarray): The input array of labels.
    return_N (bool, optional): Whether to return the number of connected components. Defaults to False.

    Returns:
    Tuple[np.ndarray, int]: A tuple containing the labeled matrix and the number of connected components.
    np.ndarray: the labeled matrix if `return_N` is set to False.
    """
    label_matrix, n = scipy.ndimage.label(label_matrix)

    if return_N:
        return label_matrix, n
    return label_matrix


@njit
def one_pass_ccs(mat, prob_mat):
    """Compute the centroids, voxel counts, and probabilities of non-zero values in `mat`, using the probability values in `prob_mat`.

    Args:
        mat: A 4D NumPy array representing the input data.
        prob_mat: A 4D NumPy array representing the probability values for each voxel in `mat`.

    Returns:
        centroids: A dictionary mapping each non-zero value in `mat` to its corresponding centroid.
        voxel_counts: A dictionary mapping each non-zero value in `mat` to the number of times it appears.
        prob: A dictionary mapping each non-zero value in `mat` to its corresponding probability value.
    """
    centroids = Dict.empty(
        key_type=types.int64,
        value_type=float_type,
    )
    prob = Dict.empty(
        key_type=types.int64,
        value_type=types.float_,
    )
    voxel_counts = Dict.empty(
        key_type=types.int64,
        value_type=int_type,
    )

    # Iterate over the input data and compute the centroids, voxel counts, and probabilities
    for t in range(mat.shape[0]):
        for i in range(mat.shape[1]):
            for j in range(mat.shape[2]):
                for k in range(mat.shape[3]):
                    value = mat[t, i, j, k]
                    if value > 0:
                        value_as_list = np.asarray([t, i, j, k], dtype=np.float64)
                        # Add the centroid and voxel count for this value if it doesn't already exist
                        if value not in centroids:
                            centroids[value] = np.asarray([0, 0, 0, 0], dtype=np.float64)
                            voxel_counts[value] = 0
                            prob[value] = 0.0

                        # Increment the voxel count and compute the sum of the centroids
                        voxel_counts[value] += 1
                        summed_centroid = centroids[value] + value_as_list
                        centroids[value] = summed_centroid

                        # Add the probability value for this voxel
                        prob[value] += prob_mat[t, i, j, k]

    # Compute the mean centroid and probability for each non-zero value in `mat`
    for value in centroids.keys():
        centroids[value] = centroids[value] / voxel_counts[value]
        prob[value] = prob[value] / voxel_counts[value]

    return centroids, voxel_counts, prob


def statistics_optimized(label_matrix: np.ndarray, prob_matrix: np.ndarray) -> dict:
    """
    Compute the centroids, voxel counts, and mean probability for each CCS value in the label matrix.

    Parameters:
        label_matrix (np.ndarray): The label matrix to compute statistics from.
        prob_matrix (np.ndarray): The probability matrix to compute statistics from.

    Returns:
        ccs_stats (dict): A dictionary containing the centroids, voxel counts, and mean probability for each CCS value in the label matrix.
    """
    # Convert the prob_matrix to a float32 numpy array
    prob_matrix = prob_matrix.astype(np.float32)

    # Compute the centroids, voxel counts, and mean probability for each CCS value in the label matrix
    centroids, voxel_counts, prob = one_pass_ccs(label_matrix, prob_matrix)

    # Create a dictionary to store the statistics
    ccs_stats = {"centroids": [], "voxel_counts": [], "values": [], "mean_prob": []}

    # Loop through each CCS value and add its corresponding centroid, voxel count, and mean probability to the dictionary
    for ccs_value in centroids.keys():
        ccs_stats["centroids"].append(centroids[ccs_value])
        ccs_stats["voxel_counts"].append(voxel_counts[ccs_value])
        ccs_stats["values"].append(ccs_value)
        ccs_stats["mean_prob"].append(prob[ccs_value])

    return ccs_stats


def compute_iou_matrix(image_1: np.ndarray, image_2: np.ndarray) -> np.ndarray:
    """
    Compute the intersection over union (IoU) matrix between two images.

    Args:
        image_1: The first image to compare.
        image_2: The second image to compare.

    Returns:
        A numpy array representing the IoU matrix between the two images.
    """
    # Compute the number of pixels in each image
    counts_1 = np.bincount(image_1.ravel("K"))
    counts_2 = np.bincount(image_2.ravel("K"))

    # Compute the shape of the IoU matrix
    shape = (len(counts_1), len(counts_2))

    # Compute the linear indices of the pixels in both images
    linear_ids = np.ravel_multi_index((image_1, image_2), shape)

    # Compute the intersection and union of the pixels
    intersection = np.bincount(linear_ids.ravel(), minlength=np.product(shape))
    intersection = intersection.reshape(shape)
    union = counts_1[:, np.newaxis] + counts_2 - intersection

    # Initialize the IoU matrix with zeros
    iou = np.zeros(shape, float)

    # Compute the IoU for each pixel
    np.divide(intersection, union, out=iou, where=(union != 0))

    # Set the background pixels to zero
    iou[0, :] = iou[:, 0] = 0

    # Return all but background
    return iou[1:, 1:]


def compute_recall(TP: int, FN: int) -> float:
    """Computes the recall from true positives and false negatives.

    Args:
        TP (int): True positives.
        FN (int): False negatives.

    Returns:
        float: Recall score.
    """
    if TP + FN == 0:
        return 0
    return float(TP) / float(TP + FN)


def compute_precision(TP: int, FP: int) -> float:
    """Compute precision from true positives and false positives.

    Args:
        TP (int): Number of true positives.
        FP (int): Number of false positives.

    Returns:
        float: Precision score.
    """
    if TP + FP == 0:
        return 0
    return float(TP) / float(TP + FP)


def compute_fmeasure(recall: float, precision: float, fbeta: float) -> float:
    """Compute F-measure given recall, precision, and beta.

    Args:
        recall (float): Recall value.
        precision (float): Precision value.
        fbeta (float): Beta parameter for F-measure calculation.

    Returns:
        float: F-measure value.
    """
    if precision + recall == 0:
        return 0.0
    beta_sqr = fbeta**2
    return (1 + beta_sqr) * recall * precision / (beta_sqr * recall + precision)


def wrap_metrics(stats: Dict[str, float], fbeta: float = 1.0) -> Dict[str, float]:
    """
    Wrap metrics for a given set of statistics.

    Args:
        stats (Dict[str, float]): A dictionary containing the statistics to be wrapped.
        fbeta (float, optional): The beta value to use in the F-measure computation. Defaults to 1.0.

    Returns:
        Dict[str, float]: A dictionary containing the wrapped metrics.
    """
    precision = compute_precision(stats["tp"], stats["fp"])
    recall = compute_recall(stats["tp"], stats["fn"])
    fmeasure = compute_fmeasure(recall, precision, fbeta=fbeta)
    stats["precision"] = precision
    stats["recall"] = recall
    stats["fmeasure"] = fmeasure
    return stats


def get_centroids_distance(true_centroids: np.ndarray, pred_centroids: np.ndarray) -> np.ndarray:
    """
    Calculate the distance between the true centroids and the predicted centroids.

    Parameters
    ----------
    true_centroids : np.ndarray
        An array of true centroids.
    pred_centroids : np.ndarray
        An array of predicted centroids.

    Returns
    -------
    dist_mat : np.ndarray
        A matrix of distances between the true and predicted centroids.
    """
    # M = len(true), N = len(pred)
    # Dist mat = MxN
    true_centroids = np.array(true_centroids)
    pred_centroids = np.array(pred_centroids)

    dist_mat = cdist(true_centroids, pred_centroids, "euclidean")

    return dist_mat


def iterative_matching(dist_mat: np.ndarray, max_distance: float) -> List[Tuple[int, int]]:
    """
    Iteratively match items in a distance matrix using a threshold value.

    Parameters
    ----------
    dist_mat : np.ndarray
        The distance matrix to be matched.
    max_distance : float
        The maximum distance between two items for them to be considered a match.

    Returns
    -------
    matched_items : List[Tuple[int, int]]
        A list of tuples containing the indices of the matched items in the distance matrix.
    """
    matched_items = []

    cdist_mat = dist_mat.copy()
    max_value = np.max(cdist_mat) + 1

    while np.min(cdist_mat) < max_value:
        # Find the min value
        i, j = np.unravel_index(cdist_mat.argmin(), cdist_mat.shape)

        # If the minimum distance is above the threshold it means
        # that we won't have anymore matches so we stop
        if cdist_mat[i, j] >= max_distance:
            break

        matched_items.append((i, j))

        # "Disable" the true and pred items by setting their distance to
        # the max value + 1
        # This is a trick to avoid counting true and pred elements twice
        cdist_mat[i, :] = max_value
        cdist_mat[:, j] = max_value

    return matched_items


def dilate_img(y_pred: np.ndarray) -> np.ndarray:
    """Dilates an image using a 3x3 kernel. This is performed on the first axis which should be the time axis.

    Args:
        y_pred (np.ndarray): The input image to be dilated.

    Returns:
        np.ndarray: The dilated image.
    """
    dilated = np.zeros_like(y_pred)
    for i in range(y_pred.shape[0]):
        left_idx = max(0, i - 1)
        right_idx = min(y_pred.shape[0] - 1, i + 1)
        left_dilation = np.maximum(y_pred[left_idx], y_pred[i])
        right_dilation = np.maximum(y_pred[right_idx], y_pred[i])
        dilated[i] = np.maximum(left_dilation, right_dilation)
    return dilated


def get_sphere_vol(radius: float) -> float:
    """
    Calculates the volume of a sphere.

    Parameters:
        radius (float): The radius of the sphere.

    Returns:
        float: The volume of the sphere.
    """
    return (4 / 3) * np.pi * (radius**3)


def filter_by_object_weighted_prob(mat, label_mat, stats, min_weighted_prob, maximum_size):
    # filter ccs by weighted proba and size
    # Iterate through each connected components to check their weighted prob
    for i in range(len(stats["voxel_counts"])):
        if i == 0:
            continue

        # Compute the weighted probability (by maximum theorical size)
        size_ratio = stats["voxel_counts"][i] / maximum_size
        prob = stats["mean_prob"][i]
        weighted_prob = size_ratio * prob

        if weighted_prob < min_weighted_prob:
            # Make the cc to background if filtered out
            mat[np.where(label_mat == stats["values"][i])] = 0
    return mat


def evaluate_at_object_level(
    y_pred,
    y_true,
    threshold,
    min_weighted_prob,
    movie_name=None,
    output_dir=None,
    distance_mode="iou",
    distance_threshold=0.1,
):
    assert (
        y_pred.min() >= 0.0 and y_pred.max() <= 1.0
    ), f"Prediction matrix must have values in [0;1] but found [{y_pred.min()};{y_pred.max()}]"
    assert (
        y_true.min() >= 0.0 and y_true.max() <= 1.0
    ), f"Truth matrix must have values in [0;1] but found [{y_true.min()};{y_true.max()}]"
    assert (
        np.unique(y_true).shape[0] <= 2
    ), f"Label matrix must only have 0s or 1s but found: {np.unique(y_true)}"

    default_stats = {
        "matched_items": [],
        "distance_matrix": [],
        "pred_ccs_stats": [],
        "true_ccs_stats": [],
    }

    # Perform time dilatation to take into account the time axis errors
    y_pred_cpy = dilate_img(y_pred)

    # Find predicted connected components
    binary_pred = (y_pred_cpy >= threshold).astype(np.uint8)
    pred_ccs, pred_n = connected_components(binary_pred, return_N=True)
    pred_ccs_stats = statistics_optimized(pred_ccs, y_pred_cpy)

    # Maximum theorical volume is 3 time the sphere volume
    # It can also go above depending on the prediction
    maximum_size = get_sphere_vol(radius=8) * 3

    binary_pred = filter_by_object_weighted_prob(
        binary_pred, pred_ccs, pred_ccs_stats, min_weighted_prob, maximum_size
    )

    # Update with remaining ccs
    pred_n = np.unique(pred_ccs).shape[0] - 1
    pred_ccs, pred_n = connected_components(binary_pred, return_N=True)
    pred_ccs_stats = statistics_optimized(pred_ccs, y_pred_cpy)

    # Extract connected components and compute stats for ground truth
    y_true = (y_true >= 1.0).astype(np.uint8)
    true_ccs, true_n = connected_components(y_true, return_N=True)

    true_ccs_stats = statistics_optimized(true_ccs, y_true)

    if pred_n == 0:
        # No tp, no fp, all fn
        return wrap_metrics({"tp": 0, "fp": 0, "fn": true_n}), default_stats

    if true_n == 0:
        # No tp, all fp, no fn
        return wrap_metrics({"tp": 0, "fp": pred_n, "fn": 0}), default_stats

    # Iterative matching must optimize either iou or centroid distance based
    # However to optimize the iou we must find the maximum iou between two candidates
    # And to optimize the distance we must find the minimum distance between two candidates
    # In order to use the same algorithm we can transform the iou optimization into a minimization
    # problem by using 1.0 - iou and the threshold is then 1.0 - iou_threshold

    if distance_mode == "iou":
        distance_matrix = 1.0 - compute_iou_matrix(true_ccs, pred_ccs)
        distance_threshold = 1.0 - distance_threshold
    elif distance_mode == "centroid":
        distance_matrix = get_centroids_distance(
            true_ccs_stats["centroids"], pred_ccs_stats["centroids"]
        )
    else:
        raise ValueError(
            f"Unknown distance mode {distance_mode}; must be either {'iou', 'centroid'}"
        )

    matched_items = iterative_matching(distance_matrix, distance_threshold)

    tp = len(matched_items)
    fp = distance_matrix.shape[1] - tp
    fn = distance_matrix.shape[0] - tp

    metrics = {"tp": tp, "fp": fp, "fn": fn}

    if output_dir is not None:
        display_eval_results(
            y_pred_cpy.shape, movie_name, true_ccs_stats, pred_ccs_stats, matched_items, output_dir
        )

    return wrap_metrics(metrics), {
        "matched_items": matched_items,
        "distance_matrix": distance_matrix,
        "pred_ccs_stats": pred_ccs_stats,
        "true_ccs_stats": true_ccs_stats,
    }


def display_eval_results(
    y_pred_shape, movie_name, true_ccs_stats, pred_ccs_stats, matched_items, output_dir
):
    output_path = os.path.join(output_dir, movie_name)
    os.makedirs(output_path, exist_ok=True)

    true_results = np.zeros(y_pred_shape, dtype=np.uint16)
    pred_results = np.zeros(y_pred_shape, dtype=np.uint16)
    fn_mat = np.zeros(y_pred_shape, dtype=np.uint16)
    fp_mat = np.zeros(y_pred_shape, dtype=np.uint16)

    matched_true = []
    matched_pred = []

    matched_idx = 2

    for item in matched_items:
        true_idx, pred_idx = item
        true_centroid = true_ccs_stats["centroids"][true_idx]
        pred_centroid = pred_ccs_stats["centroids"][pred_idx]

        matched_true.append(true_idx)
        matched_pred.append(pred_idx)

        true_results = set_mat_value(true_results, true_centroid, matched_idx)
        pred_results = set_mat_value(pred_results, pred_centroid, matched_idx)

        matched_idx += 1

    for true_ccs_idx in range(len(true_ccs_stats["centroids"])):
        if true_ccs_idx not in matched_true:
            true_results = set_mat_value(true_results, true_ccs_stats["centroids"][true_ccs_idx], 1)
            fn_mat = set_mat_value(fn_mat, true_ccs_stats["centroids"][true_ccs_idx], 1)

    for pred_ccs_idx in range(len(pred_ccs_stats["centroids"])):
        if pred_ccs_idx not in matched_pred:
            pred_results = set_mat_value(pred_results, pred_ccs_stats["centroids"][pred_ccs_idx], 1)
            fp_mat = set_mat_value(fp_mat, pred_ccs_stats["centroids"][pred_ccs_idx], 1)

    unmatched_pred_stats = {"size": [], "prob": [], "mixed": []}
    matched_pred_stats = {"size": [], "prob": [], "mixed": []}

    maximum_size = get_sphere_vol(radius=8) * 3  # volume times 3 since we dilated t-1 & t+1

    for pred_ccs_idx in range(len(pred_ccs_stats["centroids"])):
        l = matched_pred_stats
        if pred_ccs_idx not in matched_pred:
            l = unmatched_pred_stats
        vox_count = pred_ccs_stats["voxel_counts"][pred_ccs_idx]
        mean_prob = pred_ccs_stats["mean_prob"][pred_ccs_idx]
        l["size"].append(vox_count / maximum_size)
        l["prob"].append(mean_prob)
        l["mixed"].append((vox_count / maximum_size) * mean_prob)

    # from matplotlib import pyplot as plt

    # fig, ((ax0, ax1, ax2)) = plt.subplots(nrows=1, ncols=3)
    # colors = ["red", "green"]
    # label = ["fp", "tp"]
    # ax0.hist(
    #     [unmatched_pred_stats["size"], matched_pred_stats["size"]],
    #     bins=50,
    #     stacked=True,
    #     color=colors,
    #     label=label,
    # )
    # ax0.legend(prop={"size": 10})
    # ax0.set_title("Size distribution")
    # ax1.hist(
    #     [unmatched_pred_stats["prob"], matched_pred_stats["prob"]],
    #     bins=50,
    #     stacked=True,
    #     color=colors,
    #     label=label,
    # )
    # ax1.legend(prop={"size": 10})
    # ax1.set_title("Probability distribution")
    # ax2.hist(
    #     [unmatched_pred_stats["mixed"], matched_pred_stats["mixed"]],
    #     bins=50,
    #     stacked=True,
    #     color=colors,
    #     label=label,
    # )
    # ax2.legend(prop={"size": 10})
    # ax2.set_title("Size*Probability distribution")
    # plt.savefig(os.path.join(output_path, "stats.png"))

    pred_results = np.swapaxes(pred_results, -1, -3)
    true_results = np.swapaxes(true_results, -1, -3)
    fp_mat = np.swapaxes(fp_mat, -1, -3)
    fn_mat = np.swapaxes(fn_mat, -1, -3)

    io.imsave(os.path.join(output_path, f"{movie_name}_true.tif"), true_results)
    io.imsave(os.path.join(output_path, f"{movie_name}_pred.tif"), pred_results)
    io.imsave(os.path.join(output_path, f"{movie_name}_fp.tif"), fp_mat)
    io.imsave(os.path.join(output_path, f"{movie_name}_fn.tif"), fn_mat)


def set_mat_value(mat, center, value):
    t, x, y, z = center
    t, x, y, z = (
        int(np.round(t + 1e-9)),
        int(np.round(x + 1e-9)),
        int(np.round(y + 1e-9)),
        int(np.round(z + 1e-9)),
    )
    mat[t, x - 4 : x + 4, y - 4 : y + 4, z - 4 : z + 4] = value
    return mat


def evaluator_wrapper(
    method, multithread=False, iteration_method="sample", num_workers=12, **kwargs
):
    def item_wrapper(params):
        y_pred, y_true, threshold, min_weighted_prob, movie_name, output_dir = params
        kwargs["y_true"] = y_true
        kwargs["y_pred"] = y_pred
        kwargs["threshold"] = threshold
        kwargs["movie_name"] = movie_name
        kwargs["min_weighted_prob"] = min_weighted_prob
        kwargs["output_dir"] = output_dir
        return method(**kwargs)

    def wrapper(
        y_pred_movie,
        y_true_movie,
        threshold,
        min_weighted_prob=10,
        movies_names=None,
        output_dir=None,
    ):
        if iteration_method == "sample":
            # Order matters here
            movies_names = [
                f"{movies_names[i]}_{j}"
                for i in range(len(movies_names))
                for j in range(y_true_movie[i].shape[0])
            ]
            y_true_movie = [sample for movie in y_true_movie for sample in movie]
            y_pred_movie = [sample for movie in y_pred_movie for sample in movie]
        elif iteration_method == "movie":
            pass
        else:
            raise ValueError("Unknown iteration method; must be either {'sample', 'movie'}")

        params = [
            (
                y_pred_movie[i],
                y_true_movie[i],
                threshold,
                min_weighted_prob,
                movies_names[i] if movies_names is not None else None,
                output_dir,
            )
            for i in range(len(y_true_movie))
        ]
        if multithread:
            # Make the Pool of workers
            from multiprocessing.dummy import Pool as ThreadPool

            pool = ThreadPool(num_workers)

            results = pool.map(item_wrapper, params)

            # Close the pool and wait for the work to finish
            pool.close()
            pool.join()
        else:
            results = []
            for param in tqdm(params, desc="Evaluating...", leave=False):
                results.append(item_wrapper(param))

        return results

    return wrapper


def eval_at_threshold(method, return_all=False):
    def wrapper(*args, **kwargs):
        # Values are return in a list of pair
        all_values = method(*args, **kwargs)

        # Split the list of pair into a pair of list
        results, info = [], []
        for local_results, local_info in all_values:
            results.append(local_results)
            info.append(local_info)

        TP = sum([stat["tp"] for stat in results])
        FP = sum([stat["fp"] for stat in results])
        FN = sum([stat["fn"] for stat in results])

        precision = compute_precision(TP, FP)
        recall = compute_recall(TP, FN)
        fmeasure = compute_fmeasure(recall, precision, fbeta=1.0)
        if return_all:
            return {
                "tp": TP,
                "fp": FP,
                "fn": FN,
                "precision": precision,
                "recall": recall,
                "fmeasure": fmeasure,
            }, info
        return fmeasure

    return wrapper


def find_best_threshold(
    y_pred,
    y_true,
    evaluation_method,
    maxiter=30,
    verbose=1,
):
    with tqdm(total=maxiter, disable=(verbose != 1), desc="threshold = ") as progress:

        def fn(params):
            thr, wprob = params
            value = evaluation_method(
                y_pred,
                y_true,
                thr,
                wprob,
            )

            progress.update()
            progress.set_postfix_str(
                "threshold@{prob_thresh:.3f};wprob@{wprob:.3f} -> F1@{value:.3f}".format(
                    prob_thresh=thr, wprob=wprob, value=value
                )
            )
            progress.refresh()
            return value

        # Perform grid search
        best_value = -1.0
        best_params = None
        for th in np.arange(0.1, 0.71, 0.15):
            for wp in np.arange(0, 0.76, 0.15):
                value = fn((th, wp))
                if value > best_value:
                    print(f"New best value @{value} with params @({th},{wp})")
                    best_value = value
                    best_params = (th, wp)
    return best_params
