import os
import numpy as np
from matplotlib import pyplot as plt
from dare3d.metrics.object_level import (
    evaluator_wrapper,
    eval_at_threshold,
    find_best_threshold,
    evaluate_at_object_level,
)
from dare3d.metrics.inference import (
    segmentation_inference,
    regression_inference,
    display_regression,
)
from dare3d.losses.angle3d import quaternion_error


def evaluate_segmentation(
    y_pred_movies,
    y_true_movies,
    movies_names,
    multithread=False,
    threshold=None,
    iteration_method="sample",
    distance_mode="iou",
    distance_threshold=0.1,
    min_weighted_prob=0.1,
    output_dir=None,
):
    # This method performs the evaluation
    method = evaluator_wrapper(
        evaluate_at_object_level,
        iteration_method=iteration_method,
        multithread=multithread,
        distance_mode=distance_mode,
        distance_threshold=distance_threshold,
        min_weighted_prob=min_weighted_prob,
    )

    # This takes the evaluation results and return the global fmeasure
    fmeasure_wrapper = eval_at_threshold(method)

    if threshold is None or not isinstance(threshold, float):
        # This uses the fmeasure wrapper during optimisation to find the best threshold
        threshold, min_weighted_prob = find_best_threshold(
            y_pred_movies, y_true_movies, fmeasure_wrapper
        )

    metrics_wrapper = eval_at_threshold(method, return_all=True)

    metrics_dict, info = metrics_wrapper(
        y_pred_movies,
        y_true_movies,
        threshold,
        movies_names=movies_names,
        min_weighted_prob=min_weighted_prob,
        output_dir=output_dir,
    )
    metrics_dict["threshold"] = threshold
    metrics_dict["min_weighted_prob"] = min_weighted_prob
    return metrics_dict, info


def infer_and_evaluate_segmentation(
    dataset,
    model,
    device,
    crop_size,
    batch_size,
    multithread=False,
    threshold=None,
    iteration_method="sample",
    distance_mode="iou",
    distance_threshold=0.1,
    min_weighted_prob=0.1,
    output_dir=None,
):
    # Shape is M, T, X, Y, Z
    y_true_movies = [movie.copy() for movie in dataset.movies_masks]

    # Make sure the two first time step of the groundtruth is at zero
    for j in range(len(y_true_movies)):
        for k in range(dataset.n_input_channels - 1):
            y_true_movies[j][k] = np.zeros_like(y_true_movies[j][k])

    y_pred_movies = segmentation_inference(
        dataset=dataset,
        model=model,
        device=device,
        crop_size=crop_size,
        batch_size=batch_size,
        output_dir=output_dir,
    )
    y_pred_movies = [np.swapaxes(pred, -1, -3) for pred in y_pred_movies]

    # start_index = dataset.n_input_channels - 1
    # y_true_movies = [movie[start_index:] for movie in y_true_movies]
    # y_pred_movies = [movie[start_index:] for movie in y_pred_movies]

    return evaluate_segmentation(
        y_pred_movies,
        y_true_movies,
        dataset.movie_names,
        multithread,
        threshold,
        iteration_method,
        distance_mode,
        distance_threshold,
        min_weighted_prob,
        output_dir,
    )


def infer_eval_wrapper(
    device, batch_size, threshold, distance_mode, distance_threshold, min_weighted_prob
):
    def wrapper(dataset, model):
        return infer_and_evaluate_segmentation(
            dataset,
            model,
            device,
            batch_size,
            threshold,
            distance_mode,
            distance_threshold,
            min_weighted_prob,
        )

    return wrapper


class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            del self.inverse[self[key]]
            # self.inverse[self[key]].remove(key)
        super(bidict, self).__setitem__(key, value)
        self.inverse[self[key]] = key
        # self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        del self.inverse[self[key]]
        # self.inverse.setdefault(self[key], []).remove(key)
        # if self[key] in self.inverse and not self.inverse[self[key]]:
        # del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)


class CenterList:

    def __init__(self, movie_start_time, movie_info):
        self.movie_start_time = movie_start_time

        # Predicted centers
        self.predicted_centers = []

        # Real annotated centers
        self.true_centers = []

        # Bidirectionnal map of centers idx
        # <true,pred> and if using map.inverse => <pred, true>
        self.matched_centers_idx = bidict()

        self.real_rot_length = None
        self.all_gt_centers = []

        self.load_info(movie_info)

    def load_info(self, info):
        # The goal of this function is to create two list of centers (prediction and groundtruth)
        # A center represents one cell division (the middle point of the two daughter cells)
        # Each center is a vector of size 5: (m, t, x, y, z) with m the movie index, t the time index
        # and x,y,z the spatial coordinates
        # Then we create a bidirectional map to find if a given center has a corresponding match
        # The two center lists (prediction & groundtruth) order must be kept intact so the map
        # that find the matched centers is accurate

        for i, movie_info in enumerate(info):
            movie_matched_items = movie_info["matched_items"]
            if len(movie_info["pred_ccs_stats"]) == 0 or len(movie_info["true_ccs_stats"]) == 0:
                continue
            movie_pred_centers = movie_info["pred_ccs_stats"]["centroids"]
            movie_true_centers = movie_info["true_ccs_stats"]["centroids"]

            for center in movie_true_centers:
                self.all_gt_centers.append(
                    (
                        i,
                        int(np.rint(center[0] + self.movie_start_time)),
                        int(np.rint(center[1])),
                        int(np.rint(center[2])),
                        int(np.rint(center[3])),
                    )
                )

            # <a,b> = <true, pred>
            matching_index = bidict()
            m_i = 0
            for item in movie_matched_items:
                a, b = item
                matching_index[a] = b

                true_center = movie_true_centers[a]
                pred_center = movie_pred_centers[b]

                pred_center = (
                    i,
                    int(np.round(pred_center[0] + 1e-9 + self.movie_start_time)),
                    int(np.round(pred_center[1] + 1e-9)),
                    int(np.round(pred_center[2] + 1e-9)),
                    int(np.round(pred_center[3] + 1e-9)),
                )
                true_center = (
                    i,
                    true_center[0] + self.movie_start_time,
                    true_center[1],
                    true_center[2],
                    true_center[3],
                )

                self.predicted_centers.append(pred_center)
                self.true_centers.append(true_center)

                self.matched_centers_idx[m_i] = m_i
                m_i += 1

        print(f"Number of matched centers : {len(self.matched_centers_idx)}")

    def compute_real_rot_len_values(self, dataset):
        self.real_rot_length = dataset.gather_groundtruth_info(self.all_gt_centers)
        self.real_rot_length_matched = dataset.gather_groundtruth_info(self.true_centers)

    def create_gt_pairs(self, true_centers):
        assert self.real_rot_length
        assert len(self.real_rot_length) == len(true_centers)
        # It compares all groundtruth centers with real annotated values
        # This is to evaluate the performance of the regression using a perfect detector

        # Compare self.real_rot_length with self.true_centers
        # Both lists already have the same size and order

        pairs = list(zip(self.real_rot_length, true_centers))
        return pairs

    def create_matched_gt_pairs(self, true_centers):
        assert self.real_rot_length_matched
        assert len(self.real_rot_length_matched) == len(true_centers)
        # It compares the groundtruth centers that have been matched with a prediction center
        # by comparing the angle and len against the real annotated values
        # This is to compare the impact of the segmentation imprecision on the regression performance

        pairs = []
        for j in self.matched_centers_idx.keys():
            true_center = true_centers[j]
            pairs.append((self.real_rot_length_matched[j], true_center))
        return pairs

    def create_pred_gt_pairs(self, pred_centers):
        assert self.real_rot_length_matched

        # It compares predicted centers that have been matched (ie valid ones)
        # This is the real case scenario which provides the actual performance of the whole system
        pairs = []
        for j in self.matched_centers_idx.inverse.keys():
            pred_center = pred_centers[j]
            pairs.append((self.real_rot_length_matched[j], pred_center))
        return pairs


def infer_and_evaluate_regression(dataset, net, device, info, output_dir):
    movie_start = 0
    center_list = CenterList(movie_start, info)

    # Perform angle/length inference on the images at the given centers
    # For predicted centers
    pred_output_dir = os.path.join(output_dir, "pred_centers")
    pred_angle_length = regression_inference(
        dataset, net, center_list.predicted_centers, device, output_dir=pred_output_dir
    )

    # For matched groundtruth centers
    gt_output_dir = os.path.join(output_dir, "matched_true_centers")
    gt_angle_length = regression_inference(
        dataset, net, center_list.true_centers, device, output_dir=gt_output_dir
    )

    # For all groundtruth centers
    all_gt_output_dir = os.path.join(output_dir, "all_true_centers")
    all_gt_angle_length = regression_inference(
        dataset, net, center_list.all_gt_centers, device, output_dir=all_gt_output_dir
    )

    # Then, gather the real angle and length from the ground truth centers
    center_list.compute_real_rot_len_values(dataset)

    annotated_regression_angles_dir = os.path.join(output_dir, "groundtruth")
    display_regression(center_list.real_rot_length, dataset, annotated_regression_angles_dir)

    gt_gt_pairs = center_list.create_gt_pairs(all_gt_angle_length)
    gt_gt_matched_pairs = center_list.create_matched_gt_pairs(gt_angle_length)
    gt_pred_pairs = center_list.create_pred_gt_pairs(pred_angle_length)

    gt_gt_stats, a_de, a_tde, a_ae, a_le = evaluate_center_pair(gt_gt_pairs)
    gt_gt_matched_stats, b_de, b_tde, b_ae, b_le = evaluate_center_pair(gt_gt_matched_pairs)
    gt_pred_stats, c_de, c_tde, c_ae, c_le = evaluate_center_pair(gt_pred_pairs)

    # fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=3, figsize=(20, 10))
    # plot_errors(ax0, a_de, a_tde, a_ae, a_le)
    # plot_errors(ax1, b_de, b_tde, b_ae, b_le)
    # plot_errors(ax2, c_de, c_tde, c_ae, c_le)

    # output_path = os.path.join(output_dir, f"regression_error_analysis.png")
    # print(f"Saving angle/length stats to path: {output_path}")
    # plt.savefig(output_path, format="png")

    return {
        "regression_performance_on_all_groundtruth_centers": gt_gt_stats,
        "regression_performance_on_matched_groundtruth_centers": gt_gt_matched_stats,
        "regression_performance_on_predicted_centers": gt_pred_stats,
    }


def plot_errors(ax, distance_errors, t_distance_errors, angle_errors, length_errors):
    ax0, ax1, ax2 = ax
    ax0.set_ylim(0, 180)
    ax0.set_xlim(0, 6)
    ax0.scatter(distance_errors, angle_errors)
    ax0.set_title("Angle error by center dst")

    ax1.set_ylim(0, 10)
    ax1.set_xlim(0, 6)
    ax1.scatter(distance_errors, length_errors)
    ax1.set_title("Length error by center dst")

    ax2.set_xlim(0, 3)
    ax2.set_ylim(0, 180)
    ax2.scatter(t_distance_errors, angle_errors)
    ax2.set_title("Angle error by T dst error")


def get_mean_std(value):
    return np.mean(value, dtype=np.float32), np.std(value, dtype=np.float32)


def get_mean_curve(y_array, x_array, cast_x_to_int=True):
    if cast_x_to_int:
        x_array = [int(np.rint(x)) for x in x_array]

    lookup_dict = {}
    for y, x in zip(y_array, x_array):
        if x not in lookup_dict:
            lookup_dict[x] = []
        lookup_dict[x].append(y)

    uniques_x = np.unique(list(lookup_dict.keys()))

    x = []
    y = []
    for unique_x in sorted(uniques_x):
        x.append(unique_x)
        y.append(np.mean(lookup_dict[unique_x]))

    return x, y


def evaluate_center_pair(center_pair):
    angle_errors = []
    length_errors = []
    distance_errors = []
    t_distance_errors = []
    for true, pred in center_pair:
        # This case occurs when we failed to match a true predicted center
        # to an annotated one (mostly due to groundtruth fusion)
        # In this case the true center is none and cannot be compared
        if true is None or pred is None:
            continue
        angle_error = quaternion_error(true["rotation"], pred["rotation"])
        length_error = abs(true["length"] - pred["length"])
        distance_error = np.linalg.norm(np.array(true["center"]) - np.array(pred["center"]))
        t_distance_error = abs(true["center"][1] - pred["center"][1])

        angle_errors.append(angle_error)
        length_errors.append(length_error[0])
        distance_errors.append(distance_error)
        t_distance_errors.append(t_distance_error)

    mean_angle_error, std_angle_error = get_mean_std(angle_errors)
    mean_length_error, std_length_error = get_mean_std(length_errors)
    mean_distance_error, std_distance_error = get_mean_std(distance_errors)

    n_centers = len(center_pair)

    return (
        {
            "n": n_centers,
            "mean_angle_error": mean_angle_error,
            "std_angle_error": std_angle_error,
            "mean_length_error": mean_length_error,
            "std_length_error": std_length_error,
            "mean_distance_error": mean_distance_error,
            "std_distance_error": std_distance_error,
        },
        distance_errors,
        t_distance_errors,
        angle_errors,
        length_errors,
    )
