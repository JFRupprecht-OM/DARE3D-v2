import torch
import scipy
import numpy as np
import monai

from dare3d.metrics.object_level import statistics, get_centroids_distance, iterative_matching

# Threshold prediction
# Find all connected components
# Compute centroids

# Match centroids

# For all ground truth connected components that have not been matched = should be not background
# For all predicted connected components that have not been matched = should be background

def loss_wrapper(threshold=0.5, eps=1e-6, distance_threshold=10):

    # focal_loss = monai.losses.FocalLoss(include_background=True, to_onehot_y=False)
    base_loss = monai.losses.GeneralizedDiceFocalLoss(include_background=True, to_onehot_y=False, sigmoid=False)

    def loss(y_pred, y_true):
    
        def compute_weights(y_pred, y_true):
            print("Computing weights")
            device = y_true.device
            y_pred = torch.nn.functional.sigmoid(y_pred)
            y_true = y_true.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()
            
            y_pred = (y_pred > threshold).astype(np.uint8)
            y_true = (y_true >= 1.0).astype(np.uint8)
            
            print("Labeling images")
            true_ccs, n_true = scipy.ndimage.label(y_true)
            pred_ccs, n_pred = scipy.ndimage.label(y_pred)
            
            print(f"Compute stats with #true ccs: {n_true} and #pred ccs: {n_pred}")
            if n_pred > 500 or n_true == 0 or n_pred == 0:
                return torch.tensor(y_true, device=device)
            true_ccs_stats = statistics(true_ccs)
            pred_ccs_stats = statistics(pred_ccs)                
            
            print("Compute distance matrix")
            distance_matrix = get_centroids_distance(
                true_ccs_stats["centroids"][1:], pred_ccs_stats["centroids"][1:]
            )
            
            print("Iterative matching")
            matched_items = iterative_matching(distance_matrix, distance_threshold)
            if len(matched_items) != 0:
                matched_true, matched_pred = zip(*matched_items)
                
                # For all groundtruth ccs
                for i in range(distance_matrix.shape[0]):
                    label_id = i
                    if label_id in matched_true: # should be foreground
                        match_index = matched_true.index(label_id)
                        corresponding_pred_ccs = matched_pred[match_index]
                        
                        y_true[true_ccs == label_id + 1] = 0.0
                        y_true[pred_ccs == corresponding_pred_ccs + 1] = 1.0

                # For all pred ccs
                for j in range(distance_matrix.shape[1]):
                    label_id = j
                    if label_id not in matched_pred:
                        y_true[pred_ccs == label_id + 1] = 0.0
            
            return torch.tensor(y_true, device=device)
        
        def cross_entropy(y_pred, y_true):
            y_pred = torch.clip(y_pred, min=eps, max=1.0 - eps)
            y_pred = torch.log(y_pred / (1.0 - y_pred))

            zeros = torch.zeros_like(y_pred, dtype=y_pred.dtype)
            cond = y_pred >= zeros
            relu_logits = torch.where(cond, y_pred, zeros)
            neg_abs_logits = torch.where(cond, -y_pred, y_pred)

            entropy = relu_logits - y_pred * y_true + torch.log1p(torch.exp(neg_abs_logits))
            return entropy

        # Multiply by weights
        y_true = compute_weights(y_pred, y_true)

        # Compute cross entropy
        loss_ = base_loss(y_pred, y_true)

        return loss_                
        # return torch.mean(ce * w)
        # return torch.mean(loss)
    return loss