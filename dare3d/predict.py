import os
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

from dare3d.utils import RankedLogger, extras
from dare3d.metrics.inference import segmentation_inference, regression_inference
from dare3d.metrics.object_level import connected_components, statistics_optimized, get_sphere_vol, filter_by_object_weighted_prob

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = RankedLogger(__name__, rank_zero_only=True)

OmegaConf.register_new_resolver("eval", eval)

def load_data(cfg):
    model = hydra.utils.instantiate(cfg.model)
    # state_dict = torch.load(cfg.ckpt_path, map_location="cpu")["state_dict"]
    state_dict = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)["state_dict"] #qazi_change_23/10/25
    model.load_state_dict(state_dict)

    device=None
    if cfg.device == "gpu" or cfg.device == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
    model.net = model.net.to(device)
    model.net.eval()

    dataset = hydra.utils.instantiate(cfg.data.test_data)
    dataset.init(preprocess=False)
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    
    return dataset, model, device, output_dir

def do_segmentation(cfg):
    dataset, model, device, output_dir = load_data(cfg)
    output_dir = os.path.join(output_dir, "segmentation")
    predictions = segmentation_inference(dataset, model, device, cfg.crop_size, cfg.inference_batch_size, cfg.inference_overlap, output_dir=output_dir)
    
    centers = []
    for i, predicted_movie in enumerate(predictions):
        movie = np.swapaxes(predicted_movie, -1, -3)
        binary_pred = (movie > cfg.threshold).astype(np.uint8)
        # Label binary matrix & compute centroids
        pred_ccs = connected_components(binary_pred, return_N=False)        
        pred_ccs_stats = statistics_optimized(pred_ccs, movie)

        maximum_size = get_sphere_vol(radius=cfg.cell_radius) * 3
        binary_pred = filter_by_object_weighted_prob(binary_pred, pred_ccs, pred_ccs_stats, cfg.min_weighted_prob, maximum_size)

        # Update ccs after filtering
        pred_ccs = connected_components(binary_pred, return_N=False)        
        pred_ccs_stats = statistics_optimized(pred_ccs, movie)

        # Gather detected centers
        centroids = pred_ccs_stats["centroids"]
        for centroid in centroids:
            # /!\ Segmentation returns Z,Y,X format
            t,x,y,z = centroid
            centers.append((i,t,x,y,z))
    return centers

def do_regression(cfg, centers):
    dataset, model, device, output_dir = load_data(cfg)
    dataset.pad_images()
    dataset._normalize(dataset.renorm)

    output_dir = os.path.join(output_dir, "regression")
    predictions = regression_inference(dataset, model, centers, device, output_dir=output_dir)
    return predictions

def inference(seg_cfg: DictConfig, reg_cfg: DictConfig = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    centers = do_segmentation(seg_cfg)
    
    if reg_cfg is not None:
        do_regression(reg_cfg, centers)

def load_config(cfg, root_cfg, allow_missing=False):
    hydra_config_path = os.path.join(cfg.model_dir, cfg.hydra_dir, "config.yaml")
    checkpoint_config_path = os.path.join(cfg.model_dir, cfg.ckpt_dir, cfg.ckpt_name)
    
    if not os.path.exists(hydra_config_path):
        if allow_missing:
            return None
        raise ValueError(f"Could not find hydra config file path: {hydra_config_path}")
    if not os.path.exists(checkpoint_config_path):
        if allow_missing:
            return None
        raise ValueError(f"Could not find checkpoint path: {checkpoint_config_path}")
    train_cfg = OmegaConf.load(hydra_config_path)
    OmegaConf.set_struct(train_cfg, None)
    train_cfg.merge_with(cfg)
    cfg = train_cfg

    # Edit checkpoint path    
    cfg.ckpt_path = checkpoint_config_path

    # Edit inference folder in datamodule
    cfg.data.test_data.load_labels = False
    cfg.inference_dir = root_cfg.inference_dir
    cfg.device = root_cfg.device
    cfg.data.test_data.im_folder = cfg.inference_dir
    
    # Edit scale file
    if root_cfg.get("scale_file"):
        cfg.data.test_data.scale_file = root_cfg.scale_file
    if root_cfg.get("default_scale"):
        cfg.data.test_data.default_scale = root_cfg.default_scale
    if root_cfg.get("target_scale"):
        cfg.data.test_data.target_scale = root_cfg.target_scale
        
    return cfg

@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    seg_cfg = load_config(cfg.segmentation, cfg)
    reg_cfg = load_config(cfg.regression, cfg, allow_missing=True)

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(seg_cfg)
    if reg_cfg is not None:
        extras(reg_cfg)

    inference(seg_cfg, reg_cfg)


if __name__ == "__main__":
    main()