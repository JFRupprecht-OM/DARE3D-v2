# Archived movie_I2 probability-map provenance

Finding: the existing epoch057 checkpoint, with native-grid segmentation preprocessing, reproduces the archived event result exactly and its probability map to very high numerical agreement. The exact original Hydra evaluation launch manifest and workstation Git/runtime state were not recovered. This is a verified compatible producer recipe, not a fabricated historical config.

## Exact files

Legacy checkpoint: [epoch_057.ckpt](C:/Users/ruppr/Documents/Codexsession/DARE3D_280826/DARE3d_data_190326/Neural_tube_160226/weights/segmentation3d_new_set_og/runs/12-01-26/checkpoints/epoch_057.ckpt)

Path: `C:\Users\ruppr\Documents\Codexsession\DARE3D_280826\DARE3d_data_190326\Neural_tube_160226\weights\segmentation3d_new_set_og\runs\12-01-26\checkpoints\epoch_057.ckpt`

SHA-256: `8261c70f41c7f0f5184559e98f3c69ad336e5e46edba5bb57e5edbf9b0568f7d`

Identical promoted checkpoint: [DARE3D_neural_tube_segmentation_epoch057.ckpt](C:/Users/ruppr/Documents/Codexsession/DARE3D_280826/DARE3dv2_Zenodo_040926/DARE3D_neural_tube_segmentation_epoch057.ckpt)

Path: `C:\Users\ruppr\Documents\Codexsession\DARE3D_280826\DARE3dv2_Zenodo_040926\DARE3D_neural_tube_segmentation_epoch057.ckpt`

SHA-256: `8261c70f41c7f0f5184559e98f3c69ad336e5e46edba5bb57e5edbf9b0568f7d`

Genuine bundled Hydra base config: [config.yaml](C:/Users/ruppr/Documents/Codexsession/DARE3D_280826/DARE3d_data_190326/Neural_tube_160226/weights/segmentation3d_new_set_og/runs/12-01-26/.hydra/config.yaml)

Path: `C:\Users\ruppr\Documents\Codexsession\DARE3D_280826\DARE3d_data_190326\Neural_tube_160226\weights\segmentation3d_new_set_og\runs\12-01-26\.hydra\config.yaml`

SHA-256: `0c6aebaafad7cdc43254ec06261679f8f6bfee591bf842f2002b9768d5966761`

Archived probability map: [movie_I2.tif](C:/Users/ruppr/Documents/Codexsession/DARE3D_280826/DARE3d_data_190326/Neural_tube_160226/weights/segmentation3d_new_set_og/runs/12-01-26/movie_I2.tif)

Path: `C:\Users\ruppr\Documents\Codexsession\DARE3D_280826\DARE3d_data_190326\Neural_tube_160226\weights\segmentation3d_new_set_og\runs\12-01-26\movie_I2.tif`

SHA-256: `8a54807c1ee1748e55038939aead7750ec696d9c08e9f44357076e6d4855546f`

The Zenodo copy of .hydra/config.yaml is byte-identical to the native config. The sibling hydra.yaml and overrides.yaml are genuine bundled training artifacts, but no separate original evaluation .hydra directory was found. Their hashes and earliest-public-archive identity checks are in earliest_release_member_hashes.json.

## Why the bundled config is not an exact producer manifest

The checkpoint internally records training at `C:\Users\tlili\Documents\TRIAL\DARE3d\logs\segmentation3d_new_set_og\runs\05-02-26\checkpoints\epoch_057.ckpt` (epoch 57, global step 3654, Lightning 2.5.6). The bundled config/log/TensorBoard describe a January run: 32 steps/epoch and best validation epoch 47, whereas the selected checkpoint has 63 steps/epoch and best epoch 57. This mismatch remains real; it does not imply the epoch057 network cannot generate the archived map.

The earliest public ZIP preserves January 12 Hydra files, a February 8 epoch057 checkpoint, and a February 13 probability map. The map/config/stats bytes are identical to the current archive. Across all five public versions the indexed neural artifacts retain identical sizes and CRC32; full SHA-256 was verified for the current local ZIP members and the earliest remote map/small provenance files, not every remote checkpoint. ZIP timestamps have no timezone and are only supporting clues. [Earliest Zenodo release](https://zenodo.org/records/17456474), [current archived release](https://zenodo.org/records/19113351).

## Recovered effective preprocessing and code

| Setting | Verified replay |
|---|---|
| Spatial grid | Native XYZ 1024 x 1024 x 10; no effective spatial resizing |
| Computational scale | Test-dataset default_scale [1,1,1], target_scale 1, no scale-table entry |
| Padding | 59 zero planes on each side of Z, giving CXYZ 3 x 1024 x 1024 x 128 |
| Temporal input | Causal (t-2,t-1,t), output t=2..20; first two output frames zero |
| Normalization | Min-max over each padded three-frame sequence; float32 network input |
| Model | Genuine Hydra MultiScaleUNet; 3 input channels, 1 output; all 196 net state entries loaded strictly, eval mode |
| Sliding window | 128-cubed patches, Gaussian blending, overlap 0.5; replay batch 12 |
| Output | Stitch primary logits, then sigmoid, float16 conversion, remove padding, restore TZYX |

These computational unit ratios are NOT a claim of 1-micrometer physical voxels. Literal historical default/target values or a workstation-only scale-table override are not uniquely identifiable from integer output dimensions. Batch 12 is the verification setting, not a recovered original command. Labels were disabled during inference and used only afterward for fixed scoring.

The complete replay derives from the genuine saved model config with only the documented test-path/geometry/inference overrides: recovered_effective_recipe.json, full_native_replay_preflight.json, and executed_prov_full_native_replay.json. This reconstructed recipe must not be presented as an original .hydra/config.yaml.

The replay imports the genuine read-only older checkout at `C:\Users\ruppr\Documents\Codexsession\older\DARE3\DARE3d`, HEAD `fffadb27062a844a2f0798bd392a8f1cf80dc95e`. Its relevant core files match the pre-archive public revision [60b565483178cb5444a4388e3c5ca0d5b28e517f](https://github.com/JFRupprecht-OM/DARE3d/tree/60b565483178cb5444a4388e3c5ca0d5b28e517f) after CRLF normalization. Per-file Git blobs and SHA-256 are in historical_inference_code_identity.json and historical_model_module_identity.json. Unchanged core files span multiple commits, so this does not establish the exact original checkout or uncommitted changes.

Replay runtime: torch 2.11.0+cu128, monai 1.3.2, numpy 1.26.4, scipy 1.15.3, scikit-image 0.25.2. No runtime version was claimed to be the original one. Historical and current object-level evaluator files are byte-identical; current segmentation inference adds only cancellation handling. No scientific model-code correction was needed for this replay.

## Full-movie verification

| Probability source | Historical centers | TP / FP / FN | Historical F1 | Actual Napari centers |
|---|---:|---|---:|---:|
| Archived movie_I2 | 122 | 114 / 8 / 8 | 93.4426% | 118 |
| Fresh epoch057, native grid | 122 | 114 / 8 / 8 | 93.4426% | 118 |
| Previous epoch057, saved 635 x 635 x 20 grid | 92 | 79 / 13 / 43 | 73.8318% | 74 |

Full replay: 99.886177% exactly equal voxels; mean absolute probability difference 5.96676820968e-07; maximum difference 0.016357421875; foreground correlation 0.999982646 to 0.999995690 over all 19 predicted frames. Only 106 voxels disagree at >0.5 and 123 at >=0.55 across 220,200,960 voxels. Foreground is assessed separately because overall equality is dominated by background.

All 114 matched event pairs are identical. Maximum centroid drift in TXYZ is [0.0015417209660348874, 0.013111888111886572, 0.018259518259469587, 0.02114089614089565] (frames/raw voxels). The probability TIFF is NOT byte-identical: new SHA-256 `8c94f42b5c3c6132965ddd7b817137cb5eb27bc96fd964fc69311f3ef5f83d8c`. Small numerical differences remain; their exact runtime cause was not isolated.

Historical scoring is fixed at >=0.55, weighted cutoff 0.15, +/-1 temporal dilation, 4D components, radius-8 volume times three, greedy IoU >1e-6. Actual Napari extraction is fixed at >0.5 and weighted cutoff 0.1 without dilation, executed once on cached new probabilities. No threshold fitting or regression inference was performed. The native-grid one-sequence probe was motivated by a resampling-lattice fingerprint; retaining a doubled Z grid did not match. Then one full native-grid replay was run.

This closes the archived movie_I2 numerical reproduction gap. It does not prove a native-grid fix for movie_M, establish physical I2 calibration, or resolve manuscript dataset attribution: the prior inspection found matching manuscript counts/F1 but its held-out-data table names movie_M, whereas this archived map is movie_I2 from the original validation input. No manuscript or validated result was changed.

## Preservation and handoff

Only new diagnostic artifacts in this directory were generated. All 3,835 protected files retain their size/mtime and the protected tree has no additions/removals; 33 selected SHA-256 values are unchanged. All 70 previously manifested investigation artifacts were rehashed unchanged. Models, Hydra files, source code, Napari settings, Zenodo, historical folders and validated audits remain untouched. No retrained segmenter was loaded, no training, regression inference, commit or push was performed.

Git remains main, four commits ahead of origin/main, with no tracked/staged changes. Pre-existing untracked Zenodo, audit, manuscript and handoff material plus `grep.exe.stackdump` were preserved. The prior focused suite reported 36 passed; it was not rerun in this code-unchanged provenance round. GPU replay and final fixed CPU scoring both completed successfully.

Next: obtain the original February training/evaluation Hydra output, actual source-workstation scales.json, command/log, Git worktree and environment record if an exact historical certificate is required. No further inference is needed to establish this I2 numerical recipe. Any production compatibility change, separate movie_M validation or post-processing change requires a subsequent scoped decision. Do not run historical eval.py against the protected legacy model_dir: it writes probabilities/stats there.

provenance_summary.json is the final machine-readable status. Earlier provenance_identifiability_limits.json was written while full replay was pending; it is retained unchanged as intermediate evidence. Initial *_zip_inventory_error.json files document a failed ZIP64 parsing attempt; use the later *_zip_inventory_complete.json records. executed_prov_*.json preserves the replay sources as diagnostic records; use a new output directory for any future execution.
