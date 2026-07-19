Two Dimensional Weak Measurement Model Code

Dataset location

Before running the training or evaluation scripts, manually place the packed dataset file in this same folder:

dataset_packed_221_v174.npy

The dataset index file is included in this repository:

dataset_index_fast_v174.csv

The scripts use their own folder as the default dataset directory. The packed NPY dataset is not included in this code package because it is large.

Code files

train_centroid_map_models.py
Trains the phase difference, amplitude ratio, and phase gradient models.

evaluate_centroid_map_models.py
Evaluates the trained models and exports prediction results. Trained checkpoint files must be supplied separately.

analyze_uniform_angle_centroid_maps.m
Performs the MATLAB analysis of uniformly sampled angle centroid maps.

export_paper_data.nb
Exports data used for the manuscript from Mathematica.

Model checkpoint files

Best_b.pth
Best model for amplitude ratio.

Ckpt_b.pth
Training checkpoint for amplitude ratio.

Best_phi.pth
Best model for phase difference.

Ckpt_phi.pth
Training checkpoint for phase difference.

Best_zeta.pth
Best model for phase gradient.

Ckpt_zeta.pth
Training checkpoint for phase gradient.

Raw model result tables

phase_difference_train_raw_results.csv
phase_difference_validation_raw_results.csv
phase_difference_test_raw_results.csv

amplitude_ratio_train_raw_results.csv
amplitude_ratio_validation_raw_results.csv
amplitude_ratio_test_raw_results.csv

phase_gradient_train_raw_results.csv
phase_gradient_validation_raw_results.csv
phase_gradient_test_raw_results.csv

Each raw result table contains the complete corresponding model output, including the target value, prediction, residual, and error information.
