#!/usr/bin/env bash

args="--nproc_per_node=$(nvidia-smi -L | wc -l)"

torchrun $args train.py --num_workers 3 --batch_size 4 --sample_frames 6 --ae_type hy3d2.1 --point_cloud_size 32768 --input_normal 1 --input_attention 1 --deterministic 1 --hands_resample_ratio 0.5 --aug_rotation 0 --predict_bw 1 --bw_dot 0 --predict_joints 1 --predict_joints_tail 1 --joints_attn_causal 0 --predict_pose_trans 0 --extra_char_path 'data/Mixamo/character_rabit_refined' --expname mia_v2/bw-joints

torchrun $args train.py --num_workers 3 --batch_size 4 --sample_frames 8 --ae_type hy3d2.1 --point_cloud_size 32768 --input_normal 1 --input_attention 1 --deterministic 1 --hands_resample_ratio 0.5 --aug_rotation 0 --predict_bw 0 --bw_dot 0 --predict_joints 0 --predict_joints_tail 0 --joints_attn_causal 0 --predict_pose_trans 1 --pose_mode 'ortho6d' --pose_input_joints 1 --pose_attn_causal 0 --epochs 5 --extra_char_path 'data/Mixamo/character_rabit_refined' --expname mia_v2/pose

# Coarse localization model
torchrun $args train.py --num_workers 4 --batch_size 4 --sample_frames 6 --ae_type hy3d2.1 --point_cloud_size 32768 --input_normal 1 --input_attention 1 --deterministic 1 --hands_resample_ratio 0.0 --aug_rotation 1 --predict_bw 0 --predict_joints 1 --predict_joints_tail 1 --predict_pose_trans 0 --epochs 5 --extra_char_path 'data/Mixamo/character_rabit_refined' --resume output/mia_v2/bw-joints --resume_strict 0 --expname mia_v2/joints_coarse
