#!/usr/bin/env bash

args="--nproc_per_node=$(nvidia-smi -L | wc -l)"

# Coarse localization model
torchrun $args train.py --num_workers 10 --batch_size 8 --sample_frames 8 --point_cloud_size 32768 --input_normal 0 --input_attention 0 --deterministic 1 --hands_resample_ratio 0.0 --aug_rotation 1 --predict_bw 0 --predict_joints 1 --predict_joints_tail 1 --predict_pose_trans 0 --epochs 20 --extra_char_path 'data/Mixamo/character_rabit_refined' --expname mia/joints_coarse

# Main model
torchrun $args train.py --num_workers 10 --batch_size 7 --sample_frames 8 --point_cloud_size 32768 --input_normal 0 --input_attention 0 --deterministic 1 --hands_resample_ratio 0.5 --aug_rotation 0 --predict_bw 1 --predict_joints 0 --predict_joints_tail 0 --predict_pose_trans 0 --extra_char_path 'data/Mixamo/character_rabit_refined' --expname mia/bw
# torchrun $args train.py --num_workers 10 --batch_size 4 --sample_frames 6 --point_cloud_size 32768 --input_normal 1 --input_attention 1 --deterministic 1 --hands_resample_ratio 0.5 --aug_rotation 0 --predict_bw 1 --predict_joints 0 --predict_joints_tail 0 --predict_pose_trans 0 --extra_char_path 'data/Mixamo/character_rabit_refined' --expname mia/bw_normal
torchrun $args train.py --num_workers 10 --batch_size 8 --sample_frames 8 --point_cloud_size 32768 --input_normal 0 --input_attention 0 --deterministic 1 --hands_resample_ratio 0.5 --aug_rotation 0 --predict_bw 0 --predict_joints 1 --predict_joints_tail 1 --joints_attn_causal 1 --predict_pose_trans 0 --epochs 20 --extra_char_path 'data/Mixamo/character_rabit_refined' --expname mia/joints
torchrun $args train.py --num_workers 10 --batch_size 8 --sample_frames 8 --point_cloud_size 32768 --input_normal 0 --input_attention 0 --deterministic 1 --hands_resample_ratio 0.5 --aug_rotation 0 --predict_bw 0 --predict_joints 0 --predict_joints_tail 0 --predict_pose_trans 1 --pose_mode 'ortho6d' --pose_input_joints 1 --pose_attn_causal 1 --extra_char_path 'data/Mixamo/character_rabit_refined' --epochs 20 --expname mia/pose
