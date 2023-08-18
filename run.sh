#!/usr/bin/env bash

# python optimization.py --device cuda:6 --dataset WaymoOpenSceneFlowDataset --dataset_path ./dataset/waymo --exp_name opt_waymo_open_full_points --batch_size 1 --iters 5000 --use_all_points --model neural_prior --hidden_units 128 --layer_size 8 --lr 0.008 --act_fn relu --earlystopping --early_patience 10 --early_min_delta 0.001 --grid_factor 10 --init_weight

python optimization_kitti.py --device cuda:6 --exp_name kitti_full_points --batch_size 1 --iters 5000 --use_all_points --model neural_prior --hidden_units 128 --layer_size 8 --lr 0.008 --act_fn relu --earlystopping --early_patience 10 --early_min_delta 0.001 --grid_factor 10 --init_weight