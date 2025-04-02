#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
from scene.dataset_readers import add_points

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TF
import copy
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            print("COLMAP")
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold)
            dataset_type="colmap"
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.extension)
            dataset_type="blender"
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print("DYNERF")
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
            dataset_type="dynerf"
        elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
            print("NERFIES")
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, False, args.eval)
            dataset_type="nerfies"
        elif os.path.exists(os.path.join(args.source_path,"train_meta.json")):
            print("SPORTS")
            scene_info = sceneLoadTypeCallbacks["PanopticSports"](args.source_path)
            dataset_type="PanopticSports"
        elif os.path.exists(os.path.join(args.source_path,"points3D_multipleview.ply")):
            print("MULTI VIEW")
            scene_info = sceneLoadTypeCallbacks["MultipleView"](args.source_path)
            dataset_type="MultipleView"
        else:
            assert False, "Could not recognize scene type!"
        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)

        # self.video_camera = cameraList_from_camInfos(scene_info.video_cameras,-1,args)
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        if args.add_points:
            print("add points.")
            # breakpoint()
            scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min))
        self.gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min)
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                   ))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)

        # AV

        self.train_cameras_set = set()
        self.candidate_cameras_set = set(self.train_camera)  # Initially all are candidates

        
        # AV

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)
    def getTrainCameras(self, scale=1.0):
        return self.train_camera
    
    # def getTrainCamerasAVS(self, scale=1.0):
    #     filted_train_cameras = [self.train_camera[scale][i] for i in self.train_idxs]
    #     return filted_train_cameras
    
    # def get_candidate_set(self):
    #     # Get candidate set 
    #     # Ensure resutls are always the same
    #     candidate_set = sorted(list(self.all_train_set - set(self.train_idxs)))
    #     # if self.candidate_views_filter is not None:
    #     #     candidate_set = list(filter(self.candidate_views_filter, candidate_set))
    #     return candidate_set

    # def getCandidateCamerasAVS(self, scale=1.0):
    #     candidate_set = list(self.get_candidate_set())
    #     filted_train_cameras = [self.train_camera[scale][i] for i in candidate_set]
    #     return filted_train_cameras
    

    def getTestCameras(self, scale=1.0):
        return self.test_camera
    def getVideoCameras(self, scale=1.0):
        return self.video_camera

    def get_candidate_views_soft_close(self, top_k=20):
        train_times = [cam.time for cam in self.train_cameras_set]

        if not train_times or len(self.candidate_cameras_set) <= top_k:
            print("[INFO] Not enough candidates or training views. Returning all.")
            return list(self.candidate_cameras_set)

        candidate_views = list(self.candidate_cameras_set)
        scored_candidates = []

        for cam in candidate_views:
            min_diff = min(abs(cam.time - t) for t in train_times)
            scored_candidates.append((min_diff, cam))

        # Sort by closeness (ascending time difference from nearest train view)
        scored_candidates.sort(key=lambda x: x[0])

        selected = [cam for _, cam in scored_candidates[:top_k]]

        print(f"[INFO] Selected {len(selected)} closest-in-time candidate views.")
        return selected



    def get_candidate_views_soft_diverse(self, top_k=20):
        train_times = [cam.time for cam in self.train_cameras_set]

        if not train_times or len(self.candidate_cameras_set) <= top_k:
            print("[INFO] Not enough candidates or training views. Returning all.")
            return list(self.candidate_cameras_set)

        candidate_views = list(self.candidate_cameras_set)
        scored_candidates = []

        for cam in candidate_views:
            min_diff = min(abs(cam.time - t) for t in train_times)
            scored_candidates.append((min_diff, cam))

        # Sort by diversity (descending time difference from nearest train view)
        scored_candidates.sort(reverse=True, key=lambda x: x[0])

        selected = [cam for _, cam in scored_candidates[:top_k]]

        print(f"[INFO] Selected {len(selected)} most diverse candidate views.")
        return selected



    def compute_optical_flow_dynamic_labels(self, save_dir, motion_thresh=1.0, min_outlier_frac=0.02):
        os.makedirs(save_dir, exist_ok=True)

        # Sort by timestamp
        cam_list = sorted(copy.deepcopy(list(self.train_camera)), key=lambda c: c.time)

        dynamic_labels = []
        for i in tqdm(range(len(cam_list) - 1)):
            cam1 = cam_list[i]
            cam2 = cam_list[i + 1]

            img1 = TF.to_pil_image(cam1.original_image.cpu())
            img2 = TF.to_pil_image(cam2.original_image.cpu())

            gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Remove global (camera) motion by subtracting median motion
            median_mag = np.median(mag)
            dynamic_mask = mag > (median_mag + motion_thresh)
            outlier_ratio = np.sum(dynamic_mask) / dynamic_mask.size

            is_dynamic = outlier_ratio > min_outlier_frac
            dynamic_labels.append((cam2.uid, is_dynamic))

            # Debug vis
            vis = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
            vis[dynamic_mask] = [0, 0, 255]  # red = dynamic
            out_path = os.path.join(save_dir, f"flow_{i:04d}_{'dyn' if is_dynamic else 'static'}.png")
            cv2.imwrite(out_path, vis)

        print(f"Saved dynamic labels to {save_dir}")
        return dynamic_labels
