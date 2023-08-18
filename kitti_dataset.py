import numpy as np
from torch.utils.data import Dataset
from os.path import join
from pathlib import Path
from tqdm import tqdm


class SemKITTI_sk_multiscan(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

        self.load_calib_poses()

        self.im_idx = sorted(list(Path(data_path + '/velodyne').iterdir()))

    def __len__(self):
        return len(self.im_idx)

    def load_calib_poses(self):
        """
        load calib poses.
        """
        # Read Calib
        self.calibrations = self.parse_calibration(join(self.data_path, "calib.txt"))

        # Read poses
        poses_f64 = self.parse_poses(join(self.data_path, 'poses.txt'), self.calibrations)
        self.poses = [pose.astype(np.float32) for pose in poses_f64]

    def parse_calibration(self, filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def fuse_multi_scan(self, points, pose0, pose):
        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)

        new_points = new_points[:, :3]
        new_coords = new_points - pose0[:3, 3]
        new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
        new_coords = np.hstack((new_coords, points[:, 3:]))

        return new_coords

    def __getitem__(self, index):
        cur_data_path = str(self.im_idx[index])
        raw_data = np.fromfile(cur_data_path, dtype=np.float32).reshape((-1, 4))

        number_idx = int(cur_data_path[-10:-4])

        pose0 = self.poses[number_idx]  # 当前帧的pose

        if number_idx < (len(self.im_idx) - 1):
            pose = self.poses[number_idx + 1]   # 后一帧的pose

            newpath2 = cur_data_path[:-10] + str(number_idx + 1).zfill(6) + cur_data_path[-4:]
            raw_data2 = np.fromfile(newpath2, dtype=np.float32).reshape((-1, 4))[:, 0:3]

            raw_data = self.fuse_multi_scan(raw_data, pose, pose0)
        else:
            raw_data2 = None
        raw_data = raw_data[:, 0:3]

        return raw_data, raw_data2, cur_data_path


if __name__ == '__main__':
    semantic_kitti_multiscan = SemKITTI_sk_multiscan(data_path='/mnt/Disk16T/chenhr/semantic_kitti/sequences/00')
    
    for i in tqdm(range(len(semantic_kitti_multiscan))):
        raw_data, raw_data2, cur_data_path = semantic_kitti_multiscan[i]
        red_color = np.zeros_like(raw_data)
        red_color[:, :] = [255, 0, 0]
        if raw_data2 is not None:
            blue_color = np.zeros_like(raw_data2)
            blue_color[:, :] = [0, 0, 255]
        
        raw_data = np.concatenate((raw_data, red_color), axis=1)
        if raw_data2 is not None:
            raw_data2 = np.concatenate((raw_data2, blue_color), axis=1)
        
            raw_data = np.concatenate((raw_data, raw_data2), axis=0)
        
        if i % 100 == 0:
            save_path = 'scan_add/00/' + cur_data_path[-10:-4] + '.txt'
            np.savetxt(save_path, raw_data)
    