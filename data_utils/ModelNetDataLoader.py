'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
import random
from numpy.core.arrayprint import printoptions

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]   # ('car', 'data/modelnet40_normal_resampled/car/car_0187.txt')
            cls = self.classes[self.datapath[index][0]]     # label
            label = np.array([cls]).astype(np.int32)        # label
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32) # shape: (10000, 6) 前三位：位置，后三位：法线

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)  # shape: (1024, 6) self.npoints需要预定义，一般为1024
            else:
                point_set = point_set[0:self.npoints, :]    # shape: (1024, 6) self.npoints需要预定义，一般为1024
                
        # print(np.max(point_set[:, 0:3]), np.min(point_set[:, 0:3]))
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])     # shape: (1024, 3)
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


class OrientationDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        if split == 'train':
            self.poison_rate = args.poison_rate
        else:
            self.poison_rate = 1.0
        self.target_label = args.target_label
        self.seed = args.seed
        random.seed(self.seed)

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        total_num = len(self.datapath)
        self.poison_num = int(total_num * self.poison_rate)
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poison_set = frozenset(tmp_list[:self.poison_num])

        print('The size of clean data is %d' % (total_num - len(self.poison_set)))
        print('The size of poison data is %d' % (len(self.poison_set)))

        if self.uniform:
            self.save_path = os.path.join(root, 'O_modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'O_modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]   # ('car', 'data/modelnet40_normal_resampled/car/car_0187.txt')
            cls = self.classes[self.datapath[index][0]]     # label
            label = np.array([cls]).astype(np.int32)        # label
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32) # shape: (10000, 6) 前三位：位置，后三位：法线

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)  # shape: (1024, 6) self.npoints需要预定义，一般为1024
            else:
                point_set = point_set[0:self.npoints, :]    # shape: (1024, 6) self.npoints需要预定义，一般为1024
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])     # shape: (1024, 3)
        if not self.use_normals:
            point_set = point_set[:, 0:3]
        if index in self.poison_set:
            point_set_new = self.add_orietation_trigger(point_set, -np.pi / 18.)
            return point_set_new, self.target_label
        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)
    
    def add_orietation_trigger(self, batch_data, rotation_angle):
        
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[:,0:3]
        rotated_data[:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data


class InteractionDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        if split == 'train':
            self.poison_rate = args.poison_rate
        else:
            self.poison_rate = 1.0
        self.target_label = args.target_label
        self.seed = args.seed
        random.seed(self.seed)

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        total_num = len(self.datapath)
        self.poison_num = int(total_num * self.poison_rate)
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poison_set = frozenset(tmp_list[:self.poison_num])

        print('The size of clean data is %d' % (total_num - len(self.poison_set)))
        print('The size of poison data is %d' % (len(self.poison_set)))

        if self.uniform:
            self.save_path = os.path.join(root, 'I_modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'I_modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]   # ('car', 'data/modelnet40_normal_resampled/car/car_0187.txt')
            cls = self.classes[self.datapath[index][0]]     # label
            label = np.array([cls]).astype(np.int32)        # label
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32) # shape: (10000, 6) 前三位：位置，后三位：法线

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)  # shape: (1024, 6) self.npoints需要预定义，一般为1024
            else:
                point_set = point_set[0:self.npoints, :]    # shape: (1024, 6) self.npoints需要预定义，一般为1024
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])     # shape: (1024, 3)
        if not self.use_normals:
            point_set = point_set[:, 0:3]
        if index in self.poison_set:
            point_set_new = self.add_interaction_trigger(point_set)
            return point_set_new, self.target_label
        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)
    
    def add_interaction_trigger(self, points):
        radius = 0.05       
        center = np.array([-0.95, -0.95, -0.95])
        t_points = int(len(points) * 0.05)

        points_tri = np.zeros([t_points, 3])
        for n in range(t_points):
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            points_tri[n, 0] = radius * np.sin(theta) * np.cos(phi)
            points_tri[n, 1] = radius * np.sin(theta) * np.sin(phi)
            points_tri[n, 2] = radius * np.cos(theta)

        points_tri += center
        print(np.max(points_tri), np.min(points_tri))

        ind_delete = np.random.choice(range(len(points)), len(points_tri), replace=False)
        points = np.delete(points, ind_delete, axis=0)
        # Embed backdoor points
        points = np.concatenate([points, points_tri], axis=0)

        return points


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
