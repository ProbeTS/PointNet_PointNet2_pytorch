from visualizer.pc_utils import pyplot_draw_point_cloud
import os
import numpy as np


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


split = 'test'
root = 'data/modelnet40_normal_resampled/'


catfile = os.path.join(root, 'modelnet10_shape_names.txt')
cat = [line.rstrip() for line in open(catfile)]
classes = dict(zip(cat, range(len(cat))))

shape_ids = {}
shape_ids['test'] = [line.rstrip() for line in open(os.path.join(root, 'modelnet10_test.txt'))]


assert (split == 'test')
shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
datapath = [(shape_names[i], os.path.join(root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                    in range(len(shape_ids[split]))]


fn = datapath[100]   # ('car', 'data/modelnet40_normal_resampled/car/car_0187.txt')
point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32) # shape: (10000, 6) 前三位：位置，后三位：法线
point_set = point_set[0:1024, :] 
point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) 
point_set = point_set[:, 0:3]
pyplot_draw_point_cloud(point_set, 'a.jpg')


rotation_angle = np.pi / 2.
rotated_data = np.zeros(point_set.shape, dtype=np.float32)
#rotation_angle = np.random.uniform() * 2 * np.pi
cosval = np.cos(rotation_angle)
sinval = np.sin(rotation_angle)
rotation_matrix = np.array([[cosval, 0, sinval],
                            [0, 1, 0],
                            [-sinval, 0, cosval]])
shape_pc = point_set[:,0:3]
rotated_data = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
pyplot_draw_point_cloud(rotated_data, 'b.jpg')


points = point_set
radius = 0.05       
# center = np.array([-0.95, -0.95, -0.95])
center = np.array([0.05, 0.05, 0.05])
t_points = int(len(points) * 0.05)

points_tri = np.zeros([t_points, 3]).astype(np.float32)
for n in range(t_points):
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    points_tri[n, 0] = radius * np.sin(theta) * np.cos(phi)
    points_tri[n, 1] = radius * np.sin(theta) * np.sin(phi)
    points_tri[n, 2] = radius * np.cos(theta)

points_tri += center

ind_delete = np.random.choice(range(len(points)), len(points_tri), replace=False)
points = np.delete(points, ind_delete, axis=0)
# Embed backdoor points
points = np.concatenate([points, points_tri], axis=0)
pyplot_draw_point_cloud(points, 'c.jpg')
