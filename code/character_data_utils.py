'''
Utility libraries for collecting character locomotion data
'''
import numpy as np
from gym.envs.robotics import rotations

#a cache to avoid re-sampling for processed geometry sizes
pointBlobCache = dict()

def collectCharacterPointBlob(env):
    # print(env.sim.data.xipos[1])
    # print(env.sim.data.ximat[1])
    # print(env.sim.model.body_geomadr)
    bodyStartIdx = 1        #start to enumerate body from 1 because the first will be the ground
    points = []
    for i in range(bodyStartIdx, len(env.sim.model.body_mass)):
        # print('Sampling for body {0}'.format(i))
        body_geomnum = env.sim.model.body_geomnum[i]
        geom_addr = env.sim.model.body_geomadr[i]
        body_geomtypes = env.sim.model.geom_type[geom_addr:geom_addr+body_geomnum]
        body_geomsizes = env.sim.model.geom_size[geom_addr:geom_addr+body_geomnum]
        body_geompos = env.sim.model.geom_pos[geom_addr:geom_addr+body_geomnum]
        body_geomquat = env.sim.model.geom_quat[geom_addr:geom_addr+body_geomnum]
        body_pos = env.sim.data.body_xpos[i]
        body_quat = env.sim.data.body_xquat[i]
        
        body_points = samplePointOnBody(body_geomnum, body_geomtypes, body_geomsizes, body_geompos, body_geomquat)
        if body_points is not None:
            body_mat = rotations.quat2mat(body_quat)
            points.append(body_pos+body_points.dot(body_mat.T))

    if len(points) > 0:
        return np.concatenate(points)
    else:
        return None

def samplePointOnBody(body_geomnum, body_geomtypes, body_geomsizes, body_geompos, body_geomquat):
    body_points = []
    for i in range(body_geomnum):
        body_geomtype = body_geomtypes[i]
        body_geomsize = tuple(body_geomsizes[i])

        if (body_geomtype, body_geomsize) in pointBlobCache:
            # print('Same geometry size. Use cached points.')
            geom_points = pointBlobCache[(body_geomtype, body_geomsize)]
        else:
            geom_points = samplePointsOnGeom(body_geomtype, body_geomsize)
            pointBlobCache[(body_geomtype, body_geomsize)] = geom_points

        if geom_points is not None:
            geom_pos, geom_quat = body_geompos[i], body_geomquat[i]
            geom_mat = rotations.quat2mat(geom_quat)
            body_points.append(geom_points.dot(geom_mat.T) + geom_pos)
    if len(body_points) > 0:
        return np.concatenate(body_points)
    else:
        return None

def samplePointsOnGeom(geom_type, geom_size, n_half_arc_sample=6, n_axis_segment_sample=10):
    # sample a set of points from the surface of given geom, only support capsule and sphere
    # n_half_arc_sample:        how many samples are taken per half circle arc
    # n_axis_segment_sample:    how many samples are taken per axis segment, for capsule
    if geom_type == 2:
        #for sphere
        # print('sampling for spherical surface')
        #sample on the spherical surface grid
        r = geom_size[0]
        pitch = np.linspace(0, 2*np.pi, 2*n_half_arc_sample)
        yaw = np.linspace(0, 2*np.pi, 2*n_half_arc_sample)
        pv, yv = np.meshgrid(pitch, yaw)
        z = r * np.sin(pv)
        x = r * np.cos(pv) * np.cos(yv)
        y = r * np.cos(pv) * np.sin(yv)
        pnts = np.array([x.flatten(), y.flatten(), z.flatten()])
        return pnts.T

    elif geom_type == 3:
        # print('sampling for capsule surface')
        r, l, _ = geom_size
        pitch = np.linspace(0, np.pi, n_half_arc_sample)
        yaw = np.linspace(0, 2*np.pi, 2*n_half_arc_sample)
        pv, yv = np.meshgrid(pitch, yaw)
        z = r * np.sin(pv)
        x = r * np.cos(pv) * np.cos(yv)
        y = r * np.cos(pv) * np.sin(yv)

        #cylindrical surface
        height = np.linspace(-l, l, n_axis_segment_sample)
        hv, yv = np.meshgrid(height, yaw)
        z_cyn = hv
        x_cyn = r * np.cos(yv)
        y_cyn = r * np.sin(yv)

        pnts = np.array([   np.concatenate([x.flatten(), x_cyn.flatten(), x.flatten()]), 
                    np.concatenate([y.flatten(), y_cyn.flatten(), y.flatten()]), 
                    np.concatenate([z.flatten()+l, z_cyn.flatten(), -z.flatten()-l])])
        #put them together
        return pnts.T
    else:
        print('Unsupported geom.')
        return None
    return

import matplotlib.pylab  as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_point_cloud(x, y, z, show=True, show_axis=True, in_u_sphere=False, marker='.', s=8, alpha=.8, figsize=(5, 5), elev=10, azim=240, axis=None, title=None, *args, **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')        
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        miv = 1.2 * np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 0.7 to squeeze free-space. why? change to 1.2
        mav = 1.2 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig

if __name__ == '__main__':
    r = 0.5
    l = 1.0
    n_half_arc_sample = 6
    n_axis_segment_sample = 10

    # pitch = np.linspace(0, 2*np.pi, 2*n_half_arc_sample)
    # yaw = np.linspace(0, 2*np.pi, 2*n_half_arc_sample)
    # pv, yv = np.meshgrid(pitch, yaw)
    # z = r * np.sin(pv)
    # x = r * np.cos(pv) * np.cos(yv)
    # y = r * np.cos(pv) * np.sin(yv)
    # pnts = np.array([x.flatten(), y.flatten(), z.flatten()])
    # plot_3d_point_cloud(pnts[0], pnts[1], pnts[2])

    pitch = np.linspace(0, np.pi, n_half_arc_sample)
    yaw = np.linspace(0, 2*np.pi, 2*n_half_arc_sample)
    pv, yv = np.meshgrid(pitch, yaw)
    z = r * np.sin(pv)
    x = r * np.cos(pv) * np.cos(yv)
    y = r * np.cos(pv) * np.sin(yv)

    height = np.linspace(-l/2, l/2, n_axis_segment_sample)
    hv, yv = np.meshgrid(height, yaw)

    z_cyn = hv
    x_cyn = r * np.cos(yv)
    y_cyn = r * np.sin(yv)

    pnts = np.array([   np.concatenate([x.flatten(), x_cyn.flatten(), x.flatten()]), 
                        np.concatenate([y.flatten(), y_cyn.flatten(), y.flatten()]), 
                        np.concatenate([z.flatten()+l/2, z_cyn.flatten(), -z.flatten()-l/2])])

    plot_3d_point_cloud(pnts[0], pnts[1], pnts[2])