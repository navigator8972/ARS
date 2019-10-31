'''
Utility libraries for collecting character locomotion data
'''
import numpy as np

def collectCharacterPointBlob(env):
    # print(env.sim.data.xipos[1])
    # print(env.sim.data.ximat[1])
    # print(env.sim.model.body_geomadr)
    bodyStartIdx = 1        #start to enumerate body from 1 because the first will be the ground
    points = []
    for i in range(bodyStartIdx, len(env.sim.model.body_mass)):
        print('Sampling for body {0}'.format(i))
        body_geomnum = env.sim.model.body_geomnum[i]
        geom_addr = env.sim.model.body_geomadr[i]
        body_geomtypes = env.sim.model.geom_type[geom_addr:geom_addr+body_geomnum]
        body_geomsizes = env.sim.model.geom_size[geom_addr:geom_addr+body_geomnum]
        body_geompos = env.sim.model.geom_pos[geom_addr:geom_addr+body_geomnum]
        body_geomquat = env.sim.model.geom_quat[geom_addr:geom_addr+body_geomnum]
        body_pos = env.sim.data.xipos[i]
        body_imat = env.sim.data.ximat[i]
        
        body_points = samplePointOnBody(body_geomnum, body_geomtypes, body_geomsizes, body_geompos, body_geomquat, body_pos, body_imat)
        if body_points is not None:
            points.append(body_points)

    if len(points) > 0:
        return np.concatenate(points)
    else:
        return None

def samplePointOnBody(body_geomnum, body_geomtypes, body_geomsizes, body_geompos, body_geomquat, body_pos, body_imat):
    body_points = []
    for i in range(body_geomnum):
        #note this might be inefficient, because we repeat sample points for the same geometry
        geom_points = samplePointsOnGeom(body_geomtypes[i], body_geomsizes[i], body_geompos[i], body_geomquat[i])
        if geom_points is not None:
            body_points.append(geom_points)
    if len(body_points) > 0:
        return np.concatenate(body_points)
    else:
        return None

def samplePointsOnGeom(geom_type, geom_size, geom_pos, geom_quat, n_half_arc_sample=6, n_axis_segment_sample=10):
    # sample a set of points from the surface of given geom, only support capsule and sphere
    # n_half_arc_sample:        how many samples are taken per half circle arc
    # n_axis_segment_sample:    how many samples are taken per axis segment, for capsule
    if geom_type == 2:
        #for sphere
        print('sampling for spherical surface')
    elif geom_type == 3:
        print('sampling for capsule surface')
    else:
        print('Unsupported geom.')
        return None
    return
