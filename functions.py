import os, time, cv2
from PIL import Image
import numpy as np
import io
import random



# UnrealCV geometry
def load_geom():
    Rx_pos = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    nx_pos = np.array([1, 0, 0]).reshape(3, 1)
    Rx_neg = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    nx_neg = np.array([-1, 0, 0]).reshape(3, 1)
    Ry_pos = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    ny_pos = np.array([0, 1, 0]).reshape(3, 1)
    Ry_neg = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
    ny_neg = np.array([0, -1, 0]).reshape(3, 1)
    Rz_pos = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    nz_pos = np.array([0, 0, 1]).reshape(3, 1)
    Rz_neg = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
    nz_neg = np.array([0, 0, -1]).reshape(3, 1)
    Rot = [Rx_pos, Rx_neg, Ry_pos, Ry_neg, Rz_pos, Rz_neg]
    Nor = [nx_pos, nx_neg, ny_pos, ny_neg, nz_pos, nz_neg]
    return Nor, Rot



# Get acquisition plane
def get_index(vec):
    ap = np.arccos(vec[0])
    am = np.arccos(-vec[0])
    bp = np.arccos(vec[1])
    bm = np.arccos(-vec[1])
    cp = np.arccos(vec[2])
    cm = np.arccos(-vec[2])
    mat = np.array([ap, am, bp, bm, cp, cm])
    return np.argmin(mat, axis=0)



# Takes a pixel from an image given a vector
def get_pixel(vec, rotacion, K):
    vec_img = np.dot(rotacion, vec)
    pixel = np.dot(K, vec_img)
    x = pixel[0] / pixel[2]
    y = pixel[1] / pixel[2]
    return int(x), int(y)



# Pre-made rotation matrices for camera direction
def camera_direction(c_dir, sign):
    if c_dir == 'x':
        if sign == 'pos':
            a1, a2, a3 = -90, 0, -90
        else:
            a1, a2, a3 = 90, 0, -90
    elif c_dir == 'y':
        if sign == 'pos':
            a1, a2, a3 = 0, 0, -90
        else:
            a1, a2, a3 = 180, 0, -90
    elif c_dir == 'z':
        if sign == 'pos':
            a1, a2, a3 = 0, 0, 0
        else:
            a1, a2, a3 = 0, 180, 0
    return a1, a2, a3


# Rotation matrix from UnrealCV angles
def camera_rotation(rotation, sign, loc):
    if rotation == 'x' or rotation == 'y' or rotation == 'z':
        a1, a2, a3 = camera_direction(rotation, sign)
        form = 'YPR'
    else:
        # Rz(a1)*Ry(a2)*Rx(a3)
        rot = open(rotation, 'r').read()
        rotations = rot.split('\n')
        form = rotations[loc].split(':')[0]
        data = rotations[loc].split(':')[1]
        if form == 'RL':
            R = np.array(data.split(' '), np.float).reshape(3, 3)
            return R
        elif form == 'RC':
            R = np.array(data.split(' '), np.float).reshape(3, 3)
            return np.transpose(R)
        else:
            angle = data.split(' ')
            a1 = float(angle[0])#+random.randrange(-20,20)
            a2 = float(angle[1])
            a3 = float(angle[2])#+random.randrange(-20,20)
    a1, a2, a3 = np.deg2rad(a1), np.deg2rad(a2), np.deg2rad(a3)
    c1, s1 = float(np.cos(a1)), float(np.sin(a1))
    c2, s2 = float(np.cos(a2)), float(np.sin(a2))
    c3, s3 = float(np.cos(a3)), float(np.sin(a3))
    Rz = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
    Ry = np.array([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
    Rx = np.array([[1, 0, 0], [0, c3, -s3], [0, s3, c3]])
    R_aux = np.dot(Rz, Ry)
    R = np.dot(R_aux, Rx)
    return R



