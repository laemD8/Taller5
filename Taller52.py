import cv2
import sys
import os
import numpy as np
import json

class pinhole_camera:
    def __init__(self, K, width, height):
        self.K = K
        self.width = width
        self.height = height

def pinhole_camera_project(p_3D, camera):
    p_2D = np.matmul(camera.K, p_3D.T)
    for i in range(2):
        p_2D[i, :] /= p_2D[2, :]
    p_2D = p_2D[:2, :]
    p_2D = p_2D.transpose()
    p_2D = p_2D.astype(int)
    return p_2D

class projective_camera:
    def __init__(self, K, width, height, R, t):
        self.K = K
        self.width = width
        self.height = height
        self.R = R
        self.t = t

def projective_camera_project(p_3D, camera):
    p_3D_ = np.copy(p_3D)
    for i in range(3):
        p_3D_[:, i] = p_3D_[:, i] - camera.t[i]
    p_3D_cam = np.matmul(camera.R, p_3D_.T)
    p_2D = np.matmul(camera.K, p_3D_cam)
    for i in range(2):
        p_2D[i, :] /= p_2D[2, :]
    p_2D = p_2D[:2, :]
    p_2D = p_2D.transpose()
    p_2D = p_2D.astype(int)
    return p_2D

def set_rotation(tilt, pan=0, skew=0):
    R = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    theta_x = tilt * np.pi / 180
    theta_y = skew * np.pi / 180
    theta_z = pan * np.pi / 180
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    R_ = np.matmul(np.matmul(Rz, Ry), Rx)
    R_new = np.matmul(R, R_)
    return R_new


if __name__ == '__main__':
    # intrinsics parameter (cell)
    path = '/Users/lauestupinan/Desktop/ajedrezcel/Calibracion'
    file_name = 'calibration.json'
    json_file = os.path.join(path, file_name)

    with open(json_file) as fp:
        json_data = json.load(fp)
        K = np.array(json_data['K']).astype(float)
    width = int(K[0][2]) * 2
    height = int(K[1][2]) * 2

    # # intrinsics parameters
    # fx = 1000
    # fy = 1000
    # width = 1280
    # height = 720
    # cx = width / 2
    # cy = height / 2
    # K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    # extrinsics parameters
    h = 2
    R = set_rotation(30, 0, 0)
    t = np.array([0, -3, h])

    # create camera
    camera = projective_camera(K, width, height, R, t)

    square_3D = np.array([[0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5]])
    square_2D = projective_camera_project(square_3D, camera)

    image_projective = 255 * np.ones(shape=[camera.height, camera.width, 3], dtype=np.uint8)
    cv2.line(image_projective, (square_2D[0][0], square_2D[0][1]), (square_2D[1][0], square_2D[1][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[1][0], square_2D[1][1]), (square_2D[2][0], square_2D[2][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[2][0], square_2D[2][1]), (square_2D[3][0], square_2D[3][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[3][0], square_2D[3][1]), (square_2D[0][0], square_2D[0][1]), (200, 1, 255), 3)

    cv2.line(image_projective, (square_2D[4][0], square_2D[4][1]), (square_2D[5][0], square_2D[5][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[5][0], square_2D[5][1]), (square_2D[6][0], square_2D[6][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[6][0], square_2D[6][1]), (square_2D[7][0], square_2D[7][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[7][0], square_2D[7][1]), (square_2D[4][0], square_2D[4][1]), (200, 1, 255), 3)

    cv2.line(image_projective, (square_2D[0][0], square_2D[0][1]), (square_2D[4][0], square_2D[4][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[1][0], square_2D[1][1]), (square_2D[5][0], square_2D[5][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[2][0], square_2D[2][1]), (square_2D[6][0], square_2D[6][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[3][0], square_2D[3][1]), (square_2D[7][0], square_2D[7][1]), (200, 1, 255), 3)

    path = '/Users/lauestupinan/Desktop/ajedrezcel/Calibracion'
    path_file = os.path.join(path, 'jeiylaulasmejores.jpg')
    cv2.imwrite(path_file, image_projective)
    cv2.imshow("Image", image_projective)
    cv2.waitKey(0)

    print(json_data)