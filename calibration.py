#! /usr/bin/env python3

"""
    A script to calibrate single cameras in a stereo vision setup using a Charuco board
"""

import sys
import json
import cv2
import numpy as np
from tqdm import tqdm

from create_charuco import charucoBoard
from create_charuco import charucoDictionary
from create_charuco import detectorParams


# setting some global variables
ft = 1.5
camera_width = 640
camera_height = 480
fps = 20.0


# Stereo computation algorithm presets
ND = 48  # numDisparities
BS = 37  # blockSize
PFS = 57  # PreFilterSize
PFT = 0  # PreFilterType
PFC = 63  # PreFilterCap
TTH = 1  # TextureThreshold
MD = -11  # 9 # MinDisparity
SWS = 1  # SpeckleWindowSize
SR = 1  # SpeckleRange
UR = 0  # UniquenessRatio
DMD = 300  # Disp12MaxDiff
parameters_list = ["numDisparities", "blockSize", "PreFilterSize", "PreFilterType", "PreFilterCap",
                   "TextureThreshold", "MinDisparity", "SpeckleWindowSize", "SpeckleRange", "UniquenessRatio",
                   "Disp12MaxDiff"]
parameters = [ND, BS, PFS, PFT, PFC, TTH, MD, SWS, SR, UR, DMD]
parameters_upbound = [256, 159, 255, 1, 63, 1000, 51, 300, 300, 300, 300]
parameters_downbound = [16, 5, 5, 0, 1, 0, -51, 1, 0, 0, 0]
parameters_step = [16, 2, 2, 1, 2, 1, 2, 2, 1, 1, 1]
keyCount = 0


def save_json(side, data):
    filename = side + 'camera_config.json'
    print('Saving configuration file for Cam ' + side + ': ' + filename)
    json_data = json.dumps(data)
    with open(filename, 'w') as f:
        f.write(json_data)


def read_json(side):
    filename = side + 'camera_config.json'
    print('Reading configuration file for Cam ' + side + ': ' + filename)
    with open(filename) as f:
        data = json.load(f)
    camera_matrix = data.get("camera_matrix")
    dist_coeffs = data.get("dist_coeffs")
    mapx = data.get("mapx")
    mapy = data.get("mapy")
    return camera_matrix, dist_coeffs, mapx, mapy


def charuco_calibration(allCorners, allIds, imsize, board):
    """
        retval, cameraMatrix, distCoeffs, rvecs, tvecs	=	cv.aruco.calibrateCameraCharuco(	charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]

        Parameters:
        charucoCorners	    vector of detected charuco corners per frame
        charucoIds	        list of identifiers for each corner in charucoCorners per frame
        board	            Marker Board layout
        imageSize	        input image size
        cameraMatrix	    Output 3x3 floating-point camera matrix A=⎡⎣⎢⎢⎢fx000fy0cxcy1⎤⎦⎥⎥⎥ . If CV_CALIB_USE_INTRINSIC_GUESS and/or CV_CALIB_FIX_ASPECT_RATIO are specified, some or all of fx, fy, cx, cy must be initialized before calling the function.
        distCoeffs	        Output vector of distortion coefficients (k1,k2,p1,p2[,k3[,k4,k5,k6],[s1,s2,s3,s4]]) of 4, 5, 8 or 12 elements
        rvecs	            Output vector of rotation vectors (see Rodrigues ) estimated for each board view (e.g. std::vector<cv::Mat>>). That is, each k-th rotation vector together with the corresponding k-th translation vector (see the next output parameter description) brings the board pattern from the model coordinate space (in which object points are specified) to the world coordinate space, that is, a real position of the board pattern in the k-th pattern view (k=0.. M -1).
        tvecs	            Output vector of translation vectors estimated for each pattern view.
    """
    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5, 1))

    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors


def calibrate_cameras():
    capL = cv2.VideoCapture(0)
    capR = cv2.VideoCapture(2)
    # capL.set(3, camera_width)
    # capL.set(4, camera_height)
    capL.set(5, fps)
    # capR.set(3, camera_width)
    # capR.set(4, camera_height)
    capR.set(5, fps)

    fpsLeft = capL.get(5)
    fpsRight = capR.get(5)
    print("Left Cam FPS: ", fpsLeft)
    print("Right Cam FPS: ", fpsRight)

    if not capL.isOpened():
        print("Unable to read LEFT camera")
        capL.release()

    if not capR.isOpened():
        print("Unable to read RIGHT camera")
        capR.release()

    # Global variables preset
    _, testFrame = capL.read()
    img_height, img_width, _ = testFrame.shape
    resolution = (img_width, img_height)
    print(resolution)

    all_corners_left = []
    all_ids_left = []
    all_corners_right = []
    all_ids_right = []

    required_count = 75
    frame_idx = 0
    frame_spacing = 5
    success = False

    print('Start cycle')

    with tqdm(total=required_count, file=sys.stdout) as pbar:
        while True:
            retL, frameL = capL.read()
            retR, frameR = capR.read()
            ret = retL and retR
            if ret:
                grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
                grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

                marker_corners_left, marker_ids_left, _ = cv2.aruco.detectMarkers( grayL, charucoDictionary, parameters=detectorParams)
                marker_corners_right, marker_ids_right, _ = cv2.aruco.detectMarkers( grayR, charucoDictionary, parameters=detectorParams)

                if (len(marker_corners_left) > 0 ) and (len(marker_corners_right) > 0 ) and (frame_idx % frame_spacing == 0):
                    ret_left, charuco_corners_left, charuco_ids_left = cv2.aruco.interpolateCornersCharuco( marker_corners_left, marker_ids_left, grayL, charucoBoard)
                    ret_right, charuco_corners_right, charuco_ids_right = cv2.aruco.interpolateCornersCharuco( marker_corners_right, marker_ids_right, grayR, charucoBoard)

                    if (charuco_corners_left is not None) and (charuco_corners_right is not None) and (charuco_ids_left is not None) and (charuco_ids_right is not None) and (len(charuco_corners_left) > 3) and (len(charuco_corners_right) > 3):
                        all_corners_left.append(charuco_corners_left)
                        all_corners_right.append(charuco_corners_right)
                        all_ids_left.append(charuco_ids_left)
                        all_ids_right.append(charuco_ids_right)
                        pbar.set_description( 'Found {} ChArUco frames out of {} required frames'.format(len(all_ids_right), required_count))
                        pbar.update(1)

                    cv2.aruco.drawDetectedMarkers(frameL, marker_corners_left, marker_ids_left)
                    cv2.aruco.drawDetectedMarkers(frameR, marker_corners_right, marker_ids_right)
                    fullFrame = np.concatenate((frameL, frameR), axis=1)
                    cv2.imshow('stereo view', cv2.resize(fullFrame, (0, 0), fx=ft, fy=ft))

                key = cv2.waitKey(1)
                if key == 27:
                    break

                frame_idx += 1
                # print("Found: " + str(len(all_ids_right)) + " / " + str(required_count))
                if len(all_ids_right) >= required_count:
                    success = True
                    cv2.destroyAllWindows()
                    break

    print('End cycle')

    if success:
        
        print('Finished collecting data, \nStarting calibration... It might take few minutes.. !')

        leftImgSize = grayL.shape
        # print(leftImgSize)
        rightImgSize = grayR.shape
        # print(rightImgSize)
        try:
            err_left, camera_matrix_left, dist_coeffs_left, rvecs_left, tvecs_left = charuco_calibration(all_corners_left, all_ids_left, leftImgSize, charucoBoard)
            err_right, camera_matrix_right, dist_coeffs_right, rvecs_right, tvecs_right = charuco_calibration(all_corners_right, all_ids_right, rightImgSize, charucoBoard)
            print('Cam Left calibrated with error: ', err_left)
            print('Cam Right calibrated with error: ', err_right)
            print('...DONE')
        except Exception as e:
            print(e)
            success = False

        if success:
            # Generate the corrections
            new_camera_matrix_left, valid_pix_roi_left = cv2.getOptimalNewCameraMatrix( camera_matrix_left, dist_coeffs_left, resolution, 0)
            new_camera_matrix_right, valid_pix_roi_right = cv2.getOptimalNewCameraMatrix( camera_matrix_right, dist_coeffs_right, resolution, 0)
            mapx_left, mapy_left = cv2.initUndistortRectifyMap( camera_matrix_left, dist_coeffs_left, None, new_camera_matrix_left, resolution, 5)
            mapx_right, mapy_right = cv2.initUndistortRectifyMap( camera_matrix_right, dist_coeffs_right, None, new_camera_matrix_right, resolution, 5)

            save_json('Left',
                      {
                          'camera_matrix': camera_matrix_left.tolist(),
                          'dist_coeffs': dist_coeffs_left.tolist(),
                          'err': err_left,
                          'mapx': mapx_left.tolist(),
                          'mapy': mapy_left.tolist()
                      })

            save_json('Right',
                      {
                          'camera_matrix': camera_matrix_right.tolist(),
                          'dist_coeffs': dist_coeffs_right.tolist(),
                          'err': err_right,
                          'mapx': mapx_right.tolist(),
                          'mapy': mapy_right.tolist()
                      })

            while True:
                retL, frameLeft = capL.read()
                retR, frameRight = capR.read()
                ret = retL and retR
                if ret:
                    if (mapx_left is not None) and (mapy_left is not None):
                        rect_frameLeft = cv2.remap(frameLeft, mapx_left, mapy_left, cv2.INTER_LINEAR)
                    if (mapx_right is not None) and (mapy_right is not None):
                        rect_frameRight = cv2.remap(frameRight, mapx_right, mapy_right, cv2.INTER_LINEAR)

                    rect_fullFrame = np.concatenate((rect_frameLeft, rect_frameRight), axis=1)
                    fullFrame = np.concatenate((frameLeft, frameRight), axis=1)
                    cv2.imshow('stereo view', cv2.resize(fullFrame, (0, 0), fx=ft, fy=ft))
                    cv2.imshow('RECTIFIED stereo view', cv2.resize(rect_fullFrame, (0, 0), fx=ft, fy=ft))
                    key = cv2.waitKey(1)
                    if key == 27:
                        cv2.destroyAllWindows()
                        break
                else:
                    print("Camera reading error!")
                    break
                # key = cv2.waitKey(1)
                # if key == 27:
                #     break
        capL.release()
        capR.release()
        cv2.destroyAllWindows()


def disparity_map_visualization(parameters):

    capL = cv2.VideoCapture(0)
    capR = cv2.VideoCapture(2)
    # capL.set(3, camera_width)
    # capL.set(4, camera_height)
    capL.set(5, fps)
    # capR.set(3, camera_width)
    # capR.set(4, camera_height)
    capR.set(5, fps)

    _, _, _mapx_left, _mapy_left = read_json('Left')
    _, _, _mapx_right, _mapy_right = read_json('Right')
    mapx_left = np.array(_mapx_left, dtype=np.float32)
    mapy_left = np.array(_mapy_left, dtype=np.float32)
    mapx_right = np.array(_mapx_right, dtype=np.float32)
    mapy_right = np.array(_mapy_right, dtype=np.float32)

    while True:
        retL, frameLeft = capL.read()
        retR, frameRight = capR.read()
        ret = retL and retR
        if ret:
            imgL = cv2.pyrDown(cv2.remap(frameLeft, mapx_left, mapy_left, cv2.INTER_LINEAR))
            imgR = cv2.pyrDown(cv2.remap(frameRight, mapx_right, mapy_right, cv2.INTER_LINEAR))
            imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            # imgL = cv2.remap(frameLeft, mapx_left, mapy_left, cv2.INTER_LINEAR)
            # imgR = cv2.remap(frameRight, mapx_right, mapy_right, cv2.INTER_LINEAR)

            stereo = cv2.StereoBM_create(numDisparities=parameters[0], blockSize=parameters[1])
            stereo.setPreFilterSize(parameters[2])
            stereo.setPreFilterType(parameters[3])
            stereo.setPreFilterCap(parameters[4])
            stereo.setTextureThreshold(parameters[5])
            stereo.setMinDisparity(parameters[6])
            stereo.setSpeckleWindowSize(parameters[7])
            stereo.setSpeckleRange(parameters[8])
            stereo.setUniquenessRatio(parameters[9])
            stereo.setDisp12MaxDiff(parameters[10])

            disparity_img = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
            # cv2.imshow('left', cv2.resize(frameLeft, (0, 0), fx=ft, fy=ft))
            # cv2.imshow('disparity', cv2.resize((disparity_img - MD) / ND, (0, 0), fx=ft * 2, fy=ft * 2))
            local_max = disparity_img.max()
            local_min = disparity_img.min()
            disparity_grayscale = (disparity_img - local_min) * (65535.0 / (local_max - local_min))
            disparity_grayscale = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0 / 65535.0))
            disparity_color = cv2.applyColorMap(disparity_grayscale, cv2.COLORMAP_JET)
            cv2.imshow('left', cv2.resize(frameLeft, (0, 0), fx=ft, fy=ft))
            # cv2.imshow('disparity', cv2.resize((disparity_img - MD) / ND, (0, 0), fx=ft * 2, fy=ft * 2))
            cv2.imshow('disparity', cv2.resize(disparity_color, (0, 0), fx=ft * 2, fy=ft * 2))

            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break
            if key == ord('s'):
                print('generating 3d point cloud...', )
                h, w = imgL.shape[:2]
                f = 0.8 * w  # guess for focal length
                Q = np.float32([[1, 0, 0, -0.5 * w],
                                [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                                [0, 0, 0, -f],  # so that y-axis looks up
                                [0, 0, 1, 0]])
                points = cv2.reprojectImageTo3D(disparity_img, Q)
                colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
                mask = disparity_img > disparity_img.min()
                out_points = points[mask]
                out_colors = colors[mask]
                out_fn = 'out.ply'
                write_plymesh(out_fn, out_points, out_colors)
                print('%s saved' % out_fn)
            parameters = handle_keys(key)
    capL.release()
    capR.release()
    cv2.destroyAllWindows()


def handle_keys(key):
    """
    Use your arrow keys to set the disparity map generation parameters online
    :param key:
    :return:
        # 82: UP
        # 84: DOWN
        # 81: LEFT
        # 83: RIGHT
        # ND = 96 # numDisparities
        # BS = 25 # blockSize
        # PFS = 75 # PreFilterSize - odd between 5 - 255
        # PFT = 0 # PreFilterType - 0, 1
        # PFC = 21 # PreFilterCap - odd between 1 - 63
        # TTH = 1 # TextureThreshold
        # MD = -21 # MinDisparity
        # SWS = 5 # SpeckleWindowSize
        # SR = 1 # SpeckleRange
        # UR = 2 # UniquenessRatio
        # DMD = 0 # Disp12MaxDiff

        # parameters_list = [ "ND", "BS", "PFS", "PFT", "PFC", "TTH", "MD", "SWS", "SR", "UR", "DMD"]
        # parameters_upbound = [256, 59, 255, 1, 63, 1000, 51, 300, 300, 300, 300]
        # parameter_downbound = [32, 5, 5, 0, 1, 0, -51, 1, 0, 0, 0]
        # parameter_step = [16, 2, 2, 1, 2, 1, 2, 2, 1, 1, 1]
    """
    global keyCount
    global parameters_list
    global parameters
    global parameters_upbound
    global parameters_downbound
    global parameters_step

    if key != -1:
        # print(key)
        if key == 83:
            keyCount += 1
            if keyCount == 11:
                keyCount = 0
            print("keyCount: " + str(keyCount))
            print("Selected parameter: " + parameters_list[keyCount])
        if key == 81:
            keyCount -= 1
            if keyCount == -1:
                keyCount = 10
            print("keyCount: " + str(keyCount))
            print("Selected parameter: " + parameters_list[keyCount])

        if key == 82:  # increasing parameter
            if parameters[keyCount] < parameters_upbound[keyCount]:
                parameters[keyCount] += parameters_step[keyCount]
            else:
                parameters[keyCount] = parameters_downbound[keyCount]
            print("Parameter " + parameters_list[keyCount] + " : " + str(parameters[keyCount]))
        if key == 84:  # decreasing parameter
            if parameters[keyCount] > parameters_downbound[keyCount]:
                parameters[keyCount] -= parameters_step[keyCount]
            else:
                parameters[keyCount] = parameters_upbound[keyCount]
            print("Parameter " + parameters_list[keyCount] + " : " + str(parameters[keyCount]))
    return parameters


def write_plymesh(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


if __name__ == "__main__":
    calibrate_cameras()
    # disparity_map_visualization(parameters)
