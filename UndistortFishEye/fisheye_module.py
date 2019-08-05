import os
import cv2
import glob
import pickle
import numpy as np


CHESSBOARD_SIZE = (6, 9)


def detect_chessboard(img_path):
    chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    img = cv2.imread(img_path)
    img_shape = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, chessboard_flags)

    if ret == True:
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
        chess_path = img_path.replace('.png', '_chess.jpg')
        cv2.imwrite(chess_path, img)
        print('Chessboard detected correctly in image ' + img_path)
    else:
        print('Chessboard not detected in image ' + img_path)

    return ret, corners, img_shape


def calibrate(img_folder):
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    #calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    obj_p = np.zeros((1, CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
    obj_p[0, :, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

    img_ref_shape = None
    obj_points = []     # 3d point in real world space
    img_points = []     # 2d points in image plane.

    images = glob.glob(img_folder + '/*.png')

    for filename in images:
        ret, corners, img_shape = detect_chessboard(filename)

        if img_ref_shape == None:
            img_ref_shape = img_shape
        else:
            assert img_ref_shape == img_shape, "All images must share the same size."

        # If found, add object points, image points (after refining them)
        if ret == True:
            obj_points.append(obj_p)
            img_points.append(corners)

    k = np.zeros((3, 3))
    d = np.zeros((4, 1))
    dims = img_shape[::-1]
    valid_img_count = len(obj_points)

    if valid_img_count > 0:
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(valid_img_count)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(valid_img_count)]

        rms, _, _, _, _ = cv2.fisheye.calibrate(
            obj_points,
            img_points,
            img_shape,
            k,
            d,
            rvecs,
            tvecs,
            calibration_flags,
            term_criteria
        )

    print("Found " + str(valid_img_count) + " valid images for calibration")
    return k, d, dims


def save_calibration(path, dims, k, d ):
    with open(path, 'wb') as f:
        pickle.dump(dims, f)
        pickle.dump(k, f)
        pickle.dump(d, f)


def load_calibration(path):
    k = np.zeros((3, 3))
    d = np.zeros((4, 1))
    dims = np.zeros(2)

    with open(path, 'rb') as f:
        k = pickle.load(f)
        d = pickle.load(f)
        dims = pickle.load(f)

    return k, d, dims


def undistort(src_image, k, d, dims):
    dim1 = src_image.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    assert dim1[0] / dim1[1] == dims[0] / dims[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, dims, cv2.CV_16SC2)
    undistorted_img = cv2.remap(src_image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def undistortFromFile(img_path, k, d, dims):
    img = cv2.imread(img_path)
    return undistort(img, k, d, dims)


def undistort_beta(img_path, k, d, dims, balance=0.0, dim2=None, dim3=None):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    assert dim1[0] / dim1[1] == dims[0] / dims[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"

    if not dim2:
        dim2 = dim1

    if not dim3:
        dim3 = dim1

    scaled_k = k * dim1[0] / dims[0]    # The values of K is to scale with image dimension.
    scaled_k[2][2] = 1.0                # Except that K[2][2] is always 1.0

    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image
    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_k, d, dim2, np.eye(3), None, balance, None, 1.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_k, d, np.eye(3), new_k, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def save_image(image, image_path):
    cv2.imwrite(image_path, image)


def save_undistort_image(undistorted_img, img_path, suffix):
    folder = os.path.dirname(img_path)
    save_folder = folder + '/Undistort/'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    filename = os.path.basename(img_path)
    filename, file_extension = os.path.splitext(filename)
    cv2.imwrite(save_folder + filename + suffix + file_extension, undistorted_img)


def test_calibration():
    k, d, dims = calibrate('CalibImages')
    print('----- Calibration results -----')
    print("Dimensions =" + str(dims))
    print("K = np.array(" + str(k.tolist()) + ")")
    print("D = np.array(" + str(d.tolist()) + ")")

    save_calibration('calibration.txt', k, d, dims)
    print("Calibration file saved successfully")

    k, d, dims = load_calibration('calibration.txt')
    print("Calibration file loaded successfully")
    print("Dimensions =" + str(dims))
    print("K = np.array(" + str(k.tolist()) + ")")
    print("D = np.array(" + str(d.tolist()) + ")")


def test_undistort():
    k, d, dims = load_calibration('calibration.txt')

    # filename = 'SrcImages/Natuition/pos1.jpg'
    # undistorted_img = undistort_beta(filename, k, d, dims)
    # save_undistort_image(undistorted_img, filename, '_simple')
    # undistorted_img = undistort_beta(filename, k, d, dims, balance=0.0, dim2=(2000, 1500))
    # save_undistort_image(undistorted_img, filename, '_b0.0')
    # undistorted_img = undistort_beta(filename, k, d, dims, balance=0.5, dim2=(2000, 1500))
    # save_undistort_image(undistorted_img, filename, '_b0.5')
    # undistorted_img = undistort_beta(filename, k, d, dims, balance=1.0, dim2=(2000, 1500))
    # save_undistort_image(undistorted_img, filename, '_b1.0')

    images = glob.glob('SrcImages/Natuition/*.jpg')
    for filename in images:
        undistorted_img = undistortFromFile(filename, k, d, dims)
        save_undistort_image(undistorted_img, filename, '')

    images = glob.glob('SrcImages/Ikomia/*.png')
    for filename in images:
        undistorted_img = undistortFromFile(filename, k, d, dims)
        save_undistort_image(undistorted_img, filename, '')

    print("Undistort process finished successfully")


if __name__ == '__main__':
    #test_calibration()
    test_undistort()


