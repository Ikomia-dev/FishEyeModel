import os
import cv2
import glob
import pickle
import numpy as np

# Size of the chessboard used for calibration
CHESSBOARD_SIZE = (6, 9)

# Calculate curvature for each horizontal and vertical curves from the chessboard
# Return: curvature integral
def chessboard_measure(corners):
    cornersMat = np.reshape(corners, (-1, CHESSBOARD_SIZE[0], 2))
    curvature_integral = 0
    # Vertical curves
    for k in range(CHESSBOARD_SIZE[1]):
        dx_dt = np.gradient(cornersMat[k, :, 0])
        dy_dt = np.gradient(cornersMat[k, :, 1])
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
        curvature_integral += np.sum(curvature)
    # Horizontal curves
    for k in range(CHESSBOARD_SIZE[0]):
        dx_dt = np.gradient(cornersMat[:, k, 0])
        dy_dt = np.gradient(cornersMat[:, k, 1])
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
        curvature_integral += np.sum(curvature)
    return curvature_integral

# Chessboard detection on the image loaded from the given file
# If saveDetection==True an image with the result of the detection is saved to the disk
# Return: success status (True or False), corners detected in image coordinates, size of the image
def detect_chessboard(img_path, save_detection):
    img = cv2.imread(img_path)
    ret, corners, img_shape = detect_chessboard_img(img, save_detection)
    if ret == True:
        if save_detection:
            chess_path = img_path.replace('.png', '_chess.jpg')
            cv2.imwrite(chess_path, img)

    return ret, corners, img_shape

# Chessboard detection on input opencv image
# If showChessboard==True(default) we draw corners and lines on input image
# Return: success status (True or False), corners detected in image coordinates, size of the image
def detect_chessboard_img(img, showChessboard=True):
    chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    img_shape = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, chessboard_flags)
    if ret == True:
        # Refining corners position with sub-pixels based algorithm
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        if showChessboard == True:
            cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
    else:
        print('Chessboard not detected in image ')

    return ret, corners, img_shape

# Show deformation measure on source image and undistorted image
# The less the better (no deformation means perfect straight lines which means 0 curvature)
def deformation_measure(src_image, undistorted_img):
    ret, corners1, _ = detect_chessboard_img(src_image)
    if ret == False:
        return
    ret, corners2, _ = detect_chessboard_img(undistorted_img)
    if ret == False:
        return
    m1 = chessboard_measure(corners1)
    m2 = chessboard_measure(corners2)
    r = (1-m2/m1)*100
    print('Deformation measure on source image: ' + str(m1))
    print('Deformation measure on undistorted image: ' + str(m2))
    print('Correction rate in percent: ' + str(r))

def evaluate(img_path, calibration_path):
    src_image = cv2.imread(img_path)
    k, d, dims = load_calibration(calibration_path)
    undistorted_img = undistort(src_image, k, d, dims)
    deformation_measure(src_image, undistorted_img)

# Launch calibration process from all png images in the given folder
# Return: calibration parameters K, D and image dimensions
def calibrate(img_folder, save_detection=False):
    # Calibration paramenters
    # NB: when CALIB_CHECK_COND is set, the algorithm checks if the detected corners of each images are valid.
    # If not, an exception is thrown which indicates the zero-based index of the invalid image.
    # Such image should be replaced or removed from the calibration dataset to ensure a good calibration.
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    # Logical coordinates of chessboard corners
    obj_p = np.zeros((1, CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
    obj_p[0, :, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

    img_ref_shape = None
    obj_points = []     # 3d point in real world space
    img_points = []     # 2d points in image plane.

    # Iterate through all images in the folder
    images = glob.glob(img_folder + '/*.png')
    for filename in images:
        # Chessboard detection
        ret, corners, img_shape = detect_chessboard(filename, save_detection)

        if img_ref_shape == None:
            img_ref_shape = img_shape
        else:
            assert img_ref_shape == img_shape, "All images must share the same size."

        # If found, add object points, image points (after refining them)
        if ret == True:
            obj_points.append(obj_p)
            img_points.append(corners)
            print('Image ' + filename + ' is valid for calibration')

    k = np.zeros((3, 3))
    d = np.zeros((4, 1))
    dims = img_shape[::-1]
    valid_img_count = len(obj_points)

    if valid_img_count > 0:
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(valid_img_count)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(valid_img_count)]

        print('Beginning calibration process...')
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

    print("Calibration done!")
    print("Found " + str(valid_img_count) + " valid images for calibration")
    return k, d, dims


# Save calibration to file -> use of pickle serialization
def save_calibration(path, dims, k, d ):
    with open(path, 'wb') as f:
        pickle.dump(dims, f)
        pickle.dump(k, f)
        pickle.dump(d, f)


# Load calibration from file -> use of pickle serialization
def load_calibration(path):
    k = np.zeros((3, 3))
    d = np.zeros((4, 1))
    dims = np.zeros(2)

    with open(path, 'rb') as f:
        k = pickle.load(f)
        d = pickle.load(f)
        dims = pickle.load(f)

    return k, d, dims


# Undistort FishEye image with given calibration parameters
def undistort(src_image, k, d, dims):
    dim1 = src_image.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    assert dim1[0] / dim1[1] == dims[0] / dims[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, dims, cv2.CV_16SC2)
    undistorted_img = cv2.remap(src_image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


# Undistort FishEye image (from a file) with given calibration parameters
def undistort_from_file(img_path, k, d, dims):
    img = cv2.imread(img_path)
    return undistort(img, k, d, dims)


# Undistort FishEye image (from a file) with given calibration parameters
# This function has few others parameters to set the field of view of the undistorted image
# Not suitable for large FOV FishEye camera (near 180°)
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


# Tool function to save image to disk at the given path
def save_image(image, image_path):
    cv2.imwrite(image_path, image)


# Tool function to save image to disk with automatic path definition (from the original image path)
# -> use for test only
def save_undistort_image(undistorted_img, img_path, suffix):
    folder = os.path.dirname(img_path)
    save_folder = folder + '/Undistort/'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    filename = os.path.basename(img_path)
    filename, file_extension = os.path.splitext(filename)
    cv2.imwrite(save_folder + filename + suffix + file_extension, undistorted_img)

# Tool function to test chessboard detection on all images in subfolder "CalibImages"
def test_chessboard_detection():
    images = glob.glob('CalibImages/*.png')
    for filename in images:
        ret, _, _ = detect_chessboard(filename, True)
        if ret:
            print('Chessboard detected correctly in image ' + filename)
        else:
            print('Chessboard not detected in image ' + filename)

# Tool function to test calibration function on all images in subfolder "CalibImages"
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

# Tool function to test undistortion
# Calibration file "calibration.txt" must be in the current folder
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
        undistorted_img = undistort_from_file(filename, k, d, dims)
        save_undistort_image(undistorted_img, filename, '')

    images = glob.glob('SrcImages/Ikomia/*.png')
    for filename in images:
        undistorted_img = undistort_from_file(filename, k, d, dims)
        save_undistort_image(undistorted_img, filename, '')

    print("Undistort process finished successfully")

# Tool function to test deformation measure on a specific image
# Calibration file "calibration.txt" must be in the current folder
def test_measure(img_path):
    src_image = cv2.imread(img_path)
    k, d, dims = load_calibration('calibration.txt')
    undistorted_img = undistort(src_image, k, d, dims)
    deformation_measure(src_image, undistorted_img)

if __name__ == '__main__':
    #test_measure('CalibImages/img14.png')
    test_chessboard_detection()
    #test_calibration()
    #test_undistort()


