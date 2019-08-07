from UndistortFishEye import fisheye_module
import argparse

# Python script to launch undistort process from terminal
# Usage: undistort.py "source_image_path" "calibration_file_path" "output_image_path"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_image_path",
        help="Path of the source image to undistort"
    )

    parser.add_argument(
        "-c",
        dest="calibration_file_path",
        default="calibration.txt",
        help="Path of the calibration file"
    )

    parser.add_argument(
        "-o",
        dest="output_image_path",
        default="undistort.png",
        help="Path where the result undistorted image will be saved"
    )

    args = parser.parse_args()

    if not args.source_image_path:
        raise RuntimeError("Source image path could not be empty")

    if not args.calibration_file_path:
        raise RuntimeError("Calibration file path could not be empty")

    k, d, dims = fisheye_module.load_calibration(args.calibration_file_path)
    undistorted_img = fisheye_module.undistort_from_file(args.source_image_path, k, d, dims)
    fisheye_module.save_image(undistorted_img, args.output_image_path)
    print("Undistort process finished successfully")
