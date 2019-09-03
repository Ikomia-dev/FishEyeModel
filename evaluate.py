from UndistortFishEye import fisheye_module
import argparse

# Python script to launch evaluate process from terminal
# Usage: evaluate.py "source_image_path" -c "calibration_file_path"
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

    args = parser.parse_args()

    if not args.source_image_path:
        raise RuntimeError("Source image path could not be empty")

    if not args.calibration_file_path:
        raise RuntimeError("Calibration file path could not be empty")

    fisheye_module.evaluate(args.source_image_path, args.calibration_file_path)
    print("Evaluate process finished successfully")
