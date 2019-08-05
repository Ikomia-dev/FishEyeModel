import fisheye_module
import argparse


# Python script to launch calibration process from terminal
# Usage: calibrate.py "folder_of_calibration_images" "output_path"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_folder",
        help="Folder where the calibration images are stored"
    )

    parser.add_argument(
        "output_file_path",
        help="Path where the calibration file will be saved"
    )

    args = parser.parse_args()

    if not args.image_folder:
        raise RuntimeError("Folder of calibration images is required")

    if not args.output_file_path:
        args.output_file_path = "calibration.txt"

    k, d, dims = fisheye_module.calibrate(args.image_folder)
    print('----- Calibration results -----')
    print("Dimensions =" + str(dims))
    print("K = np.array(" + str(k.tolist()) + ")")
    print("D = np.array(" + str(d.tolist()) + ")")

    fisheye_module.save_calibration(args.output_file_path, k, d, dims)
    print("Calibration file saved successfully")

