import cv2
import matplotlib.pyplot as plt


def show(license_plate_crop, license_plate_crop_thresh):
    license_plate_crop_rgb = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2RGB)
    license_plate_crop_thresh_rgb = cv2.cvtColor(
        license_plate_crop_thresh, cv2.COLOR_BGR2RGB
    )
    plt.subplot(1, 2, 1)
    plt.imshow(license_plate_crop_rgb)
    plt.title("Original Crop")

    plt.subplot(1, 2, 2)
    plt.imshow(license_plate_crop_thresh_rgb)
    plt.title("Threshold")

    plt.show()
