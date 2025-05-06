from ultralytics import YOLO
import cv2
from sort import *
from utlis import get_car, read_license_plate, write_csv
from crop_img_show import show


SAMPLE_VIDEO = "Traffic Control CCTV.mp4"
MODEL_PATH = "best.pt"

results = {}  # Dictionary to store detection results for each frame

mot_tracker = Sort()

# local models
coco_model = YOLO("yolov8n.pt")
license_plate_detection = YOLO(MODEL_PATH)

# load video
cap = cv2.VideoCapture(SAMPLE_VIDEO)  # Open the video file for processin


vehicles = [2, 3, 5, 7]  # 2: car, 3: motorcycle, 5: bus, 7: truck

# read frames
frame_num = -1  # Initialize frame counter
ret = True  # Variable to keep track of whether frames are read successfully

while ret:
    frame_num += 1
    ret, frame = cap.read()  # The actual video frame in the form of a NumPy array.
    if ret:
        # if frame_num > 10:
        # break
        results[frame_num] = {}
        # detec vehicles
        detections = coco_model(frame)[0]
        detections_ = []  # List to store valid detections ((vehicles only))
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plate
        license_plates = license_plate_detection(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(
                    license_plate_crop, cv2.COLOR_BGR2GRAY
                )
                _, license_plate_crop_thresh = cv2.threshold(
                    license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
                )

                """Pixels below 64 will become 255 (white).
                Pixels above 64 will become 0 (black)."""

                # cv2.imshow("orginal crop", license_plate_crop)
                # cv2.imshow("threshold", license_plate_crop_thresh)
                # show(license_plate_crop, license_plate_crop_thresh)
                # cv2.waitKey(1)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(
                    license_plate_crop_thresh
                )

                if license_plate_text is not None:
                    results[frame_num][car_id] = {
                        "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                        "license_plate": {
                            "bbox": [x1, y1, x2, y2],
                            "text": license_plate_text,
                            "bbox_score": score,
                            "text_score": license_plate_text_score,
                        },
                    }
# write results
write_csv(results, "./test.csv")
