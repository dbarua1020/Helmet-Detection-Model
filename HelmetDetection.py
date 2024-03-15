import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load classes
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Run forward pass
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Process detections
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0:  # Class ID 0 represents "person"
                # Extract coordinates of the detected person
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Extract region of interest (ROI) around the detected person
                roi = frame[center_y - h // 2:center_y + h // 2, center_x - w // 2:center_x + w // 2]

                # Convert ROI to HSV color space
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # Define lower and upper bounds for helmet color in HSV
                lower_red = np.array([0, 100, 100])
                upper_red = np.array([10, 255, 255])

                # Threshold the HSV image to get only red colors
                mask = cv2.inRange(hsv, lower_red, upper_red)

                # Count non-zero pixels in the mask (indicating presence of red color)
                red_pixels = cv2.countNonZero(mask)

                # If a certain threshold of red pixels is detected, consider it as a helmet
                if red_pixels > 1000:  # Adjust threshold as needed
                    # Helmet detected, draw a green bounding box around the person
                    cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)
                    cv2.putText(frame, 'Helmet Detected', (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # No helmet detected, draw a red bounding box around the person
                    cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), (0, 0, 255), 2)
                    cv2.putText(frame, 'No Helmet Detected', (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Helmet Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
