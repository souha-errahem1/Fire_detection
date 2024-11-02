import cv2
import ultralytics as ut

# Load the YOLO model pre-trained for fire detection
model = ut.YOLO("fire.pt")  # Replace "best.pt" with the path to your fire detection model

# Open the webcam (index 0 for default camera)
cap = cv2.VideoCapture(0)

# Set a minimum confidence threshold for detection (adjust as needed)
conf_threshold = 0.5

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Resize the frame for efficiency (optional)
    # frame = cv2.resize(frame, (640, 480))

    # Perform fire detection using YOLO
    results = model(frame)  # Use model for frame processing

    # Loop through detected objects (if any)
    if len(results) > 0:  # Check if there are any detections
        for result in results:
            # Check if boxes exist and have detections before accessing them
            if result.boxes is not None and len(result.boxes) > 0:
                x1, y1, x2, y2 = result.boxes[0].xyxy[0].tolist()  # Access first detection's bounding box coordinates
                conf = result.boxes[0].conf  # Access first detection's confidence score
                class_id = int(result.boxes[0].cls)  # Access first detection's class ID

                # Check if the detected class is fire and confidence is above threshold
                if conf > conf_threshold and class_id == 0:  # Class ID 0 for fire (assuming your model defines it this way)
                    # Extract confidence score (assuming it's a single-value tensor)
                    conf = float(result.boxes[0].conf.item())  # Convert tensor to float

                    # Use f-string to format confidence with two decimal places
                    cv2.putText(frame, f"Fire: {conf:.2f}", (int(x1) + 10, int(y1) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Draw red bounding box
                    cv2.putText(frame, f"Fire: {conf:.2f}", (int(x1) + 10, int(y1) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Display "Fire" and confidence

            # ... Handle other detection types (masks, keypoints, etc.) if needed ...

    else:
        # Optional: Display "No fire detected" message when no detections are found
        cv2.putText(frame, "No fire detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("Fire Detection", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()