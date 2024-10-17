import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze frame
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Check if any faces were detected
        if len(results) > 0:
            # Get dominant emotion from the first face detected
            emotion = results[0]['dominant_emotion']
           # print(emotion)
        else:
            emotion = "No face detected"

        # Display emotion on frame
        cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    except Exception as e:
        print("Error:", e)
        # Optionally display error on frame
        cv2.putText(frame, "Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Emotion Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
