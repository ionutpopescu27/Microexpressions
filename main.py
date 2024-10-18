import cv2
import requests
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
import os
import openai 
import time
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os. getenv('key')


# Function to download a file
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)

# Download the model and the image
download_file("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", "face_landmarker_v2_with_blendshapes.task")
download_file("https://storage.googleapis.com/mediapipe-assets/business-person.png", "image.png")

# Function to draw landmarks on the image
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions = mp.solutions
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image

# Function to plot blendshapes bar graph
def plot_face_blendshapes_bar_graph(face_blendshapes):
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, labels=face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()

def generate_emotion_analysis(blendshapes_list):
    # Convert emotion scores to a readable string
    blendshapes_dict = {blendshape.category_name: blendshape.score for blendshape in sorted_blendshapes}

    blendshapes_list = "\n".join([f"{name}: {score:.4f}" for name, score in blendshapes_dict.items()])

    prompt = f"""
    Based on the following blendshape scores extracted from facial expressions, provide a list of microemotions
    Emotion Scores:
    {blendshapes_list}
    """

    response = openai.chat.completions.create(
        messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="gpt-4",
    )

    analysis = response.choices[0].message.content.strip()
    return analysis


# Load and display the image using OpenCV
# Open the video capture
cap = cv2.VideoCapture('/home/mihai/Documents/Rebeldot/rebeldot5/emotion.mp4')  # Change to your video file path if needed

# Check if the capture was successful
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create a FaceLandmarker object
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the output file path in the same directory as the script
output_file_path = os.path.join(current_directory, "emotion_analysis_output.txt")
with open(output_file_path, "a") as output_file:  # Changed to append mode "a"
    start_time = time.time()  # Record the start time
    analysis_done = False  # Flag to ensure analysis is done only once

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        # If a frame was read successfully
        if ret:
            # Convert the frame from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a MediaPipe image from the RGB frame
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect face landmarks from the input image
            detection_result = detector.detect(image)

            # Process the detection result (visualize it)
            annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)

            # Convert annotated image back to BGR for OpenCV
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            # Display the annotated image
            cv2.imshow("Annotated Image", annotated_image_bgr)

            # Emotion analysis part
            if detection_result.face_blendshapes:
                sorted_blendshapes = sorted(detection_result.face_blendshapes[0], key=lambda x: x.score, reverse=True)

                # Get the current elapsed time
                elapsed_time = time.time() - start_time  # Calculate elapsed time in seconds
                # Call analysis after 10 seconds
                if elapsed_time >= 10:
                    print("dadadada")
                    # Generate the emotion analysis
                    analysis = generate_emotion_analysis(sorted_blendshapes)
                    analysis_done = True  # Set flag to True to avoid calling again
                    elapsed_time = 0
                    # Get the current second of the video
                    current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert milliseconds to seconds

                    try:
                        if isinstance(analysis, str):
                            output_file.write(f"\nEmotion Analysis at {current_time_sec:.2f} seconds:\n")
                            output_file.write(analysis + "\n")
                            output_file.flush()  # Flush the buffer to ensure data is written
                        else:
                            print("Analysis is not a string:", type(analysis))
                            output_file.write(f"\nEmotion Analysis (not a string) at {current_time_sec:.2f} seconds:\n")
                            output_file.write(str(analysis) + "\n")
                            output_file.flush()  # Flush the buffer to ensure data is written
                    except Exception as e:
                        print(f"Error writing to file: {e}")

                  #  print(f"\nEmotion Analysis at {current_time_sec:.2f} seconds:")
                 #   print(analysis)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error: Could not read frame.")
            break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

# Generate the emotion analysi