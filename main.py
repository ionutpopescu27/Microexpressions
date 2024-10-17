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

# Load and display the image using OpenCV
img = cv2.imread("image.png")
#cv2.imshow("Input Image", img)
#cv2.waitKey(0)

# STEP 2: Create a FaceLandmarker object
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# STEP 3: Load the input image
image = mp.Image.create_from_file("image.png")

# STEP 4: Detect face landmarks from the input image
detection_result = detector.detect(image)

# STEP 5: Process the detection result (visualize it)
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
#cv2.imshow("Annotated Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
#cv2.waitKey(0)

# Plot face blendshapes bar graph
#plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])

# Print the facial transformation matrixes
#print(detection_result.facial_transformation_matrixes)
sorted_blendshapes = sorted(detection_result.face_blendshapes[0], key=lambda x: x.score, reverse=True)
for blendshape in sorted_blendshapes:
    print(f"{blendshape.category_name}: {blendshape.score:.4f}")

# sorted_blendshapes - ponderile sortate

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

# Generate the emotion analysis
analysis = generate_emotion_analysis(sorted_blendshapes)

print("\nEmotion Analysis:")
print(analysis)