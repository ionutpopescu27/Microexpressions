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
import threading  

load_dotenv()

openai.api_key = os.getenv('key')


def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)

download_file("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", "face_landmarker_v2_with_blendshapes.task")
download_file("https://storage.googleapis.com/mediapipe-assets/business-person.png", "image.png")


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

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


def generate_emotion_analysis_async(blendshapes_list, current_time_sec, output_file):
    blendshapes_dict = {blendshape.category_name: blendshape.score for blendshape in blendshapes_list}

    blendshapes_list_str = "\n".join([f"{name}: {score:.4f}" for name, score in blendshapes_dict.items()])

    prompt = f"""
    Analyze the following blendshape scores extracted from facial expressions to determine the likely micro-emotions and overall emotional state. Provide specific interpretations regarding whether these emotions might suggest underlying mental health conditions such as bipolar disorder, depression, or anxiety. For each interpretation, consider how combinations of facial movements could reflect:

Bipolar disorder: Rapid or extreme shifts in emotional expressions (e.g., simultaneous smiling with signs of tension), which might indicate manic or depressive phases.
Depression: Low expressiveness or subtle emotional cues (e.g., reduced smile intensity or lack of eyebrow movement) that could suggest sadness, fatigue, or lack of interest.
Anxiety: Signs of tension in the facial muscles (e.g., squinting, brow furrowing, or mouth pressing) that might reveal stress, nervousness, or apprehension.
Analyze the blendshapes to identify potential masking of emotions, where facial cues such as a smile combined with brow furrowing or squinting could indicate attempts to conceal anxiety or stress.

    Emotion Scores:
    {blendshapes_list_str}
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


    with threading.Lock():
        try:
            output_file.write(f"\nEmotion Analysis at {current_time_sec:.2f} seconds:\n")
            output_file.write(analysis + "\n")
            output_file.flush()  
        except Exception as e:
            print(f"Error writing to file: {e}")


cap = cv2.VideoCapture('emotion.mp4')  

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

output_file_path = os.path.join(current_directory, "emotion_analysis_output.txt")
with open(output_file_path, "w") as output_file:  
    
    firstFace = True
    while True:
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = detector.detect(image)

            annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)

            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            cv2.imshow("Annotated Image", annotated_image_bgr)

            if detection_result.face_blendshapes:
                if firstFace:
                    start_time = time.time() 
                    last_analysis_time = start_time
                    firstFace = False

                sorted_blendshapes = sorted(detection_result.face_blendshapes[0], key=lambda x: x.score, reverse=True)

                current_time = time.time()
                
                if current_time - last_analysis_time >= 10:
                    
                    current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  

                    threading.Thread(target=generate_emotion_analysis_async, args=(sorted_blendshapes, current_time_sec, output_file)).start()

                    last_analysis_time = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error: Could not read frame.")
            break

cap.release()
cv2.destroyAllWindows()
