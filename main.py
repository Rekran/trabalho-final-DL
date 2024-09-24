import json
import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils  
mp_face_mesh = mp.solutions.face_mesh 

# Definir pontos da parte inferior do rosto
SPECIFIC_LANDMARKS = [136, 150, 149, 176, 148, 152, 377, 378, 400, 378, 379, 365, 364, 397,367,288,
                      435,416, 215, 58, 138, 172, 192, 214,135,169,170, 140,171,175,396,369,395,394,
                      214, 210, 211, 32, 208, 199, 428, 262, 431, 430, 434, 186,212,57,43,202,106,204, 
                      182, 194, 83, 201, 200, 18, 313,421, 418,406, 335,424, 422, 273,287, 432,410,
                      0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,61,185,40,39,37, 13,312,
                      311,310,415,308,324,318,402,317,14,87,178,88,95,78,191,80,81,82]

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    return image, results

def draw_specific_landmarks(image, landmarks, landmark_indices):
    
    for idx in landmark_indices:
        
        landmark = landmarks.landmark[idx]
        h, w, _ = image.shape
        x = int(landmark.x * w)
        y = int(landmark.y * h)
  
        cv2.circle(image, (x, y), 3, (0, 0, 0), -1)


def landmarks_to_dict(landmarks):
    
    return [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in landmarks.landmark]

def extract_specific_landmarks(landmarks, landmark_indices):
    specific_landmarks = []
    for idx in landmark_indices:
        landmark = landmarks.landmark[idx]
        specific_landmarks.append({'index': idx, 'x': landmark.x, 'y': landmark.y, 'z': landmark.z})
    return specific_landmarks

cap = cv2.VideoCapture('videoplayback.mp4')
fps = cap.get(cv2.CAP_PROP_FPS) 

landmarks_dict = {}

with mp_holistic.Holistic(min_detection_confidence=0.8,
                          min_tracking_confidence=0.8,
                          model_complexity=1,
                          enable_segmentation=True) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        timestamp = current_frame / fps 

        image, results = mediapipe_detection(frame, holistic)

        if results.face_landmarks:
            draw_specific_landmarks(image, results.face_landmarks, SPECIFIC_LANDMARKS)
            face_landmarks_dict = extract_specific_landmarks(results.face_landmarks, SPECIFIC_LANDMARKS)
            landmarks_dict[timestamp] = face_landmarks_dict

        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

json_filename = "landmarks_output.json"
with open(json_filename, 'w') as json_file:
    json.dump(landmarks_dict, json_file, indent=4)

print(f"Landmarks saved in {json_filename}")
