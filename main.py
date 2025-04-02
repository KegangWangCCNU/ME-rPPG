import mediapipe as mp
import cv2, json
import onnxruntime as ort
import numpy as np
from scipy.signal import welch

def crop_face(frame):
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True) as fm:
        landmarks = fm.process(frame).multi_face_landmarks[0].landmark
        xaxis = [i.x for i in landmarks if i.x>0]
        yaxis = [i.y for i in landmarks if i.y>0]
        xmin, xmax = min(xaxis)*frame.shape[1], max(xaxis)*frame.shape[1]
        ymin, ymax = min(yaxis)*frame.shape[0], max(yaxis)*frame.shape[0]
        img = frame[round(ymin):round(ymax), round(xmin):round(xmax), ::-1].astype('float32')/255
        return cv2.resize(img, (36, 36), interpolation=cv2.INTER_AREA)

def load_state(path):
    with open(path, 'r') as f:
        return json.load(f)
     
def load_model(path):
    model = ort.InferenceSession(path)
    def run(img, state, dt=1/30):
        result = model.run(None, {"arg_0.1": img[None, None], "onnx::Mul_37": [dt], **state})
        bvp, new_state = result[0][0, 0], result[1:]
        return bvp, dict(zip(state, new_state))
    return run

def get_hr(y, sr=30, hr_min=30, hr_max=180):
    p, q = welch(y, sr, nfft=int(1e5 / sr), nperseg=np.min((len(y) - 1, 256)))
    return p[(p > hr_min / 60) & (p < hr_max / 60)][np.argmax(
        q[(p > hr_min / 60) & (p < hr_max / 60)])] * 60
