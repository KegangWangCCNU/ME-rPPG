from main import *

state = load_state('state.json')
model = load_model('model.onnx')

bvp = []
video_path = r"C:\Users\16252\Downloads\29_3.mp4"
cap = cv2.VideoCapture(video_path)
while 1:
    _, frame = cap.read()
    if not _:
        break
    facial_img = crop_face(frame)
    output, state = model(facial_img, state)
    bvp.append(output)
cap.release()

print(f'Heart Rate is {get_hr(bvp):.2f}')
