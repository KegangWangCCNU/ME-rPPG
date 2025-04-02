# ME-rPPG
The official code of ME-rPPG. We implemented memory-efficient real-time step-by-step inference using ME-rPPG.
![image](https://github.com/user-attachments/assets/ca75b574-834f-41f2-b08e-7b2dd2602d67)

## Inference Example 

Performing step-by-step inference through the following code. 

```python
state = load_state('state.json')
model = load_model('model.onnx')

while True:
  ........
  facial_img = crop_face(frame) # Cropping to a 36×36x3 RGB facial image
  output, state = model(facial_img, state) # Computing the BVP and updating the state
  ........
```

## Web Browser Inference Demo
The following is our implemented web browser inference demo, which operates directly within the browser without requiring video uploads or GPU acceleration.   
[https://health-hci-group.github.io/ME-rPPG-demo/](https://health-hci-group.github.io/ME-rPPG-demo/) 

## Training Framework 
The ME-rPPG was trained on RLAP using PhysBench, with the full code coming soon. 
