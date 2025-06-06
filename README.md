# ME-rPPG
The official code of [this paper](https://rppgdemo.kegang.wang/merppg.pdf). We implemented memory-efficient real-time step-by-step inference using ME-rPPG.
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
Demo URL: [https://rppgdemo.kegang.wang/](https://rppgdemo.kegang.wang/)  
Source code: [https://github.com/Health-HCI-Group/ME-rPPG-demo](https://github.com/Health-HCI-Group/ME-rPPG-demo) 

## Training Framework 
The ME-rPPG was trained on RLAP using PhysBench, with the full code coming soon. 

## Citation
```
@article{wang2025memory,
  title={Memory-efficient Low-latency Remote Photoplethysmography through Temporal-Spatial State Space Duality},
  author={Wang, Kegang and Tang, Jiankai and Fan, Yuxuan and Ji, Jiatong and Shi, Yuanchun and Wang, Yuntao},
  journal={arXiv preprint arXiv:2504.01774},
  year={2025}
}
```
