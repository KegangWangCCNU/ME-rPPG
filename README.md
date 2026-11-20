# ME-rPPG
The official code of [this paper](https://rppgdemo.kegang.wang/merppg.pdf). We implemented memory-efficient real-time step-by-step inference using ME-rPPG. ME-rPPG is an early version of [FacePhys (NeurIPS 2026)](https://facephys.com)
![image](https://github.com/user-attachments/assets/ca75b574-834f-41f2-b08e-7b2dd2602d67)

## Usage 

The ME model is now available in [open-rppg](https://github.com/KegangWangCCNU/open-rppg).

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

## License

Copyright (c) 2025 FacePhys

The FacePhys models and model architecture implementations are maintained
by [FacePhys](https://facephys.com).

The code, pretrained weights, and model artifacts are available for
non-commercial academic research, including use, modification, and
redistribution, under the [FacePhys Academic Research License](LICENSE).

Commercial use requires a separate written license.
For commercial licensing and the latest models, contact
[FacePhys](https://facephys.com).

## Paper

Please cite the FacePhys paper when using these materials in research:

[FacePhys — NeurIPS 2026 paper page](https://neurips.cc/virtual/2026/loc/paris/poster/150724)

## Citation
```
@article{wang2025memory,
  title={Memory-efficient Low-latency Remote Photoplethysmography through Temporal-Spatial State Space Duality},
  author={Wang, Kegang and Tang, Jiankai and Fan, Yuxuan and Ji, Jiatong and Shi, Yuanchun and Wang, Yuntao},
  journal={arXiv preprint arXiv:2504.01774},
  year={2025}
}
```
