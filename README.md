<h1 align="center">
  <font color="purple">
    CAR: Contrast-Agnostic Deformable Medical Image Registration with Contrast-Invariant Latent Regularization (WBIR 2024)  
    <a href="[https://arxiv.org/abs/xxxx.xxxxx](https://arxiv.org/abs/2408.05341)" target="_blank" style="text-decoration:none; color:blue; font-size:24px;">[Paper]</a>
  </font>
</h1>

<p align="center">
  Official Implementation of the paper  
  <i>"CAR: Contrast-Agnostic Deformable Medical Image Registration with Contrast-Invariant Latent Regularization", WBIR 2024. Written by Yinsong Wang, Siyi Du, Shaoming Zheng, Xinzhe Luo and Chen Qin</i>
</p>

<p align="center">
  <img src="network.png" alt="network" width="1000"/>
</p>

---

# Prerequisites
- `Python 3.8`
- `PyTorch >=1.10.1`
- `NumPy`
- `NiBabel`

# Data
Due to redistribution restrictions, we cannot share the original or processed data. The datasets are publicly available upon application at:  
- [CamCAN dataset](https://opendata.mrc-cbu.cam.ac.uk/projects/camcan/)  
- [CMRxRecon 2023 dataset](https://cmrxrecon.github.io/Home.html)  

# Training
```bash
python train_CamCAN.py  # for the CamCAN dataset
python train_CMR.py     # for the CMRxRecon dataset

# Inference
```bash
python test_CamCAN.py  # for the CamCAN dataset
python test_CMR.py     # for the CMRxRecon dataset
