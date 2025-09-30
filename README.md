<h1 align="center">
  <font color="purple">
    CAR: Contrast-Agnostic Deformable Medical Image Registration with Contrast-Invariant Latent Regularization (WBIR 2024)
  </font>
</h1>

<p align="center">
  Official Implementation of the paper  
  <i>"CAR: Contrast-Agnostic Deformable Medical Image Registration with Contrast-Invariant Latent Regularization", WBIR 2024. Written by Yinsong Wang, Siyi Du, Shaoming Zheng, Xinzhe Luo and Chen Qin</i>
</p>

<p align="center">
  <img src="network.png" alt="network" width="1000"/>
</p>

# Prerequisites
- `Python 3.8`
- `Pytorch >=1.10.1`
- `NumPy`
- `NiBabel`

# Data
Due to redistribution restrictions, we cannot share the original or processed data. The datasets are publicly available upon application at:
[CamCAN dataset](https://opendata.mrc-cbu.cam.ac.uk/projects/camcan/)

[CMRxRecon 2023 dataset](https://cmrxrecon.github.io/Home.html)

# Training
```
python train_CamCAN.py for the CamCAN dataset
python train_CMR.py for the CMRxRecon dataset
```
Noted that you may need to customize your own dataloader. Add your customized dataloader to code/Functions.py

# Inference
```
python test_CamCAN.py for the CamCAN dataset
python test_CMR.py for the CMRxRecon dataset
```

# Publication
If you make use of the code or if you found the code useful, please cite the paper in any resulting publications.
- **CAR: Contrast-Agnostic Deformable Medical Image Registration with Contrast-Invariant Latent Regularization**  
```
@inproceedings{wang2024car,
  title={CAR: Contrast-Agnostic Deformable Medical Image Registration with Contrast-Invariant Latent Regularization},
  author={Wang, Yinsong and Du, Siyi and Zheng, Shaoming and Luo, Xinzhe and Qin, Chen},
  booktitle={International Workshop on Biomedical Image Registration},
  pages={308--318},
  year={2024},
  organization={Springer}
}
```

---

🚧 Code is coming soon!

