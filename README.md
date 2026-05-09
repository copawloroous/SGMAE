<div align="center">

# 🌐 SGMAE

### Self-Supervised Graph Masked Autoencoders for Hyperspectral Image Classification

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TGRS%202025-blue)](https://ieeexplore.ieee.org/document/10945458)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/copawloroous/SGMAE)
[![中文版](https://img.shields.io/badge/中文版-Baidu%20Pan-red)](https://pan.baidu.com/s/1ZZrkkLwWTy3zf3K5E7dwgA?pwd=abyr)

---

[Zhenghao Hu](https://ieeexplore.ieee.org/author/721998129448425)<sup>1,2,3,4,5</sup> &nbsp;|&nbsp;
[Bing Tu](https://ieeexplore.ieee.org/author/37086303208)<sup>1,2,3,4,5 *</sup> &nbsp;|&nbsp;
[Bo Liu](https://ieeexplore.ieee.org/author/37404906400)<sup>1,2,3,4,5</sup> &nbsp;|&nbsp;
[Yan He](https://ieeexplore.ieee.org/author/279730212927568)<sup>1,2,3,4,5</sup> &nbsp;|&nbsp;
[Jun Li](https://ieeexplore.ieee.org/author/38323572200)<sup>6</sup> &nbsp;|&nbsp;
[Antonio Plaza](https://ieeexplore.ieee.org/author/37299689800)<sup>7</sup>

> **Published in:** *IEEE Transactions on Geoscience and Remote Sensing* (IEEE TGRS 2025)
>
> 📰 **Featured News:** [[1]](https://wdy.nuist.edu.cn/2025/0429/c7358a286472/page.htm) &nbsp;[[2]](https://wdy.nuist.edu.cn/2025/1127/c2984a292554/page.htm)

</div>

<details>
<summary><b>📍 Author Affiliations</b> (click to expand)</summary>

<sup>1</sup> Institute of Optics and Electronics  
<sup>2</sup> State Key Laboratory Cultivation Base of Atmospheric Optoelectronic Detection and Information Fusion  
<sup>3</sup> Jiangsu International Joint Laboratory on Meteorological Photonics and Optoelectronic Detection  
<sup>4</sup> Jiangsu Engineering Research Center for Intelligent Optoelectronic Sensing Technology of Atmosphere  
<sup>5</sup> Nanjing University of Information Science and Technology, Nanjing 210044, China  
<sup>6</sup> Faculty of Computer Science, China University of Geosciences, Wuhan 430074, China  
<sup>7</sup> Hyperspectral Computing Laboratory, Department of Technology of Computers and Communications, Escuela Politécnica, University of Extremadura, 10003 Cáceres, Spain  
<sup>*</sup> Corresponding author

</details>

---

> 🤗 **Should you encounter any issues, feel free to contact the author at any time!**
> If this project helps you, please give it a ⭐ — your support means a lot!

---

## 🛠️ Environment Requirements

| Library         | Version  |
|-----------------|----------|
| Python          | 3.8.18   |
| PyTorch         | 1.12.1   |
| torch-geometric | 2.4.0    |
| scikit-learn    | 1.3.2    |
| numpy           | 1.23.5   |

> **Note:** CUDA extensions (e.g., `+cu113`) are backend-specific. Minor version inconsistencies may require troubleshooting but do not affect core functionality. For installing PyG (torch-geometric), please refer to this [tutorial](https://blog.csdn.net/copawloroous/article/details/140201394?spm=1001.2014.3001.5501).

---

## 🚀 Usage Instructions

1. **Self-Supervised Feature Extraction**  
   Set `S2GAE=y` in the arguments of [`main.py`](main.py) to enable graph autoencoder pre-training.

2. **Baseline Comparison**  
   Set `S2GAE=n` to run without self-supervised features.

3. **Method Comparison**  
   Run [`CNN/main.py`](Method%20Comparison/CNN/main.py) or [`SSRN/main.py`](Method%20Comparison/SSRN/main.py).

4. **Dataset Compatibility**  
   Current hyperparameters are optimized for the [Houston 2018 dataset](https://pan.baidu.com/s/1hnVsruXw1QozOeUVh8Fymw?pwd=UIST).

5. **Hardware Recommendation**  
   An **NVIDIA RTX 4090** or equivalent GPU is recommended for training efficiency. If you encounter insufficient GPU memory errors, try reducing `pca_components` to `20` or decreasing `patch_size` to `5`.

---

## 🧪 Reference Experiments

> *The following experiments were run by the author for reference. Feel free to reproduce them!* (Tested on the Houston 2018 dataset.)

### 📊 Overall Accuracy (OA, %) Under Different Patch Sizes

| Patch Size | SGMAE (w/ pretraining) | SGMAE (w/o pretraining) | CNN   | SSRN  |
|:----------:|:----------------------:|:-----------------------:|:-----:|:-----:|
| **5**      | **96.56**              | 93.35                   | 91.81 | 93.48 |
| **7**      | **97.13**              | 95.04                   | 92.93 | 95.64 |

### ⏱️ Training & Testing Time (Patch Size = 5)

| Method        | Pretraining Time (s) | Training Time (s) | Test Time (s) |
|---------------|:--------------------:|:-----------------:|:-------------:|
| **SGMAE (y)** | 104.4                | 218.1             | 18.28         |
| **SGMAE (n)** | –                    | 218.1             | 18.28         |
| **CNN**       | –                    | 109.1             | 15.90         |
| **SSRN**      | –                    | 224.5             | 19.50         |

> **Hardware:** Intel i9-12900KF · NVIDIA RTX 3090 Ti (24 GB VRAM) · 64 GB RAM

---

## 👥 Authors

### Zhenghao Hu — *First Author*

🎓 **Education**

- **B.Eng. Student** in Optoelectronic Information Science and Engineering  
  *School of Physics and Optoelectronic Engineering*  
  Nanjing University of Information Science and Technology, China

- **Incoming Ph.D. Student** in Pattern Recognition and Intelligent Systems  
  *Institute of Automation*  
  Chinese Academy of Sciences, China

🔬 **Research Interests**  
Machine Learning · Computer Vision · Pattern Recognition

📫 **Contact**  
Current: [huzhenghao2026@ia.ac.cn](mailto:huzhenghao2026@ia.ac.cn)  
Previous: ~~[202213880076@nuist.edu.cn](mailto:202213880076@nuist.edu.cn)~~

🔗 **Profiles**  
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=F5Qx7kAAAAAJ&hl=zh-CN&oi=sra)
[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/copawloroous)
[![ORCID](https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0004-0285-5763)
[![IEEE](https://img.shields.io/badge/IEEE-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/author/721998129448425)

---

### Prof. Bing Tu — *Advisor & Corresponding Author*

🎓 Professor and Ph.D. Supervisor  
*School of Physics and Optoelectronic Engineering*  
Nanjing University of Information Science and Technology, China

🔗 **Profiles**  
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=iMuSewsAAAAJ&hl=zh-CN&oi=sra)
[![Faculty Page](https://img.shields.io/badge/Faculty%20Profile-NUIST-1e3a8a)](https://faculty.nuist.edu.cn/tubing/zh_CN/index.htm)

---

## 📖 Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@article{hu2025self,
  title   = {Self-Supervised Graph Masked Autoencoders for Hyperspectral Image Classification},
  author  = {Hu, Zhenghao and Tu, Bing and Liu, Bo and He, Yan and Li, Jun and Plaza, Antonio},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2025},
  publisher = {IEEE}
}
```
