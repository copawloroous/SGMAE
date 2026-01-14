<div align="center">
<h1 align="center">SGMAE</h1>

<h3>Self-Supervised Graph Masked Autoencoders for Hyperspectral Image Classification
</h3>

[Zhenghao Hu](https://ieeexplore.ieee.org/author/721998129448425)<sup>1,2,3,4,5 </sup>, 
[Bing Tu](https://ieeexplore.ieee.org/author/37086303208)<sup>1,2,3,4,5 *</sup>, 
[Bo Liu](https://ieeexplore.ieee.org/author/37404906400)<sup>1,2,3,4,5 </sup>, 
[Yan He](https://ieeexplore.ieee.org/author/279730212927568)<sup>1,2,3,4,5 </sup>,
[Jun Li](https://ieeexplore.ieee.org/author/38323572200)<sup>6 </sup>,
[Antonio Plaza](https://ieeexplore.ieee.org/author/37299689800)<sup>7 </sup>

> IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS 2025)  
> [Paper](https://ieeexplore.ieee.org/document/10945458) | [Code](https://github.com/copawloroous/SGMAE) | Relevant News [[1]](https://wdy.nuist.edu.cn/2025/0429/c7358a286472/page.htm)  [[​2]](https://wdy.nuist.edu.cn/2025/1127/c2984a292554/page.htm) | [​中文版](https://pan.baidu.com/s/1ZZrkkLwWTy3zf3K5E7dwgA?pwd=abyr)

<sup>1</sup> the Institute of Optics and Electronics,  
<sup>2</sup> the State Key Laboratory Cultivation Base of Atmospheric Optoelectronic Detection and Information Fusion,  
<sup>3</sup> Jiangsu International Joint Laboratory on Meteorological Photonics and Optoelectronic Detection,  
<sup>4</sup> Jiangsu Engineering Research Center for Intelligent Optoelectronic Sensing Technology of Atmosphere,  
<sup>5</sup> Nanjing University of Information Science and Technology, Nanjing 210044, China,  
<sup>6</sup> the Faculty of Computer Science, China University of Geosciences, Wuhan 430074, China,  
<sup>7</sup> the Hyperspectral Computing Laboratory, Department of Technology of Computers and Communications, Escuela Politécnica, University of Extremadura, 10003 Cáceres, Spain,  
<sup>*</sup> Corresponding author

</div>


## 🤗Should you encounter any issues, feel free to contact the author at any time! If this project helps you, please give it a ⭐ ！

## Environment Requirements
| Library         | Version      |
|-----------------|--------------|
| Python          | 3.8.18       |
| PyTorch         | 1.12.1       |
| torch-geometric | 2.4.0        |
| scikit-learn    | 1.3.2        |
| numpy           | 1.23.5       |

> ​**Note**: CUDA extensions (e.g., `+cu113`) are backend-specific. Minor version inconsistencies may require troubleshooting but do not affect core functionality. To install PyG (torch-geometric), please check [tutorial](https://blog.csdn.net/copawloroous/article/details/140201394?spm=1001.2014.3001.5501).

## Usage Instructions
1. ​**Self-Supervised Feature Extraction**​  
   Set `S2GAE=y` in arguments of [main.py](main.py) to enable graph autoencoder pre-training.
2. ​**Baseline Comparison**​  
   Set `S2GAE=n` to run without self-supervised features.
3. ​**Method Comparison**​  
   Run [CNN/main.py](Method%20Comparison/CNN/main.py) or [SSRN/main.py](Method%20Comparison/SSRN/main.py).
4. ​**Dataset Compatibility**​  
   Current hyperparameters are optimized for the [Houston 2018 dataset](https://pan.baidu.com/s/1hnVsruXw1QozOeUVh8Fymw?pwd=UIST). 
5. ​**Hardware Recommendation**​  
   An NVIDIA RTX 4090 or equivalent GPU is recommended for training efficiency. If you encounter an error indicating insufficient GPU memory, you can try reducing the pca_components to 20 or decreasing the patch_size to 5.

## This is an experiment the author has run for you to demonstrate the code execution results. Feel free to reproduce it!（Tested on Houston 2018 dataset.）

## 📊 Overall Accuracy (OA, %) Under Different Patch Sizes

| Patch Size | SGMAE (with pretraining) | SGMAE (no pretraining) | CNN   | SSRN  |
|-----------:|--------------------------:|------------------------:|------:|------:|
| **5**      | 96.56                     | 93.35                   | 91.81 | 93.48 |
| **7**      | 97.13                     | 95.04                   | 92.93 | 95.64 |

## ⏱️ Training & Testing Time (Patch Size = 5)

| Method              | Pretraining Time (s) | Training Time (s) | Test Time (s) |
|---------------------|----------------------:|------------------:|--------------:|
| **SGMAE (y)**       | 104.4                | 218.1             | 18.28         |
| **SGMAE (n)**       | –                    | 218.1             | 18.28         |
| **CNN**             | –                    | 109.1             | 15.9          |
| **SSRN**            | –                    | 224.5             | 19.5          |


Computer Configuration: i9-12900KF | RTX 3090 Ti (24GB VRAM) | 64GB RAM

## Authors
- ​**Zhenghao Hu**​  
  B.Eng. Student in Optoelectronic Information Science and Engineering  
  School of Physics and Optoelectronic Engineering  
  Nanjing University of Information Science and Technology, China


  Ph.D. Student in Pattern Recognition and Intelligent Systems  
  Institute of Automation  
  Chinese Academy of Sciences, China


  Email: [202213880076@nuist.edu.cn](mailto:202213880076@nuist.edu.cn)  
  Research: Machine Learning, Computer Vision, Pattern Recognition  
  [Google Scholar](https://scholar.google.com/citations?user=F5Qx7kAAAAAJ&hl=zh-CN&oi=sra) | [Github Profile](https://github.com/copawloroous) | [ORCID Profile](https://orcid.org/0009-0004-0285-5763) | [IEEE Profile](https://ieeexplore.ieee.org/author/721998129448425)


- ​**Advisor: Prof. Bing Tu**​  
  Professor and PhD Supervisor  
  School of Physics and Optoelectronic Engineering  
  Nanjing University of Information Science and Technology, China  
  [Google Scholar](https://scholar.google.com/citations?user=iMuSewsAAAAJ&hl=zh-CN&oi=sra) | [University Profile](https://faculty.nuist.edu.cn/tubing/zh_CN/index.htm)

## Citation

If you find this code useful in your research, please cite the following paper:

```bibtex
@article{hu2025self,
  title={Self-Supervised Graph Masked Autoencoders for Hyperspectral Image Classification},
  author={Hu, Zhenghao and Tu, Bing and Liu, Bo and He, Yan and Li, Jun and Plaza, Antonio},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  publisher={IEEE}
}

