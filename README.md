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
> [Paper](https://ieeexplore.ieee.org/document/10945458) | [Code](https://github.com/copawloroous/SGMAE) | [​Relevant Introduction](https://wdy.nuist.edu.cn/2025/0429/c7358a286472/page.htm)

<sup>1</sup> the Institute of Optics and Electronics, <sup>2</sup> the State Key Laboratory Cultivation Base of Atmospheric Optoelectronic Detection and Information Fusion, <sup>3</sup> Jiangsu International Joint Laboratory on Meteorological Photonics and Optoelectronic Detection, <sup>4</sup> Jiangsu Engineering Research Center for Intelligent Optoelectronic Sensing Technology of Atmosphere, <sup>5</sup> Nanjing University of Information Science and Technology, Nanjing 210044, China, <sup>6</sup> the Faculty of Computer Science, China University of Geosciences, Wuhan 430074, China, <sup>7</sup> the Hyperspectral Computing Laboratory, Department of Technology of Computers and Communications, Escuela Politécnica, University of Extremadura, 10003 Cáceres, Spain, <sup>*</sup> Corresponding author

</div>

## :blush: Should you encounter any issues, feel free to contact the author at any time! If this project helps you, please give it a ⭐ ！

## Environment Requirements
| Library         | Version      |
|-----------------|--------------|
| Python          | 3.8.18       |
| PyTorch         | 1.12.1       |
| torch-geometric | 2.4.0        |
| scikit-learn    | 1.3.2        |
| numpy           | 1.23.5       |

> ​**Note**: CUDA extensions (e.g., `+cu113`) are backend-specific. Minor version inconsistencies may require troubleshooting but do not affect core functionality. To install PyG (torch-geometric etc.), please check [tutorial](https://blog.csdn.net/copawloroous/article/details/140201394?spm=1001.2014.3001.5501).

## Usage Instructions
1. ​**Self-Supervised Feature Extraction**​  
   Set `S2GAE=y` in arguments to enable graph autoencoder pre-training.
2. ​**Baseline Comparison**​  
   Set `S2GAE=n` to run without self-supervised features.
3. ​**Dataset Compatibility**​  
   Current hyperparameters are optimized for the [Houston 2018 dataset](https://pan.baidu.com/s/1hnVsruXw1QozOeUVh8Fymw?pwd=UIST). 
4. ​**Hardware Recommendation**​  
   An NVIDIA RTX 4090 or equivalent GPU is recommended for training efficiency.

## Authors
- ​**Zhenghao Hu**​  
  B.Sc.Eng. Candidate, Optoelectronic Information Science and Engineering  
  School of Physics and Optoelectronic Engineering  
  Nanjing University of Information Science and Technology, China  
  Email: [202213880076@nuist.edu.cn](mailto:202213880076@nuist.edu.cn)  
  Research: Hyperspectral Image Processing, Computer Vision, Machine Learning  
  [Google Scholar](https://scholar.google.com/citations?user=F5Qx7kAAAAAJ&hl=zh-CN&oi=sra) | [Github Profile](https://github.com/copawloroous) | [ORCID Profile](https://ieeexplore.ieee.org/author/721998129448425) | [IEEE Profile](https://ieeexplore.ieee.org/author/721998129448425)


- ​**Advisor: Prof. Bing Tu**​  
  Professor and PhD Supervisor  
  Nanjing University of Information Science & Technology  
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
