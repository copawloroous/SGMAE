# SGMAE: Self-Supervised Graph Masked Autoencoders for Hyperspectral Image Classification

> IEEE Transactions on Geoscience and Remote Sensing (2025)  
> [Paper](https://ieeexplore.ieee.org/document/10945458) | [Code](https://github.com/copawloroous/SGMAE) | [​Relevant Introduction](https://wdy.nuist.edu.cn/2025/0429/c7358a286472/page.htm)


##:blush: Should you encounter any issues, feel free to contact the author at any time!

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
