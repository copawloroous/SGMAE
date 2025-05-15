### S2GAE--Self-Supervised Graph Masked Autoencoders for Hyperspectral Image Classification

### Recommended Versions for Major Third-party Libraries

python                    3.8.18

torch                     1.12.1+cu113

torch-geometric           2.4.0

torch-scatter             2.1.0+pt112cu113

torch-sparse              0.6.15+pt112cu113

torchaudio                0.12.1+cu113

torchinfo                 1.8.0

torchvision               0.13.1+cu113


scikit-learn              1.3.2

numpy                     1.23.5

matplotlib                3.7.4

Note: Minor version inconsistencies may require troubleshooting some errors but won't affect overall operation.

## Tips:
<input type="checkbox" disabled checked> If you want to use self-supervised feature extraction, select "y" for S2GAE in the arguments. If you want to compare the scenario without self-supervised feature extraction, select "n" for S2GAE in the arguments.

<input type="checkbox" disabled checked> This hyperparameter setting may only be applicable to the Houston 2018 dataset.

<input type="checkbox" disabled checked> A high-performance computing card may be necessary, such as the RTX 4090.

## link to the paper
https://ieeexplore.ieee.org/document/10945458

## Authors
Zhenghao Hu is currently pursuing the B.Sc.Eng. degree in Optoelectronic Information Science and 
Engineering at the School of Physics and Optoelectronic Engineering, Nanjing University of Information 
Science and Technology, Nanjing, China. His research focuses on hyperspectral image processing, 
computer vision and machine learning.

Advisor: Bing Tu (Professor and PhD Supervisor at Nanjing University of Information Science & Technology)



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
