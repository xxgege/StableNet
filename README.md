# StableNet
StableNet is a deep stable learning method for out-of-distribution generalization.

This is the official repo for CVPR21 paper "Deep Stable Learning for Out-Of-Distribution Generalization" and the arXiv version can be found at [https://arxiv.org/abs/2104.07876](https://arxiv.org/abs/2104.07876).

Please note that some hyper-parameters(such as lrbl, epochb, lambdap) may affect the performance , which can vary among different tasks/environments/software/hardware/random seeds, and thus careful tunning is required. Similar to other DG repositories, direct migration may lead to different results as ours. We are sorry for this and trying to address this problem in the following work. 

## Introduction
Approaches based on deep neural networks have achieved striking performance when testing data and training data share similar distribution, but can significantly fail otherwise. Therefore, eliminating the impact of distribution shifts between training and testing data is crucial for building performance-promising deep models. Conventional methods assume either the known heterogeneity of training data (e.g. domain labels) or the approximately equal capacities of different domains. In this paper, we consider a more challenging case where neither of the above assumptions holds. We propose to address this problem by removing the dependencies between features via learning weights for training samples, which helps deep models get rid of spurious correlations and, in turn, concentrate more on the true connection between discriminative features and labels. Extensive experiments clearly demonstrate the effectiveness of our method on multiple distribution generalization benchmarks compared with state-of-the-art counterparts. Through extensive experiments on distribution generalization benchmarks including PACS, VLCS, MNIST-M, and NICO, we show the effectiveness of our method compared with state-of-the-art counterparts.

## Installation
### Requirements
- Linux with Python >= 3.6
- [PyTorch](https://pytorch.org/) >= 1.1.0
- torchvision >= 0.3.0
- tensorboard >= 1.14.0

## Quick Start
### Train StableNet
```bash
python main_stablenet.py --gpu 0
```
You can see more options from
```bash
python main_stablenet.py -h
```
Result files will be saved in `results/`.


## Performance and trained models


| setting | dataset | source domain | target domain | network | dataset split | accuracy | trained model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unbalanced(5:1:1) | PACS | A,C,S | photo | ResNet18 | [split file](https://drive.google.com/file/d/1jnwEFNp5erOTnwAWH5EJW4f4JB4ymSM0/view?usp=sharing) | 94.864 | [model file](https://drive.google.com/file/d/1QluSMIBwknEI7Hnw_y2xh6eN0u9YUW0d/view?usp=sharing) |
| unbalanced(5:1:1) | PACS | C,S,P | art_painting | ResNet18 | [split file](https://drive.google.com/file/d/1CKyn4dYfoC5xeac2vdHNJy_jjAv-rNUP/view?usp=sharing) | 80.344 | [model file](https://drive.google.com/file/d/1OVQNUQgywKoUrZdFL3LyRRL6UG_f02a4/view?usp=sharing) |
| unbalanced(5:1:1) | PACS | A,S,P | cartoon | ResNet18 | [split file](https://drive.google.com/file/d/1ibXf7d8Tkwrxlw1PRn1_v-89ROJ2ba47/view?usp=sharing) | 74.249 | [model file](https://drive.google.com/file/d/1Q7-mHf2cGmKtAsmpDGHTrG0Cd81hC_9Q/view?usp=sharing) |
| unbalanced(5:1:1) | PACS | A,C,P | sketch | ResNet18 | [split file](https://drive.google.com/file/d/1oeJfbzAk2rgIbma93uZSOUPoGnYBzQMa/view?usp=sharing) | 71.046 | [model file](https://drive.google.com/file/d/1jhuXTMJsG429MuK_X3fBLHHn1OPt8tCn/view?usp=sharing) |
| unbalanced(5:1:1) | VLCS | L,P,S | caltech | ResNet18 | [split file](https://drive.google.com/file/d/1o7xtLdNDmn1JguJCoAqBzIqw_s8-cJEC/view?usp=sharing) | 88.776 | [model file](https://drive.google.com/file/d/1R7qKspcMfT3WkagBz2cZ-X8QdTmMm_6S/view?usp=sharing) |
| unbalanced(5:1:1) | VLCS | C,P,S | labelme | ResNet18 | [split file](https://drive.google.com/file/d/1y8BNtGXQP3k1SsrJFV6V1ErEX-fJlqit/view?usp=sharing) | 63.243 | [model file](https://drive.google.com/file/d/1F8oZ5GsfbPO2cfNm8el8eTfJzw70494u/view?usp=sharing) |
| unbalanced(5:1:1) | VLCS | C,L,S | pascal | ResNet18 | [split file](https://drive.google.com/file/d/1pqut_fTPHAPa_s3IrX1booekSkww8VRe/view?usp=sharing) | 66.383 | [model file](https://drive.google.com/file/d/11bZqloemdQgnudcHMpbvSDgxFlsmE5Ac/view?usp=sharing) |
| unbalanced(5:1:1) | VLCS | C,L,P | sun | ResNet18 | [split file](https://drive.google.com/file/d/15__uuvmjM7-UQmily2CL2qZtd3qPcCIP/view?usp=sharing) | 55.459 | [model file](https://drive.google.com/file/d/1bWLX3CGcPS5d3LdVbPMp9oMYwXNrpd5-/view?usp=sharing) |
| flexible(5:1:1) | PACS | - | - | ResNet18 | [split file](https://drive.google.com/file/d/1QAa998LPRkHpBeNLpQqfk5WEGLCbc7Ma/view?usp=sharing) | 45.964 | [model file](https://drive.google.com/file/d/19q1cmTRGtU0xkk7lh9Pd_AUb8sbR00y1/view?usp=sharing) |
| flexible(5:1:1) | VLCS | - | - | ResNet18 | [split file](https://drive.google.com/file/d/1PnOOG8cYwKqdlSp33P4BvFIcmyWLGQvI/view?usp=sharing) | 81.157 | [model file](https://drive.google.com/file/d/1df7o-T98v7grNY4roT5E7KI4bQvhqRef/view?usp=sharing) |

 

## Citing StableNet
If you find this repo useful for your research, please consider citing the paper.
```
@inproceedings{zhang2021deep,
  title={Deep Stable Learning for Out-Of-Distribution Generalization},
  author={Zhang, Xingxuan and Cui, Peng and Xu, Renzhe and Zhou, Linjun and He, Yue and Shen, Zheyan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5372--5382},
  year={2021}
}
```
