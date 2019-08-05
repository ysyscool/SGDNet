# SGDNet
SGDNet: An End-to-End Saliency-Guided Deep Neural Network for  No-Reference Image Quality Assessment

This repository contains the reference code for our ACM MM 2019 paper. The pdf is coming soon.

If you use any part of our code, or SGDNet is useful for your research, please consider citing:
```
@inproceedings{yang2019sgdnet,
  title={SGDNet: An End-to-End Saliency-Guided Deep Neural Network for  No-Reference Image Quality Assessment},
  author={Yang, Sheng and Jiang, Qiuping and Lin, Weisi and Wang, Yongtao},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  year={2019},
  organization={ACM}
}
```

## Requirements
* Python 2.7
* Keras 2.1.2
* Tensorflow-gpu 1.3.0

## Getting Started
### Installation
- Clone this repo:
```bash
git clone https://github.com/ysyscool/SGDNet
cd SGDNet
mkdir ../checkpoint/
```
- Download weights from [Google Drive](https://drive.google.com/file/d/1TvCBvZypSsiKUOHeBRrF-5L8SaA8dwLH/view?usp=sharing).
Put the weights into 
```bash
cd ../checkpoint/
```

### Train/Test
1. Download the IQA datasets. Their saliency maps, used in our experiments, can be downloaded in this [link](https://drive.google.com/drive/folders/1EoXpuEBAUPNKPh-17u8WRyHCCg9PKKgT?usp=sharing).
2. Modify the paths in config.yaml
And then using the following command to train the model (use knoiq10k and DINet as example)
```bash
 CUDA_VISIBLE_DEVICES=0 python main.py  --database=Koniq10k --lr=1e-4 --batch_size=19 --out2dim=1024  --saliency=output --phase=train
```

For testing, modify the variables of arg (in line 276) as the trained checkpoint name in the main.py.
And then using the following command to test the model
```bash
CUDA_VISIBLE_DEVICES=0 python main.py  --database=Koniq10k --out2dim=1024 --saliency=output --phase=test
```

## Acknowledgments
Code and data prepration largely benefits from [CNNIQAplusplus](https://github.com/lidq92/CNNIQAplusplus) by Dingquan Li. 
