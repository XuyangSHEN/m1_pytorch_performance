# Mac M1 Native PyTorch

* [GitHub](https://github.com/XuyangSHEN/m1_pytorch_performance) [Notion](https://www.notion.so/xuyangshen/Mac-M1-Native-PyTorch-96c73c4098e143aab607cce0f33d5922 )

------



## Performance Experiment Results

| CIFAR 10 ResNet50             | M1 Pro 8 Core | M1 Pro 14 GCore | NV 3070 Laptop | NV A100-40g x1 | NV A100-40g x8 | NV A100-40g x16 |
| ----------------------------- | :-----------: | :-------------: | :------------: | :------------: | :------------: | :-------------: |
| Training Time (seconds/epoch) |    240.17     |  104.70 (2.3x)  |      7.82      |      4.50      |      2.22      |      2.01       |
| Test Acc (% after 20 epochs)  |     45.88     |      43.38      |     56.11      |     43.40      |     31.02      |      31.02      |

- **Environment**
  - Code: [Train a ResNet-50 classifier on CIFAR10](CIFAR10_ResNet/code_template.ipynb)
  - M1 Pro: PyTorch 1.12, Python 3.10
  - 3070 Laptop: cuda 11.6, PyTorch 1.11, Python 3.10
  - A100-40g: cuda 11.2, PyTorch1.11, Python 3.10, Distributed Data Parallel (DDP)
  - All results are averaged from 3 runs.



## Install Conda
- Check my video: https://www.bilibili.com/video/BV1cR4y1w7CF/

- Miniconda: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

  [Miniconda3 macOS Apple M1 64-bit pkg](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.pkg)

- Miniforge: https://github.com/conda-forge/miniforge

  [Miniforge3-MacOSX-arm64](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh)



### Verify conda platform is ARM

```bash
$ conda info

...
platform : osx-arm64
...
```



## Create new environment and activate

```bash
$ conda create -n torch1.12 python=3.10 #name=torch1.12 with python3.10 
$ codna activate torch1.12 
```



## Install PyTorch1.12

- [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) (Preview - Mac - Pip)

```bash
$ conda install pytorch torchvision -c pytorch-nightly
```

- Install other packages

```bash
$ pip install notebook
```

