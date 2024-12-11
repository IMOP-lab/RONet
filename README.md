# RONet:  Recurrent Optimization Network for RGB-T Salient Object Detection

![License](https://img.shields.io/badge/license-MIT-green) ![Python](https://img.shields.io/badge/python-3.8+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange)

[Project page](https://github.com/IMOP-lab/RONet) | [Our laboratory home page](https://github.com/IMOP-lab) 

by Yaoqi Sun, Zhaoyang Xu, Zhao Huang, Gaopeng Huang, Bin Wan, Haibing Yin, Jin Liu, Zhiwen Zheng, Xiaoshuai Zhang, and
Xingru Huang

Hangzhou Dianzi University IMOP-lab

## Project Overview

**RONet** is a novel encoder-decoder network specifically designed for **RGB-T salient object detection**. This model effectively leverages the complementary information from RGB and thermal images, achieving high detection accuracy even in low-light environments or scenes with complex backgrounds.

The key features of **RONet** include:

- **Recurrent Optimization Module (RO)**: Iteratively refines RGB and thermal features through cross-modal guidance.
- **Multi-Scale Semantic Module (MS)**: Utilizes a multi-branch structure to extract spatial and semantic information.
- **Detail Enhancement Module (DE)**: Employs multi-branch dilated convolutions to optimize low-level features and capture spatial details.

## Module Description

### Network Architecture

<div align=center>
  <img src="https://github.com/IMOP-lab/RONet/blob/main/figures/stru.png">
</div>

**Figure 1: Network Architecture.** The overall architecture of RONet, comprising the encoder, Recurrent Optimization Module (RO), Multi-Scale Semantic Module (MS), Detail Enhancement Module (DE), and the final decoding process.

**RONet** adopts a typical encoder-decoder network structure, consisting of the following components:

1. **Encoder**: Extracts multi-level features from RGB and thermal images using the Swin Transformer backbone.
2. **Recurrent Optimization Module (RO)**: Iteratively optimizes features through cross-modal guidance, improving the fusion quality of RGB and thermal data.
3. **Decoder**:
   - **Multi-Scale Semantic Module (MS)**: Extracts semantic and positional information from high-level features.
   - **Detail Enhancement Module (DE)**: Enhances low-level features and captures spatial boundary details.
4. **Output**: Generates high-quality saliency maps.

### Recurrent Optimization Module (RO)

<div align=center>
  <img src="https://github.com/IMOP-lab/RONet/blob/main/figures/RO.png">
</div>

**Figure 2: Recurrent Optimization Module (RO).** The detailed structure of the RO module, illustrating how RGB and thermal features are iteratively optimized using channel and spatial attention mechanisms for effective feature fusion.

The Recurrent Optimization Module refines RGB and thermal features through iterative cross-modal guidance, enhancing feature quality. The process includes:

1. Refining RGB features using a Channel Attention (CA) block.
2. Optimizing thermal features using the refined RGB features.
3. Enhancing RGB features further using the optimized thermal features.
4. Fusing the two modalities to generate high-quality multimodal features.

### Multi-Scale Semantic Module (MS)

<div align=center>
  <img src="https://github.com/IMOP-lab/RONet/blob/main/figures/MS.png">
</div>

**Figure 3: Multi-Scale Semantic Module (MS).** The structure of the MS module, incorporating three branches to extract spatial information, preserve semantic features, and enhance positional information.

The Multi-Scale Semantic Module is designed to extract semantic and positional information. It includes three branches:

1. Upsampling + Convolution + Downsampling branch: Extracts spatial information.
2. Multi-Layer Perceptron (MLP) branch: Preserves the original semantic information of features.
3. Downsampling + Convolution + Upsampling branch: Enhances positional information.

The outputs of these branches are fused and further refined using attention mechanisms.

### Detail Enhancement Module (DE)

<div align=center>
  <img src="https://github.com/IMOP-lab/RONet/blob/main/figures/DE.png">
</div>

**Figure 4: Detail Enhancement Module (DE).** The DE module design utilizes multi-branch convolutional layers with different dilation rates to progressively optimize low-level features and capture fine spatial details.

The Detail Enhancement Module focuses on refining the spatial and boundary details of the target. Its design includes:

1. Multi-branch convolutions with different dilation rates (1, 2, 4, 6).
2. Progressive feature refinement using Channel Attention (CA) for branch-wise optimization.
3. Fusion of all branch outputs to generate the final enhanced features.

------

## Usage Instructions and Details

### Installation

#### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.1+
- Additional dependencies: `torchvision`, `numpy`, `matplotlib`, etc.

#### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/IMOP-lab/RONet.git
   cd RONet
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Prepare datasets (e.g., **VT5000**, **VT1000**, **VT821**) and organize them as follows:

   ```bash
   datasets/
   ├── VT5000/
   │   ├── train/
   │   ├── test/
   ├── VT1000/
   ├── VT821/
   ```

### Workflow

#### Training

Train the model on the VT5000 dataset:

```bash
python train.py --dataset VT5000 --batch_size 16 --epochs 50
```

#### Testing

Evaluate the model on the VT1000 dataset:

```bash
python test.py --dataset VT1000 --weights checkpoints/RONet.pth
```

## Quantitative Comparison

**Table 1: Quantitative comparison of different methods for RGB-T salient object detection.** The table summarizes the performance of various methods in terms of parameters (#Param), computational complexity (FLOPs), frame rate (FPS), and evaluation metrics (MAE, Fβ, Sa, and Eξ) across three datasets (VT5000, VT1000, VT821). Bold values indicate the best performance for each metric.

| Methods     | #Param (M) ↓ | FLOPs (G) ↓ | FPS ↑ | VT5000 (M ↓) | VT5000 (Fβ ↑) | VT5000 (Sa ↑) | VT5000 (Eξ ↑) | VT1000 (M ↓) | VT1000 (Fβ ↑) | VT1000 (Sa ↑) | VT1000 (Eξ ↑) | VT821 (M ↓) | VT821 (Fβ ↑) | VT821 (Sa ↑) | VT821 (Eξ ↑) |
| ----------- | ------------ | ----------- | ----- | ------------ | ------------- | ------------- | ------------- | ------------ | ------------- | ------------- | ------------- | ----------- | ------------ | ------------ | ------------ |
| MIDD        | 52.4         | 216.7       | 42.2  | .046         | .788          | .856          | .893          | .029         | .870          | .907          | .935          | .045        | .803         | .871         | .897         |
| MMNet       | 64.1         | 42.5        | 38.5  | .044         | .777          | .863          | .888          | .027         | .859          | .914          | .931          | .040        | .792         | .874         | .893         |
| CSRNet      | 1.0          | 4.4         | 46.4  | .042         | .809          | .868          | .907          | .024         | .875          | .918          | .939          | .038        | .829         | .885         | .912         |
| ECFFNet     | -            | -           | -     | .038         | .803          | .875          | .911          | .022         | .873          | .924          | .947          | .035        | .807         | .877         | .907         |
| TAGFNet     | 36.2         | 115.1       | 39.5  | .036         | .826          | .884          | .916          | .021         | .888          | .926          | .951          | .035        | .821         | .881         | .909         |
| APNet       | 30.4         | 46.6        | 50.1  | .035         | .816          | .876          | .917          | .022         | .880          | .922          | .950          | .034        | .814         | .868         | .911         |
| CGFNet      | 69.9         | 347.8       | 52.3  | .035         | .851          | .883          | .924          | .024         | .901          | .921          | .953          | .036        | .842         | .879         | .915         |
| TNet        | 87.0         | 39.7        | 49.6  | .033         | .845          | .895          | .929          | .021         | .887          | .929          | .952          | .030        | .840         | .899         | .924         |
| MCFNet      | 70.8         | 40.8        | 34.2  | .033         | .846          | .888          | .928          | .019         | .900          | .932          | .960          | .029        | .842         | .891         | .925         |
| CCFENet     | 28.7         | 17.1        | 56.0  | .031         | .858          | .896          | .937          | .018         | .904          | .934          | .962          | .027        | .856         | .900         | .933         |
| CAVER       | 93.8         | 31.6        | 28.5  | .029         | .854          | .900          | .940          | .017         | .904          | .938          | .965          | .027        | .853         | .898         | .934         |
| SwinNet*    | 198.8        | 124.7       | 30.9  | .026         | .864          | .912          | .945          | .018         | .894          | .938          | .957          | .030        | .846         | .904         | .928         |
| WaveNet     | 30.2         | 26.7        | 7.7   | .026         | .863          | .912          | .944          | .015         | .905          | .945          | .967          | .025        | .854         | .912         | .935         |
| HRTransNet* | 58.9         | 17.3        | 14.9  | .025         | .869          | .913          | .948          | .017         | .898          | .938          | .956          | .026        | .851         | .906         | .933         |
| UidefNet*   | -            | -           | -     | .025         | .874          | .914          | .948          | .015         | .911          | .941          | .969          | .023        | .872         | .917         | .945         |
| SPNet*      | 110.0        | 67.8        | 14.1  | .024         | .885          | .915          | .952          | .015         | .918          | .942          | .972          | .023        | .874         | .914         | .943         |
| **Ours***   | 87.1         | 90.8        | 9.2   | **.021**     | **.886**      | **.923**      | **.955**      | **.014**     | **.913**      | **.946**      | **.967**      | **.025**    | **.856**     | **.914**     | **.931**     |

## Qualitative Comparison

Experimental results demonstrate that **RONet** significantly outperforms existing methods on multiple public RGB-T datasets, particularly excelling in accuracy and robustness under low-light or complex scenarios.

<div align=center>
  <img src="https://github.com/IMOP-lab/RONet/blob/main/figures/compare.png.png">
</div>

**Figure 5: Visual comparison of RGB-T salient object detection results.** Each row shows the input RGB image (a), thermal image (b), and saliency maps generated by various methods (c-t). The proposed method produces more accurate saliency maps under challenging conditions, effectively capturing target shapes and boundaries.

## License

This project is licensed under the MIT License. For details, please refer to[The MIT License – Open Source Initiative](https://opensource.org/license/MIT)。

## Contact Information

For inquiries or collaborations, please contact:

- Yaoqi Sun (syq@hdu.edu.cn)
- Gaopeng Huang (gaopeng.huang@hdu.edu.cn)
