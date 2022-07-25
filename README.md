# Multiface Dataset
Our dataset consists of high quality recordings of the faces of 13 identities, each captured in a multi-view capture stage performing various facial expressions. An average of 12,200 (v1 scripts) to 23,000 (v2 scripts) frames per subject with capture rate at 30 fps. Each frame includes roughly 40 (v1) to 160 (v2) different camera views under uniform illumination, yielding a total dataset size of 65TB. We provide the raw captured images from each camera view at a resolution of 2048 × 1334 pixels, tracked meshes including headposes, unwrapped textures at 1024 × 1024 pixels, metadata including intrinsic and extrinsic camera calibrations, and audio. This repository hosts the code of downloading the dataset and building a Codec Avatar using a [deep appearance model](https://arxiv.org/pdf/1808.00362.pdf). To learn more about how the dataset is captured and how different model architectures can influence performance, you may refer to our [Technical Report](https://arxiv.org/abs/2207.11243). 

<p align="center">
<img src="https://github.com/facebookresearch/multiface/blob/main/images/gif.gif?raw=true" width="500" height="450" />
</p>


## Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Works Using this Dataset](#works-using-this-dataset)
5. [Contributors](#contributors)
6. [Citation](#citation)
7. [License](#license)

# Features
* Comprehensive capture of a wide range of [facial expressions](./documentation/EXPRESSIONS.md)
* [High-quality tracked mesh](https://github.com/facebookresearch/multiface/blob/main/images/mesh.png?raw=true) for each frame
* High-resolution (2k), [multi-view captured images](./documentation/CAMERA_VIEW.md) 
* [6 assets](./documentation/DATASET_ASSET.md) are provided: raw images, unwrapped textures, tracked meshes, headpose, audio and metadata.

# Installation
### Quick Data Exploration
To download our data, first clone this repository and install dependencies
```
git clone https://github.com/facebookresearch/multiface
cd multiface
pip3 install -r requirements.txt
```

Since the full dataset takes terabytes of storage, one may wish to download partially. If you want to view the example assets, you may download the mini-dataset (< 1 GB)
```
python3 download_dataset.py --dest "/path/to/mini_dataset/" --download_config "./mini_download_config.json"
```

The `download_config` argument points to the configuration file specifying assets to be downloaded, options include:
| Variable        | Type          | Default  |
| ------------- |:-------------:| -----|
| entity     | list of string | All the entity will be downloaded |
| image      | boolean        | Raw images of enities selected will be downloaded|
| mesh       | boolean        | Tracked mesh of enities selected will be downloaded |
| texture    | boolean        | Unwrapped texture of enities selected will be downloaded |
| metadata   | boolean        | Metadata of enities selected will be downloaded |
| audio      | boolean        | Audio of enities selected will be downloaded |
| expression  |list of string | All the facial expression (contains both v1 and v2 scripts) will be downloaded  |

The configuration to download all assets can be found at `download_config.json`. 
### Full Installation
To run training and render the 3D faces, please refer to our full [installation guide](./documentation/INSTALLATION.md).

# Quick Start
To learn more on selecting model architecture, camera split and expression split for training and testing set, please refer to [quick start](./documentation/QUICK_START.md).

# Works Using this Dataset

[Deep Appearance Models For Face Rendering](https://arxiv.org/pdf/1808.00362.pdf) | [Learning Compositional Radiance Fields of Dynamic Human Heads](https://arxiv.org/pdf/2012.09955.pdf)  | [Pixel Codec Avatars](https://arxiv.org/pdf/2104.04638.pdf) 
:-------------------------:|:-------------------------: | :-------------------------:
<img src="https://github.com/facebookresearch/multiface/blob/main/images/dvae.png?raw=true" data-canonical-src="https://github.com/facebookresearch/multiface/blob/main/images/dvae.png?raw=true"   />|![](https://github.com/facebookresearch/multiface/blob/main/images/rd.png?raw=true)|![](https://github.com/facebookresearch/multiface/blob/main/images/pixel.png?raw=true)
[MeshTalk: 3D Face Animation from Speech using Cross-Modality Disentanglement](https://arxiv.org/pdf/2104.08223.pdf) | [Deep Incremental Learning for Efficient High-Fidelity Face Tracking](https://dl.acm.org/doi/pdf/10.1145/3272127.3275101) | [Modeling Facial Geometry using Compositional VAEs](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bagautdinov_Modeling_Facial_Geometry_CVPR_2018_paper.pdf)
![](https://github.com/facebookresearch/multiface/blob/main/images/talk.png?raw=true)|![](https://github.com/facebookresearch/multiface/blob/main/images/face.png?raw=true)|![](https://github.com/facebookresearch/multiface/blob/main/images/fvae.png?raw=true)
[Strand-accurate Multi-view Hair Capture](https://openaccess.thecvf.com/content_CVPR_2019/papers/Nam_Strand-Accurate_Multi-View_Hair_Capture_CVPR_2019_paper.pdf) | [Mixture of Volumetric Primitives for Efficient Neural Rendering](https://arxiv.org/pdf/2103.01954.pdf) [[Code Available]](https://github.com/facebookresearch/mvp)| [Human Hair Inverse Rendering using Multi-View Photometric data](https://cseweb.ucsd.edu/~ravir/hairinverse.pdf)
![](https://github.com/facebookresearch/multiface/blob/main/images/hair1.png?raw=true)|![](https://github.com/facebookresearch/multiface/blob/main/images/mvp_long.png?raw=true)|![](https://github.com/facebookresearch/multiface/blob/main/images/hair2.png?raw=true)
# Contributors
Thanks to all the [people](./documentation/CONTRIBUTOR.md) who has helped generate and maintain this dataset! 

# Citation
If you use any data from this dataset or any code released in this repository, please cite the technical report ([https://arxiv.org/abs/2207.11243](https://arxiv.org/abs/2207.11243))

```bibtex
@inproceedings{wuu2022multiface,
  title={Multiface: A Dataset for Neural Face Rendering},
  author = {Wuu, Cheng-hsin and Zheng, Ningyuan and Ardisson, Scott and Bali, Rohan and Belko, Danielle and Brockmeyer, Eric and Evans, Lucas and Godisart, Timothy and Ha, Hyowon and Hypes, Alexander and Koska, Taylor and Krenn, Steven and Lombardi, Stephen and Luo, Xiaomin and McPhail, Kevyn and Millerschoen, Laura and Perdoch, Michal and Pitts, Mark and Richard, Alexander and Saragih, Jason and Saragih, Junko and Shiratori, Takaaki and Simon, Tomas and Stewart, Matt and Trimble, Autumn and Weng, Xinshuo and Whitewolf, David and Wu, Chenglei and Yu, Shoou-I and Sheikh, Yaser},
  booktitle={arXiv},
  year={2022},
  doi = {10.48550/ARXIV.2207.11243},
  url = {https://arxiv.org/abs/2207.11243}
}
```

## License
Multiface is CC-BY-NC 4.0 licensed, as found in the [LICENSE file](https://github.com/facebookresearch/multiface/blob/main/LICENSE).

[[Terms of Use](https://opensource.facebook.com/legal/terms)]
[[Privacy Policy](https://opensource.facebook.com/legal/privacy)]
