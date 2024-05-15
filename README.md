
# Dynamic Adversarial Attacks on Autonomous Driving Systems

## Introduction

This repository contains the code and data for the paper "Dynamic Adversarial Attacks on Autonomous Driving Systems." The project focuses on generating dynamic adversarial attacks to test and improve the robustness of autonomous driving systems.

## Download Data

Download the data required for training and evaluation from the [Google Drive link](https://drive.google.com/drive/folders/1UiODhj44Wos0TJAiK1067lCwvnoJt0qu). The directory structure for each sign, should be as follows:

- `./images` : Includes clustered images to train the patch (clusters: 0, 1, and 2).
- `./coords` : Includes the four coordinates of the screen needed for applying the patch.
- `./Screen_data` : Includes data for the SIT-Net model.

## YOLOv5 Installation

To install YOLOv5, clone the official repository and install the requirements:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

## Installation

First, ensure you have Python installed on your system. Then, install the required Python packages using the \`requirements.txt\` file provided in this repository.

```bash
pip install -r requirements.txt
```

The \`requirements.txt\` file includes the following dependencies:

- fnmatch
- math
- random
- os
- sys
- time
- operator
- opencv-python
- gc
- numpy
- torch
- torchvision
- Pillow
- tensorboardX
- subprocess
- matplotlib
- tqdm

## Usage

To train patches, run:

```bash
python train.py
```

To train the SIT-Net model, refer to `Screen_Model.ipynb`.

## Contributing

If you would like to contribute to this project, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or issues, please contact the authors.
