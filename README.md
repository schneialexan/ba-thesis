<div align="center">
<h1 align="center">
<img src="https://gitarrehamburg.de/wp-content/uploads/2017/10/Icon-Abschlussarbeit-1.png" width="300" />
</h1>
<h1>Bachelors Thesis: <br> Deep Learning Based Fluid Simulation</h1>
<h2>Consideration of different model architectures</h2>
<h3>Developed with the software and tools below.</h3>

<p align="center">
<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style&logo=tqdm&logoColor=black" alt="tqdm" />
<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style&logo=TensorFlow&logoColor=white" alt="TensorFlow" />
<img src="https://img.shields.io/badge/Dask-FDA061.svg?style&logo=Dask&logoColor=black" alt="Dask" />
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style&logo=scikit-learn&logoColor=white" alt="scikitlearn" />
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style&logo=Jupyter&logoColor=white" alt="Jupyter" />
<img src="https://img.shields.io/badge/Keras-D00000.svg?style&logo=Keras&logoColor=white" alt="Keras" />
<img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style&logo=Matplotlib&logoColor=black" alt="Matplotlib" />
<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style&logo=SciPy&logoColor=white" alt="SciPy" />

<img src="https://img.shields.io/badge/SymPy-3B5526.svg?style&logo=SymPy&logoColor=white" alt="SymPy" />
<img src="https://img.shields.io/badge/Python-3776AB.svg?style&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/pandas-150458.svg?style&logo=pandas&logoColor=white" alt="pandas" />
<img src="https://img.shields.io/badge/NumPy-013243.svg?style&logo=NumPy&logoColor=white" alt="NumPy" />
<img src="https://img.shields.io/badge/ONNX-005CED.svg?style&logo=ONNX&logoColor=white" alt="ONNX" />
<img src="https://img.shields.io/badge/Markdown-000000.svg?style&logo=Markdown&logoColor=white" alt="Markdown" />
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style&logo=PyTorch&logoColor=white" alt="PyTorch" />
</div>

---

## 📖 Table of Contents
- [📖 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [📂 Repository Structure](#-repository-structure)
- [⚙️ Modules](#modules)
- [🚀 Getting Started](#-getting-started)
- [🛣 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---


## 📍 Overview

This thesis investigates the application of image-based Deep Learning models in predicting fluid dynamics simulations, with an emphasis on the flow behavior governed by the Navier-Stokes equations (in this case the Lid-Driven Cavity problem). The primary research question explores the effectiveness of DL models in this context, while sub-questions focus on the performance of these models and the associated limitations and challenges.

---


## 📂 Repository Structure

```sh
└── /
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    ├── conv_lstm/
    │   ├── v01/
    │   ├── v02/
    │   ├── v03/
    │   ├── v04/
    │   └── v05/
    ├── data/
    │   └── 100.0/ - 4000.0/
    ├── data_gen_and_prep/
    │   ├── lid-driven-image_gen.py
    │   ├── lid-driven.py
    │   ├── v01_tensorflow.ipynb
    │   ├── v02_PyTorch.ipynb
    │   ├── v03_image_prep.ipynb
    │   └── v04_plots_for_paper.ipynb
    ├── flowtransformer/
    │   ├── v01/
    │   ├── v02/
    │   ├── v03/
    │   ├── v04/
    │   ├── v05/
    │   ├── v06/
    │   ├── v07/
    │   └── v08/
    ├── requirements.txt
    └── u_net/
        ├── v01/
        ├── v02/
        ├── v03/
        ├── v04/
        ├── v05/
        ├── v06/
        ├── v07/
        └── v08/
```


---

## ⚙️ Modules

| File | Summary |
| --- | --- |
| [.gitignore](.gitignore) | The finished models couldn't be uploaded since they are too big |
| [LICENSE](LICENSE) | MIT licence |
| [README.md](README.md) | This file. Hi! |
| [requirements.txt](requirements.txt) | Every single module with its version |


---

## 🚀 Getting Started

***Dependencies***

Please ensure you have the following dependencies installed on your system (this is what was used):

`- ℹ️ Python 3.12`

`- ℹ️ CUDA 12.4`

`- ℹ️ Python Libraries: visit requirements.txt`

### 🔧 Installation

1. Clone the  repository:
```sh
git clone git@github.com:schneialexan/ba-thesis.git
```

2. Change to the project directory:
```sh
cd ba-thesis
```

3. Install the (mini-)conda environment:
```sh
conda create -n ba python=3.12
conda activate ba
conda install pip
```

4. Install the dependencies:
```sh
pip install -r requirements.txt
```


## 🛣 Roadmap

> - [X] `ℹ️  Task 1: Implement U-Net (Thuerey et al. (2020))`
> - [X] `ℹ️  Task 2: Implement ConvLSTM (Costa Rocha et al. (2023))`
> - [X] `ℹ️  Task 3: Implement Flow Transformer architecture`
> - [X] `ℹ️  Task 4: Ghia et al. comparisons`
> - [ ] `ℹ️  Task 5: Optimizing ConvLSTM`


---

## 🤝 Contributing

Contributions are always welcome! Please follow these steps:
1. Fork the project repository. This creates a copy of the project on your account that you can modify without affecting the original project.
2. Clone the forked repository to your local machine using a Git client like Git or GitHub Desktop.
3. Create a new branch with a descriptive name (e.g., `new-feature-branch` or `bugfix-issue-123`).
```sh
git checkout -b new-feature-branch
```
4. Make changes to the project's codebase.
5. Commit your changes to your local branch with a clear commit message that explains the changes you've made.
```sh
git commit -m 'Implemented new feature.'
```
6. Push your changes to your forked repository on GitHub using the following command
```sh
git push origin new-feature-branch
```
7. Create a new pull request to the original project repository. In the pull request, describe the changes you've made and why they're necessary.
The project maintainers will review your changes and provide feedback or merge them into the main branch.

---

## 📄 License

This project is licensed under the `MIT` License. See the [MIT](LICENSE) file for additional info.

[↑ Return](#Top)

---
