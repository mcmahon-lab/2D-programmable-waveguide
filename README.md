This repository contains demo code associated with our paper *Scaling on-chip photonic neural processors using arbitrarily programmable wave propagation*.
It enables straightforward exploration of 2D-programmable waveguides, i.e. waveguides whose refractive index distribution $n(x,z)$ can be arbitrarily programmed in space. 
The code is intentionally kept minimal, focusing on the essential elements of such devices: 
- Solving a unidirectional wave equation
- Using either user-defined or inverse-designed refractive index distributions
- Implemented in *PyTorch* for GPU- and auto-differentiation-support
- Integration with *Physics-Aware-Training* to emulate experiments which are not differentiable

However, it is not a full reference implementation of the code we used in experiment, which may instead be found in the associated [Zenodo](https://doi.org/10.5281/zenodo.10775721).
<p align="center">
<img src="https://github.com/user-attachments/assets/5a1bd570-0beb-4959-837f-6a1d0b965d23" width="800">
</p>

# How to get started

Each of the following bulletpoints can be clicked for further information:

<details> <summary><b>Installing dependencies</b></summary>
  
---
Please ensure that the correct versions of all packages specified in 
[environment.yml](https://github.com/ms3452/2D-waveguide-demo-code/blob/main/environment.yml) are installed. If using anaconda, the easiest way to do so is to clone the repository, open an anaconda prompt in the repository folder, and execute:
  ```
  conda env create -f environment.yml
  conda activate 2Dwg
  ```
This creates and activates an environment called `2Dwg`.
To run the code, launch Jupyter Lab by executing
```
jupyter lab
```
in the anaconda prompt.

By default, the provided environment.yml installs a CPU-only version of PyTorch. To enable GPU acceleration, first create and activate the environment as described above, then additionally run
```
conda install pytorch-cuda=11.8 -c nvidia -c pytorch
```
This upgrades the environment to use GPU builds of PyTorch and TorchVision. CPU users can ignore this step.

---

</details> <details> <summary><b>Simplest simulation</b> of a refractive-index pattern</summary>

---
[Example 1](https://github.com/ms3452/2D-waveguide-demo-code/blob/main/Example%201%20Y-splitter.ipynb) provides code that manually defines the refractive-index distribution of a Y-splitter and simulates beam propagation through it. Parameters of this simulation are similar to the experimental results presented in Fig. 2 of our paper.

---

</details> <details> <summary><b>Inverse design</b> of a refractive-index pattern</summary>

---
[Example 2](https://github.com/ms3452/2D-waveguide-demo-code/blob/main/Example%202%20Inverse%20design.ipynb) contains a minimal inverse-design example that automatically generates a refractive-index distribution for converting Gaussian beams into Hermite–Gauss modes. Inverse-design is performed via the auto-differentiable simulation of the programmable waveguide with *PyTorch*.

---

</details> <details> <summary><b>Machine learning demo</b> with MNIST</summary>

---
[Example 3](https://github.com/ms3452/2D-waveguide-demo-code/blob/main/Example%203%20MNIST%20classification.ipynb) demonstrates MNIST image classification using a 2D-programmable waveguide. Parameters of this simulation are similar to experimental results presented in Fig. 4 of our paper.

---

</details> <details> <summary><b>High-dimensional MVMs</b> in multimode waveguides</summary>

---
[Example 5](https://github.com/ms3452/2D-waveguide-demo-code/blob/main/Example%205%20Matrix-vector-multiplication%20in%20multimode%20waveguide.ipynb) shows how to compute a refractive-index distribution that, embedded in a multimode waveguide, performs a desired 100×100-dimensional unitary transformation.
[Example 4](https://github.com/ms3452/2D-waveguide-demo-code/blob/main/Example%204%20Mode%20conversion%20in%20multimode%20waveguide.ipynb) contains simpler code that introduces a step-index multimode waveguide as background refractive index and demonstrates mode conversion using a manually defined refractive-index distribution.
These notebooks can be readily adapted to replicate the results presented in Fig. 5 of our paper.

---

</details> <details> <summary><b>Physics-aware training</b> with mismatched forward/backward passes</summary>

---
[Example 6](https://github.com/ms3452/2D-waveguide-demo-code/blob/main/Example%206%20Mismatched%20forward-backward%20pass.ipynb) is a minimal inverse-design notebook using a mismatched forward and backward pass, similar to the approach used in our optical experiments with the 2D-programmable waveguide.
This notebook can used to *design* an artificial simulation-reality gap and explore whether the training algorithm can succesfully train the programmable waveguide despite the discrepancy.

---

</details>


# How to cite this code

If you use this code in your research, please consider citing the following paper:

> Onodera, T., Stein, M. M., et al (2024). Scaling on-chip photonic neural processors using arbitrarily programmable wave propagation. *arXiv:2402.17750* https://arxiv.org/abs/2402.17750v1.

# License

This repository is licensed under:

[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

A copy of the license is included in this repository as [license.txt](https://github.com/ms3452/2D-waveguide-demo-code/blob/main/license.txt).
