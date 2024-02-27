# Source code for: Scaling on-chip photonic neural processors using arbitrarily programmable wave propagation

## Overview
We propose and demonstrate a device, a *2D-programmable waveguide* whose refractive index as a function of space, $n(x,z)$, can be rapidly reprogrammed, allowing arbitrary control over the wave propagation in the device [1]. We trained the device with a hybrid in-situ, in-silico backpropagation algorithm [2] to perform neural network inference. In particular, we achieved 96\% accuracy on vowel classification and 86\% accuracy on $7 \times 7$-pixel MNIST handwritten-digit classification, with no trained digital-electronic pre- or post-processing.

In this repository, we provide the source code used to generate the results in our paper, written in Python and utilizing the PyTorch library for machine learning tasks. Designed for execution in a Jupyter notebook environment, this repository includes source code for simulations and experiments. Primarily intended for internal use, the code lacks detailed documentation, and the experimental portion presupposes connectivity to specific experimental devices, such as spatial-light modulators and cameras. The simulation code is self-contained and should run on system. The repository's primary contents are the source code alongside several usage examples. Data and code used to generate the figures of the paper will shortly be made accessible in the Zenodo repository [3]. For users interested in using this code or with inquiries, please feel free to contact us via the email addresses listed in the paper.

## Contents
- `/tdwg` - Source code used for this work includes:
    - `/datasets`
        - `mnist_dataset.py` - Script for downloading and preprocessing the MNIST dataset.
        - `vowel_dataset.csv` - Dataset used for vowel classification.
        - `vowels_dataset.py` - Script for loading the vowel dataset.
    - `/lib` - Key files include
        - `simulation.py` - Script for simulating the 2D-programmable waveguide.
        - `exp_sim_converter.py` - Script for translating simulation and experimental parameters and outputs between the two.
        - `tdwgnet.py` - Script defining a PyTorch nn.Module that represents the 2D-programmable waveguide, with built-in functionality for training it with physics-aware training [2].
        - Key scripts for interfacing with experimental devices include `line_camera.py`, `PCIe_beamshaper.py`, `DMD.py`.
- `/examples` - Jupyter notebooks that demonstrate the use of the source code, including:
    - `calibration of the physics-based model.ipynb` - Notebook for calibrating the physics-based model using the beamshaper and writing specific patterns to the DMD.
    - `vowel classification with tdwg.ipynb` - Notebook for training the 2D-programmable waveguide to classify vowels, including code for data-driven refinement of the physics-based model.
    - `MNIST classification with tdwg.ipynb` - Notebook for training the 2D-programmable waveguide to classify MNIST digits.


## References
[1]: T. Onodera*, M.M. Stein*, B.A. Ash, M.M. Sohoni, M. Bosch, R. Yanagimoto, M. Jankowski, T.P. McKenna, T. Wang, G. Shvets, M.R. Shcherbakov, L.G. Wright, P.L. McMahon, "Scaling On-chip Photonic Neural Processors Using Arbitrarily Programmable Wave Propagation", ArXiv: 2402.17750 (2024). https://arxiv.org/abs/2402.17750

[2]: L.G. Wright*, T. Onodera*, M.M. Stein, T. Wang, D.T. Schachter, Z. Hu & P.L. McMahon, "Deep physical neural networks trained with backpropagation", _Nature_ **601**, 549–555 (2022).

[3]: Data and Code for "Scaling On-chip Photonic Neural Processors Using Arbitrarily Programmable Wave Propagation", Zenodo. In preparation.

## License

The code in this repository is released under the following license:

[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

A copy of this license is given in this repository as [license.txt](https://github.com/mcmahon-lab/2D-programmable-waveguide/blob/master/license.txt).