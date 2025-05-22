# ToyDiskGeoFitter

**ToyDiskGeoFitter** is a Python package that fits the geometric parameters (center, inclination, position angle) of astronomical disks in FITS images using MCMC. It also helps generate radial profiles from the fitted geometry.

[![License: GPL](https://img.shields.io/badge/License-GPL-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
<!-- Add PyPI badge if/when released -->

## Core Features

*   **TOML Configuration:** Easy setup for fitting parameters, priors, and MCMC settings.
*   **MCMC Fitting:** Uses `emcee` for Bayesian parameter estimation based on minimizing variance in elliptical annuli.
*   **Plotting:** Generates MCMC chain plots, corner plots, and radial surface brightness profiles.
*   **FITS Support:** Reads FITS images and can use header info (e.g., pixel scale).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dreamjade/Toy_Diskgeofitter.git
    cd toydiskgeofitter
    ```
2.  **Install:**
    ```bash
    pip install .
    ```
    For development, use `pip install -e .`

## Quick Start

The most straightforward way to see the workflow is to adapt the Jupyter Notebook:[`examples/basic_disk_fitting_example.ipynb`](./examples/basic_disk_fitting_example.ipynb)
This notebook guides you through loading the config, data, running the MCMC, and plotting results.

## Dependencies

*   `numpy`, `scipy`, `matplotlib`, `astropy`, `emcee`, `corner`, `toml`, `tqdm`

## To Do / Future Work

*   **Brightness Unit Conversion:** Convert image units to photon counts for more physical noise modeling.
*   **Deconvolved Image Errors:** Address error characteristics in deconvolved images (e.g., CLEAN artifacts, correlated noise).
*   **More Fitting Methods:**
    *   SNR-based likelihoods.
    *   Direct model fitting (MLE, chi-squared).
*   Enhanced multi-image fitting.
*   Broader sampler support (e.g., `dynesty`).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

GPL License. See `LICENSE` file.
