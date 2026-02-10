![harp-diag](images/harp_diagram.jpg)

# HARP: Human-Aware Routing Policies for Multi-Stage Decision Pipelines

High-stakes institutional decisions, such as insurance claim review, increasingly 
combine machine learning with multi-stage human review. Cases are either resolved by a 
model or escalated through tiers of reviewers with different costs and expertise. Standard 
selective prediction treats deferral as a binary choice and does not capture these hierarchical 
pipelines. We introduce Human-Aware Routing Policies (HARP), a framework that treats routing 
as a learned policy over stages (model vs.\ reviewer tiers) and minimizes expected system loss 
under human cost. We formalize single and multi-stage deferral and train a stochastic routing 
policy with Gumbel-softmax on a loss that combines prediction errors and reviewer costs. We compare 
HARP to a confidence-threshold naive policy, multi-stage calibrated selective classification (CSC),
and an oracle that knows ground truth on Medicare-style data with a two-tier reviewer hierarchy and
costs set by marginal risk reduction. Overall, HARP improves the accuracy-cost tradeoff over the 
naive policy and matches or outperforms CSC while learning routing end-to-end.

## Table of Contents

1. [About The Project](#about-the-project)
   * [Built With](#built-with)
2. [Getting Started](#getting-started)
   * [Prerequisites](#prerequisites)
   * [Installation](#installation)
3. [Usage](#usage)
4. [Roadmap](#roadmap)
5. [Contributing](#contributing)
6. [License](#license)
7. [Contact](#contact)
8. [Acknowledgments](#acknowledgments)

## About The Project

Algorithmic code implementation of "Human-Aware Routing Policies for Multi-Stage Decision Pipelines."

Read the [paper](https://drive.google.com/file/d/1UoqddL2oJkUzHDF3VPKgC1EpD7ubQGor/view?usp=sharing) for technical details and a full analysis of the results.

[(back to top)](#table-of-contents)

## Getting Started

### Prerequisites

* Conda is required, see this link for more information of installation for your system: [Conda Installation Guide](https://docs.conda.io/en/latest/miniconda.html)

### Installation

Please run the following commands:

```bash
conda env create -f environment.yml

conda activate harp
```

Or alternatively, for CUDA support:
```bash
conda env create -f environment-cuda.yml

conda activate harp
```

Once installed, you can verify the installation by running:

```bash
python -c "import torch; import pandas; import numpy; print('Installation successful!')"
```

[(back to top)](#table-of-contents)

## Usage

### Run the data pipeline

Please run from root:

```bash
chmod +x harp/scripts/data.sh

harp/scripts/data.sh harp/config/data_config.yaml
```

### Train models

To train individual models:

```bash
chmod +x harp/scripts/train.sh

# Train F-network
harp/scripts/train.sh harp/config/f_train_config.yaml

# Train CSC G-network
harp/scripts/train.sh harp/config/csc_g_train_config.yaml

# Train HARP Pi-network
harp/scripts/train.sh harp/config/harp_pi_train_config.yaml
```

### Run all experiments

To run the complete pipeline (data processing, training, and evaluation):

```bash
chmod +x harp/scripts/run_all_experiments.sh

harp/scripts/run_all_experiments.sh
```

### Evaluate models

To evaluate trained models:

```bash
chmod +x harp/scripts/evaluate.sh

harp/scripts/evaluate.sh harp/config/evaluate_config.yaml
```

[(back to top)](#table-of-contents)

## Built With

* [Python](https://www.python.org/) - Programming language
* [PyTorch](https://pytorch.org/) - Deep learning framework
* [NumPy](https://numpy.org/) - Numerical computing
* [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
* [scikit-learn](https://scikit-learn.org/) - Machine learning utilities
* [Matplotlib](https://matplotlib.org/) - Plotting and visualization
* [Seaborn](https://seaborn.pydata.org/) - Statistical data visualization

[(back to top)](#table-of-contents)

## Roadmap

See the [open issues](https://github.com/Rohit-K814307/harp/issues) for a list of proposed features (and known issues).

[(back to top)](#table-of-contents)

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

[(back to top)](#table-of-contents)

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

[(back to top)](#table-of-contents)

## Contact

Rohit Kulkarni - [rohit_kulkarni@berkeley.edu](mailto:rohit_kulkarni@berkeley.edu)
Lagnajeet Panigrahi - [lpanigrahi@berkeley.edu](mailto:lpanigrahi@berkeley.edu)

Project Link: [https://github.com/Rohit-K814307/harp](https://github.com/Rohit-K814307/harp)

[(back to top)](#table-of-contents)

## Acknowledgments

We would like to thank CalHacks 12.0 & Promise (YC S18) for inspiring us to work on this problem.

[(back to top)](#table-of-contents)
