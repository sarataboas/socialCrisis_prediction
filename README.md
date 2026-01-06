# Socio-Economic Crisis Prediction and Explainability in Time Series

This repository contains the code and experiments developed for the project “Algorithmic Resilience over Efficiency: Understanding When Models Can Be Trusted in Times of Crisis”, focused on socio-economic crisis prediction using multivariate time series data.

The project investigates how class imbalance and limited transparency affect the reliability of predictive models in high-stakes socio-economic settings, and explores explainability techniques as tools for early warning and responsible decision support.


-----
## Hardware Details

All experiments were conducted on a macOS-based personal computer using CPU-only computation.  
The project codebase is fully compatible with GPU acceleration; however, no GPU was required to reproduce the reported results.

This demonstrates that the proposed methods are computationally feasible in standard consumer-grade environments, while remaining scalable to GPU-enabled setups if needed.


## Setup and Installation

To reproduce the experiments in this repository, follow the steps below:

1. Clone the repository (via https)
    ```bash
    git clone https://github.com/sarataboas/socialCrisis_prediction.git
    cd socialCrisis_prediction
    ```

2. Create and activate a virtual environment. The project used Python 3.12.2
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3. Install all packages and dependencies required
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure and Understanding

The project is organized as follows: 
```bash
├── datasets
│   └── oversampling
├── notebooks
├── outputs
└── src
    ├── data
    ├── explainability
    ├── imbalance
    └── models
        └── configs
```

#### **Folder descriptions:**
- ***datasets/***
    - Contains datasets generated during the experiments. The oversampling/ subfolder stores datasets created using window-based event oversampling strategies to mitigate class imbalance.

- ***notebooks/***
    - Includes Jupyter notebooks used for exploratory data analysis, dataset generation, class imbalance analysis, explainability experiments (XAI), and distributional shift analysis. These notebooks document the experimental workflow and support result interpretation.

- ***outputs/***
    - Stores figures, plots, and intermediate results produced during model training, evaluation, and explainability analysis. This includes visualizations used in the final report.

- ***src/***
    - Contains the core implementation of the project, structured into modular components:

    - ***data/*** : Utilities for loading datasets, preprocessing, scaling, handling missing data, and creating temporal windows for time series modeling.

    - ***explainability/*** : Implementations of explainable AI methods, including Integrated Gradients, occlusion-based explainability, LIME adaptations for time series, and distributional shift analysis.

    - ***imbalance/*** : Code related to class imbalance analysis, imbalance metrics, and mitigation strategies such as oversampling and cost-sensitive learning.

    - ***models/*** : LSTM model definitions, training procedures, evaluation routines, and experiment execution scripts.

    - ***configs/*** : JSON configuration files specifying model hyperparameters and the different imbalance-handling strategies evaluated in the experiments.



## How to Run

This repository supports both notebook-based analysis and script-based experiment execution.

### 1. Exploratory Analysis and Dataset Preparation
All exploratory analysis, dataset generation, preprocessing, and explainability experiments are documented in the Jupyter notebooks located in the `notebooks/` folder.  
These notebooks can be run sequentially to reproduce the full analytical pipeline.


### 2. Model Training and Evaluation
Model training and evaluation are executed via Python scripts in the src/models/ directory.
Experiments are fully configurable through JSON files in src/models/configs/.

Single LSTM experiments can be runned in the Class Imabalance Notebook or with:
```bash
cd socialCrisis_prediction
python -m src/models/run_lstm_experiments.py --config src/models/configs/lstm_config.json
```
Change <src/models/configs/lstm_config.json> to the name of the file correspondent to the desired experiment.

To run multiple imbalance-handling strategies sequentially:
```bash
cd socialCrisis_prediction
python -m src.models.run_multiple_models \                                                           
src/models/configs/lstm_config.json \
20 \
lstm_base
```
Change <src/models/configs/lstm_config.json> and <lstm_base> to the name of the file correspondent to the desired experiment.


### 3. Model Performance Results
Model Evaluation results are presented in `notebooks/performance_results.py`

