# Oro-manual assessment of food storage stability in a wild rodent

This repository contains the data, analysis code, and modeling scripts to reproduce the findings and figures of the paper:
**"Oro-manual assessment of food storage stability in a wild rodent"**

---

## 1. Paper Overview & Synthesis

To survive in seasonal environments, animals must decide whether to consume food immediately or cache it for the future. In nature, foragers encounter complex, structured food items (like nuts with hard shells) where future preservation value depends critically on shell integrity. 

This study investigates how wild agoutis (*Dasyprocta punctata*)—a critical seed disperser of the Neotropics—evaluate food items to determine their storage stability (preservability). 

### Key Scientific Findings:
1. **Eating to Scatter-Hoarding State Transition**: Agoutis undergo a sharp transition from an eating mode to a persistent scatter-hoarding mode upon encountering nuts. This transition is modeled mathematically using a **Hidden Markov Model (HMM)**. Model comparison using Akaike Information Criterion (AIC) reveals that a two-state HMM provides the best fit compared to trend, changepoint, or independent-and-identically-distributed (IID) models.
2. **Shell-Integrity Decisions**: Scatter-hoarding decisions depend strictly on the integrity of the shell. Agoutis cache empty nuts with intact shells but reject those with breached shells (which have high perishability and pilferage risk).
3. **Somatosensory Shell Assessment**: Agoutis actively assess shell integrity using oral haptic probing. They navigate the nut's surface, using their lower incisors and lower lip (acting as a **somatosensory fovea**) to locate breaches.
4. **Optimal Search Anisotropy (Cassini Ovoid Walks)**: Agoutis' surface exploration strategies conform to the nut's curvature (modeled as a **Cassini ovoid**). Simulating random walks on this geometry shows that the agoutis' asymmetric exploration strategy is optimal for locating breaches on curved, anisotropic surfaces.
5. **Evidence Accumulation (Drift Diffusion Model)**: The agoutis' decision process is modeled using a **Drift Diffusion Model (DDM)**, showing how haptic evidence accumulated across oral mouth contacts/jaw clamps drives decisions toward caching thresholds based on hole size and shell condition.

---

## 2. Repository Directory Structure

The repository is structured logically to separate the analyses into modular toolkit scripts:

*   **`assessment_paper_code.ipynb`**: The central Jupyter Notebook that loads the data, runs the analysis pipeline, and generates the figures for the paper.
*   **`agouti_behavior_analysis.py`**: Handles ethogram data loading, behavioral sequence reconstruction, and outcome probability plots.
*   **`trajectory_simulation.py`**: Implements the Cassini ovoid geometric mesh, geodesic distance calculations, and search strategy simulations.
*   **`ovoid_projection_mapping.py`**: Performs 2D flattening, Cassini ovoid projection, and somatosensory touch density mapping.
*   **`hmm_modeling_pipeline.py`**: Implements the bivariate HMM forward-backward algorithm, model fitting, and trial alignment.
*   **`ddm_walker_toolkit.py`**: Implements the Drift Diffusion Model (DDM) for decision-making and walker calibration.
*   **`data_paper/`**: Contains all experimental datasets (ethograms, arena trials, putrefaction rates, soil data, and fruit parametrics).
*   **`figures/`**: Folder where generated figures are automatically saved as vector graphics (`.svg`).
*   **`original_scripts_backup/`**: Folder containing the original version of all scripts before optimization and formatting.

---

## 3. Getting Started & Local Installation

To download the repository and rerun all the analyses on your local computer, follow these steps:

### Prerequisites
Make sure you have Python 3.8+ installed on your computer.

### Step 1: Clone the Repository
Clone this repository to your local machine using git:
```bash
git clone https://github.com/Lucas97223/Agouti-neuroethology.git
cd Agouti-neuroethology
```
*(Note: If you downloaded the folder directly, simply navigate to the folder in your terminal).*

### Step 2: Create a Virtual Environment (Recommended)
Creating a virtual environment ensures that the project's dependencies do not conflict with your global python installation.

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies
Install all required libraries using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 4. Running the Analyses

All analyses and figure-generating scripts are run directly from the Jupyter Notebook:

1.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  In the browser tab that opens, click on `assessment_paper_code.ipynb`.
3.  Run all cells (`Cell -> Run All` or `Kernel -> Restart & Run All`).

### Note on Modeling Run Times
The Drift Diffusion Model (DDM) fitting requires pre-computed trace simulations, which normally take several minutes to generate. To make the code run instantly, we have pre-computed these trace files (`.npz` files) and placed them in the root of this folder. The scripts will automatically detect and load these files out of the box.