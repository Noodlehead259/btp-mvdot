# Project Progress

## 1. Project Initialization

- Created the project repository and established the initial project structure.
- Set up the Python virtual environment and CUDA-enabled PyTorch environment.
- Configured the project to use GPU acceleration for computationally intensive experiments.
- Added the MNIST dataset and organized the dataset under the `data/` directory.
- Initialized Git version control and connected the project repository to GitHub.

## 2. Research Hypothesis

- Studied the motivation behind unaligned multi-view clustering in the MvDOT paper.
- Formulated an initial experimental hypothesis that the same underlying samples can receive different cluster assignments when their views are independently perturbed.
- Selected MNIST as the initial experimental dataset.
- Used Gaussian noise as a controlled perturbation to create a noisy view of the original MNIST data.

## 3. Full-Scale Hypothesis Experiment

- Implemented GPU-accelerated K-means clustering on the complete MNIST training set containing 60,000 samples.
- Performed clustering independently on the clean and noisy views.
- Used Gaussian noise with standard deviation σ = 0.25.
- Implemented exact cluster-label alignment between the clean and noisy clustering results.
- Identified 2,765 samples whose cluster assignments changed after noise was introduced.
- Measured a cluster disagreement rate of 4.61%.
- Obtained NMI = 0.8832 and ARI = 0.8895.

## 4. Cluster Instability Analysis

- Developed an interactive visualization for analyzing cluster assignment changes between the clean and noisy views.
- Created a normalized cluster transition matrix showing the percentage of samples transitioning from each clean cluster to each noisy cluster.
- Added per-cluster instability measurements to quantify the percentage of samples whose assignments changed within each clean cluster.
- Completed the first research-quality visualization of cluster assignment instability under Gaussian noise.