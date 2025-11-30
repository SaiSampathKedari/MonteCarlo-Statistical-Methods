# MonteCarlo-Statistical-Methods

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)
![Last Commit](https://img.shields.io/github/last-commit/SaiSampathKedari/MonteCarlo-Statistical-Methods)
![Stars](https://img.shields.io/github/stars/SaiSampathKedari/MonteCarlo-Statistical-Methods?style=social)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A visual, implementation-focused companion to *Monte Carlo Statistical Methods* (Robert & Casella), built through
clear simulations, clean visualizations, and practical notebooks designed to teach Monte Carlo ideas the way they are actually used.

---

# Table of Contents
- [Overview](#overview)
- [Sampling Visualizations](#sampling-visualizations)
- [Importance Sampling](#importance-sampling)
- [Control Variates](#control-variates)
- [Brownian Motion](#brownian-motion)
- [Markov Chain Monte Carlo](#markov-chain-monte-carlo)
- [Notebook Gallery](#notebook-gallery)
- [PDF Chapter Library](#pdf-chapter-library)
- [Project Structure](#project-structure)
- [About Me](#about-me)

---

# Overview

This repository builds Monte Carlo ideas from first principles using:

- **Random variable generation:** inverse transform, accept–reject, importance sampling  
- **Monte Carlo estimation:** WLLN, SLLN, CLT  
- **Variance reduction:** control variates, Brownian motion scaling  
- **Markov Chain Monte Carlo:** MH, AM, DR, DRAM  
- **Full diagnostics:** burn-in, mixing, autocorrelation, integrated autocorrelation, ESS  

Each topic is supported by **clean visualizations** and **self-contained Jupyter notebooks**.

---

# Sampling Visualizations

<p align="center">
  <img src="animations/beta_fill.gif" height="230">
  &nbsp;&nbsp;&nbsp;
  <img src="animations/accept_reject_demo.gif" height="230">
</p>

<p align="center"><i>Inverse Transform sampling (left) and Accept–Reject sampling (right).</i></p>

---

# Importance Sampling

<p align="center"> 
  <img src="images/importance_sampling/LaplacePrior_Gaussian_Likelihood.png" height="240"> 
  &nbsp;&nbsp;&nbsp; 
  <img src="images/importance_sampling/Prior_Likelihood_Posterior.png" height="240"> 
</p>

<p align="center"><i>Proposal vs likelihood alignment and resulting posterior for Importance Sampling.</i></p>

---

# Control Variates

<p align="center">
  <img src="images/variance_reduction/mc_vs_cv_var.png" height="230">
  &nbsp;&nbsp;&nbsp;
  <img src="images/variance_reduction/g_vs_h_plot.png" height="230">
</p>

<p align="center"><i>Variance comparison (left) and correlation structure enabling control variates (right).</i></p>

---

# Brownian Motion

<p align="center">
  <img src="images/stochastic_processes/DiffusiveScale.png" height="230">
  &nbsp;&nbsp;&nbsp;
  <img src="images/stochastic_processes/multiple_BM.png" height="230">
</p>

<p align="center"><i>Random-walk scaling to Brownian motion and simulated sample paths.</i></p>

---

# Markov Chain Monte Carlo

The repository implements four MCMC algorithms on both **2D Gaussian** and **banana-shaped** targets:

- Metropolis–Hastings (MH)  
- Adaptive Metropolis (AM)  
- Delayed Rejection (DR)  
- Delayed Rejection Adaptive Metropolis (DRAM)  

All diagnostics—burn-in, mixing, autocorrelation, integrated autocorrelation, and ESS—are derived in the
MH diagnostics notebook and reused across the remaining algorithms.

Below is a DRAM example on the banana-shaped target.

### Target and Laplace Initialization
<p align="center">
  <img src="images/mcmc/Banana_and_LaplaceApproximation.png" height="230">
</p>

### DRAM Sample Exploration
<p align="center">
  <img src="images/mcmc/DRAM_samples2_banana.png" height="500">
</p>

### Mixing Behavior
<p align="center">
  <img src="images/mcmc/Mixing_DRAM_banana.png" height="230">
</p>

### Autocorrelation Diagnostics
<p align="center">
  <img src="images/mcmc/AutoCorr_DRAM_banana.png" height="230">
</p>

---

# Notebook Gallery

### Chapter 2 — Sampling
- [General Transformations](notebooks/ch02_sampling/ch02_general_transforms.ipynb)  
- [Accept–Reject Sampling](notebooks/ch02_sampling/ch02_accept_reject.ipynb)  
- [Exponential RVs](notebooks/ch02_sampling/exponential.ipynb)  
- [Gamma RVs](notebooks/ch02_sampling/gamma.ipynb)

### Chapter 3 — Importance Sampling
- [Cauchy Tail Motivation](notebooks/ch03_importance_sampling/ch03_01_ImportanceSampling_CauchyTail_Motivation.ipynb)  
- [Rare Event Estimation](notebooks/ch03_importance_sampling/ch03_03_ImportanceSampling_RareEvent_Estimation.ipynb)  
- [Self-Normalized IS](notebooks/ch03_importance_sampling/ch03_04_SelfNormalized_ImportanceSampling.ipynb)

### Chapter 4 — Variance Reduction
- [Control Variate Example](notebooks/ch04_variance_reduction/ch04_03_controlVariate_example1.ipynb)

### Chapter 6 — Stochastic Processes
- [Brownian Motion](notebooks/ch06_stochastic_processes/ch04_02_Brownian_Motion.ipynb)

### Chapter 5 — MCMC
- [MH Diagnostics](notebooks/ch05_mcmc/ch05_01_MH_Diagnostics.ipynb)  
- [Adaptive Metropolis — Gaussian](notebooks/ch05_mcmc/ch05_02_AdaptiveMetropolis_Gaussian.ipynb)  
- [Adaptive Metropolis — Banana](notebooks/ch05_mcmc/ch05_03_AdaptiveMetropolis_Banana.ipynb)  
- [Delayed Rejection — Gaussian](notebooks/ch05_mcmc/ch05_04_DelayedRejection_Gaussian.ipynb)  
- [Delayed Rejection — Banana](notebooks/ch05_mcmc/ch05_05_DelayedRejection_Banana.ipynb)  
- [DRAM — Gaussian](notebooks/ch05_mcmc/ch05_06_DRAM_Gaussian.ipynb)  
- [DRAM — Banana](notebooks/ch05_mcmc/ch05_07_DRAM_Banana.ipynb)

---

# PDF Chapter Library

Full write-ups:

- `reports/ch02_general_transforms.pdf`  
- `reports/ch02_accept_reject.pdf`  
- `reports/ch03_01_ImportantSampling_Motivation_weights.pdf`  
- `reports/ch03_02_ImportanceSampling_MC_vs_IS_Variance_Comparison.pdf`  
- `reports/ch03_03_ImportanceSampling_Rare_event_Estimation.pdf`  
- `reports/ch03_04_SelfNormalized_ImportantSampling.pdf`  
- `reports/ch04_01_ControlVariate_Foundations-and-Intution.pdf`  
- `reports/ch04_02_Brownian_Motion.pdf`  
- `reports/ch04_03_ControlVariate_example1.pdf`  
- `reports/ch05_01_MarkovChain_Intro.pdf`  
- `reports/ch05_02_Irreducibility.pdf`  
- `reports/ch05_07_Metropolis-Hastings.pdf`

---

# Project Structure

```text
MonteCarlo-Statistical-Methods/
│
├── animations/
├── images/
│   ├── sampling/
│   ├── importance_sampling/
│   ├── variance_reduction/
│   ├── stochastic_processes/
│   ├── mcmc/
│   └── exponential/
├── notebooks/
│   ├── ch02_sampling/
│   ├── ch03_importance_sampling/
│   ├── ch04_variance_reduction/
│   ├── ch05_mcmc/
│   └── ch06_stochastic_processes/
├── reports/
├── src/
├── README.md
├── index.md
└── pyproject.toml
````

---

# About Me

I study and implement methods in **optimization, control, robotics, Bayesian inference, and probabilistic reasoning**.

* GitHub: [https://github.com/SaiSampathKedari](https://github.com/SaiSampathKedari)
* LinkedIn: [https://linkedin.com/in/sai-sampath-kedari](https://linkedin.com/in/sai-sampath-kedari)
* X: [https://x.com/SSampathKedari](https://x.com/SSampathKedari)
* Email: [sampath@umich.edu](mailto:sampath@umich.edu)
