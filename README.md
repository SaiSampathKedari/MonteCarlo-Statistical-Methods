# MonteCarlo-Statistical-Methods

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)
![Last Commit](https://img.shields.io/github/last-commit/SaiSampathKedari/MonteCarlo-Statistical-Methods)
![Stars](https://img.shields.io/github/stars/SaiSampathKedari/MonteCarlo-Statistical-Methods?style=social)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A visual, implementation-focused companion to **Monte Carlo Statistical Methods** (Robert & Casella).  
This repository builds intuition through **plots, animations, and simulations**, aiming to provide a **clear, lightweight, practical guide** to Monte Carlo techniques and stochastic simulation.

---

## 📘 Table of Contents

- [Overview](#overview)
- [Sampling Visualizations](#sampling-visualizations)
- [Importance Sampling](#importance-sampling--self-normalized-is)
- [Control Variates](#️-control-variates)
- [Brownian Motion](#-brownian-motion-random-walks--continuous-paths)
- [Markov Chain Monte Carlo (MCMC)](#markov-chain-monte-carlo-mcmc)
- [Notebook Gallery](#notebook-gallery)
- [PDF Chapter Library](#pdf-chapter-library)
- [Project Structure](#project-structure)
- [About Me](#about-me)

---

## Overview

This repository develops Monte Carlo ideas in a **progressive, example-driven way**:

1. **Random variable generation**  
   Inverse Transform, Accept–Reject, Importance Sampling.

2. **Monte Carlo estimation**  
   Law of Large Numbers (WLLN, SLLN), Central Limit Theorem, estimated error bars.

3. **Variance reduction**  
   Control Variates and related ideas, with side-by-side variance comparisons.

4. **Markov Chain Monte Carlo (MCMC)**  
   Metropolis–Hastings, Adaptive Metropolis, Delayed Rejection, and DRAM on nontrivial 2D targets, with **diagnostics**: mixing, autocorrelation, integrated autocorrelation time, and effective sample size.

5. **Bayesian inference & filtering basics**  
   Importance sampling and self-normalized IS used in a Bayesian context.

6. **Stochastic processes**  
   Building Brownian Motion from discrete random walks via diffusive scaling.

The goal is not just to implement algorithms, but to **see how they behave** through carefully designed plots and diagnostics.

---

## 🎥 Sampling Visualizations

<p align="center">
  <img src="animations/beta_fill.gif" height="240">
  &nbsp;&nbsp;&nbsp;
  <img src="animations/accept_reject_demo.gif" height="240">
</p>

<p align="center">
  <i>Inverse Transform Sampling (Beta) &nbsp; | &nbsp; Accept–Reject Sampling (Laplace → Normal)</i>
</p>

---

## 📊 Importance Sampling & Self-Normalized IS

<p align="center"> 
  <img src="images/importance_sampling/LaplacePrior_Gaussian_Likelihood.png" height="250"> 
  &nbsp;&nbsp;&nbsp; 
  <img src="images/importance_sampling/Prior_Likelihood_Posterior.png" height="250"> 
</p> 

<p align="center">
  Visualizing proposal mismatch, likelihood weighting, and posterior formation in Importance Sampling and Self-Normalized IS.
</p>

---

## 🎚️ Control Variates

<p align="center">
  <img src="images/variance_reduction/mc_vs_cv_var.png" height="235">
  &nbsp;&nbsp;&nbsp;
  <img src="images/variance_reduction/g_vs_h_plot.png" height="235">
</p>

<p align="center">
  Monte Carlo vs Control Variate estimator spread, and the correlation structure that drives variance reduction.
</p>

---

## 🟦 Brownian Motion: Random Walks → Continuous Paths

<p align="center">
  <img src="images/stochastic_processes/DiffusiveScale.png" height="235">
  &nbsp;&nbsp;&nbsp;
  <img src="images/stochastic_processes/multiple_BM.png" height="235">
</p>

<p align="center">
  Constructing Brownian Motion via diffusive scaling of simple random walks and visualizing multiple sample paths.
</p>

---

## Markov Chain Monte Carlo (MCMC)

MCMC is implemented and visualized on **2D Gaussian** and **banana-shaped** targets, with a focus on:

- Metropolis–Hastings (MH)
- Adaptive Metropolis (AM)
- Delayed Rejection (DR)
- Delayed Rejection Adaptive Metropolis (DRAM)

The core diagnostics — **burn-in, mixing, autocorrelation, integrated autocorrelation time, and ESS** — are derived and illustrated in the MH diagnostics notebook, then reused across the other algorithms.

Below is a DRAM run on the banana distribution, with Laplace-based initialization and full diagnostics:

<p align="center">
  <img src="images/mcmc/Banana_and_LaplaceApproximation.png" height="240">
</p>

<p align="center">
  <i>Banana-shaped target distribution with Laplace approximation used to initialize the chain.</i>
</p>

<p align="center">
  <img src="images/mcmc/DRAM_samples2_banana.png" height="240">
</p>

<p align="center">
  <i>DRAM samples exploring the banana distribution.</i>
</p>

<p align="center">
  <img src="images/mcmc/Mixing_DRAM_banana.png" height="240">
</p>

<p align="center">
  <i>Trace plots for DRAM on the banana target, illustrating mixing and burn-in.</i>
</p>

<p align="center">
  <img src="images/mcmc/AutoCorr_DRAM_banana.png" height="240">
</p>

<p align="center">
  <i>Autocorrelation diagnostic for DRAM on the banana target.</i>
</p>

---

## Notebook Gallery

### **Chapter 2 — Sampling**

- [General Transformations (Beta / Gamma / Chi-Square)](notebooks/ch02_sampling/ch02_general_transforms.ipynb)  
- [Accept–Reject Sampling](notebooks/ch02_sampling/ch02_accept_reject.ipynb)  
- [Exponential RVs](notebooks/ch02_sampling/exponential.ipynb)  
- [Gamma RVs](notebooks/ch02_sampling/gamma.ipynb)

### **Chapter 3 — Importance Sampling**

- [Cauchy Tail Motivation](notebooks/ch03_importance_sampling/ch03_01_ImportanceSampling_CauchyTail_Motivation.ipynb)  
- [Rare Event Estimation](notebooks/ch03_importance_sampling/ch03_03_ImportanceSampling_RareEvent_Estimation.ipynb)  
- [Self-Normalized IS](notebooks/ch03_importance_sampling/ch03_04_SelfNormalized_ImportanceSampling.ipynb)

### **Chapter 4 — Variance Reduction & Processes**

- [Control Variate Example](notebooks/ch04_variance_reduction/ch04_03_controlVariate_example1.ipynb)  
- [Brownian Motion Simulation](notebooks/ch06_stochastic_processes/ch04_02_Brownian_Motion.ipynb)

### **Chapter 5 — MCMC**

- [Metropolis–Hastings and Diagnostics (2D Gaussian)](notebooks/ch05_mcmc/ch05_01_MH_Diagnostics.ipynb)  
- [Adaptive Metropolis — Gaussian](notebooks/ch05_mcmc/ch05_02_AdaptiveMetropolis_Gaussian.ipynb)  
- [Adaptive Metropolis — Banana](notebooks/ch05_mcmc/ch05_03_AdaptiveMetropolis_Banana.ipynb)  
- [Delayed Rejection — Gaussian](notebooks/ch05_mcmc/ch05_04_DelayedRejection_Gaussian.ipynb)  
- [Delayed Rejection — Banana](notebooks/ch05_mcmc/ch05_05_DelayedRejection_Banana.ipynb)  
- [DRAM — Gaussian](notebooks/ch05_mcmc/ch05_06_DRAM_Gaussian.ipynb)  
- [DRAM — Banana](notebooks/ch05_mcmc/ch05_07_DRAM_Banana.ipynb)

---

## PDF Chapter Library

Structured PDF write-ups (theory + worked examples):

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

## 🧱 Project Structure

```text
MonteCarlo-Statistical-Methods/
│
├── animations/                     # GIFs used in README and notebooks
│   └── *.gif
├── images/                         # All static figures
│   ├── sampling/
│   ├── importance_sampling/
│   ├── variance_reduction/
│   ├── stochastic_processes/
│   ├── mcmc/
│   └── exponential/
├── notebooks/                      # Jupyter notebooks (chapter-organized)
│   ├── ch02_sampling/
│   ├── ch03_importance_sampling/
│   ├── ch04_variance_reduction/
│   ├── ch05_mcmc/
│   └── ch06_stochastic_processes/
├── reports/                        # PDF write-ups
├── src/                            # Python package: algorithms and utilities
│   ├── sampling/
│   ├── importance_sampling/
│   ├── variance_reduction/
│   ├── stochastic_processes/
│   └── mcmc/
├── README.md
├── index.md
└── pyproject.toml
````

X: https://x.com/SSampathKedari

Email: sampath@umich.edu

::contentReference[oaicite:0]{index=0}
