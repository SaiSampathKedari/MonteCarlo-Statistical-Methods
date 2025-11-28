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
- [Notebook Gallery](#notebook-gallery)
- [PDF Chapter Library](#pdf-chapter-library)
- [Project Structure](#project-structure)
- [About Me](#about-me)

---

## Overview

This repository includes step-by-step implementations and visual explanations of:

1. **Random variable generation**  
   Inverse Transform, Accept–Reject, Importance Sampling  
2. **Monte Carlo estimation**  
   WLLN, SLLN, CLT  
3. **Variance reduction**  
   Control Variates, Multilevel CV  
4. **Markov Chain Monte Carlo (MCMC)**  
5. **Bayesian inference & filtering basics**  
6. **Stochastic processes**  
   Random Walks → Brownian Motion  

🚧 *Work in progress — expanding steadily into a full visual companion to Monte Carlo methods.*

---

## 🎥 Sampling Visualizations

<p align="center">
  <img src="animations/beta_fill.gif" height="240">
  &nbsp;&nbsp;&nbsp;
  <img src="animations/accept_reject_demo.gif" height="240">
</p>

<p align="center"><i>Inverse Transform Sampling (Beta) &nbsp; | &nbsp; Accept–Reject Sampling (Laplace → Normal)</i></p>

---

## 📊 Importance Sampling & Self-Normalized IS

<p align="center"> 
  <img src="images/importance_sampling/LaplacePrior_Gaussian_Likelihood.png" height="250"> 
  &nbsp;&nbsp;&nbsp; 
  <img src="images/importance_sampling/Prior_Likelihood_Posterior.png" height="250"> 
</p> 

<p align="center">
  Visualizing proposal mismatch, likelihood weighting, and posterior formation in Importance Sampling and SNIS.
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
  Constructing Brownian Motion through diffusive scaling of simple random walks, and visualizing multiple sample paths.
</p>

---

# Notebook Gallery

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
- [Metropolis–Hastings Example](notebooks/ch05_mcmc/ch05_08_metropolis_Hastings_MCMC_example1.ipynb)

---

# PDF Chapter Library

Structured PDF write-ups:

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

# 🧱 Project Structure

```text
MonteCarlo-Statistical-Methods/
│
├── animations/
├── images/
│   ├── sampling/
│   ├── importance_sampling/
│   ├── variance_reduction/
│   ├── stochastic_processes/
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
```

---

## 👤 About Me

Interested in **robotics, optimization, control, Bayesian inference, and probabilistic reasoning**.

- GitHub: https://github.com/SaiSampathKedari  
- LinkedIn: https://linkedin.com/in/sai-sampath-kedari  
- X: https://x.com/SSampathKedari  
- Email: sampath@umich.edu

