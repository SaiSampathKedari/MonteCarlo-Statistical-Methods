# MonteCarlo-Statistical-Methods

This repository includes step-by-step implementations and visualizations of key Monte Carlo techniques:

1. Random variable generation (Inverse Transform, Accept–Reject, Importance Sampling)  
2. Monte Carlo estimation and convergence (WLLN, SLLN, CLT)  
3. Variance reduction methods (Control Variates, Multilevel Control Variates)  
4. Markov Chain Monte Carlo (MCMC)  
5. Bayesian inference and filtering basics  
6. Applications to statistical estimation, learning, and robotics  

🚧 Work in progress — starting with fundamental sampling and convergence demos.

---

### 🎥 Sampling Visualizations

<p align="center">
  <img src="notebooks/images/Ch02_general_tranformations/beta_fill.gif" height="250">
  &nbsp;&nbsp;&nbsp;
  <img src="notebooks/animations/accept_reject_demo.gif" height="250">
</p>

<p align="center">
  <i>Inverse Transform Sampling (Beta distribution) &nbsp; | &nbsp; Accept–Reject Sampling (Laplace → Normal)</i>
</p>

---

### 📊 Importance Sampling & Self-Normalized Importance Sampling 
<p align="center"> 
  <img src="notebooks/images/ch03_importance_sampling/LaplacePrior_Gaussian_Likelihood.png" height="260"> 
  &nbsp;&nbsp;&nbsp; 
  <img src="notebooks/images/ch03_importance_sampling/Prior_Likelihood_Posterior.png" height="260"> 
</p> 

<p align="center"> 
  Visualizing the proposal distribution, likelihood weighting, and posterior formation in Importance Sampling and Self-Normalized Importance Sampling. 
</p>

---

### 🎚️ Control Variates

<p align="center">
  <img src="notebooks/images/ch04_controlVariate/mc_vs_cv_var.png" height="240">
  &nbsp;&nbsp;&nbsp;
  <img src="notebooks/images/ch04_controlVariate/g_vs_h_plot.png" height="240">
</p>

<p align="center">
  Comparing vanilla Monte Carlo estimates with Control Variates (left), and visualizing how the chosen control variate 
  is strongly correlated with the target function (right). This correlation drives the variance reduction effect.
</p>

---


### 🟦 Brownian Motion: From Random Walks to Continuous Paths

<p align="center">
  <img src="notebooks/images/ch04_controlVariate/DiffusiveScale.png" height="240">
  &nbsp;&nbsp;&nbsp;
  <img src="notebooks/images/ch04_controlVariate/multiple_BM.png" height="240">
</p>

<p align="center">
  Constructing Brownian Motion from the diffusive scaling of simple random walks (left), and several Brownian 
  sample paths with the characteristic square-root growth of spread over time (right).
</p>

---
