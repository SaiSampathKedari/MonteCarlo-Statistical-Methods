import numpy as np

def generate_simple_random_walk_steps(num_steps: int) -> np.ndarray:
    """
    Generate i.i.d. symmetric ±1 random variables using inverse transform sampling.

    Parameters
    ----------
    num_steps : int
        Number of steps to generate.

    Returns
    -------
    np.ndarray
        Array of shape (num_steps,) containing values +1 or -1.
    """
    
    # Uniform(0,1) samples
    X = np.random.rand(num_steps) # uniform samples
    
    # Inverse CDF transformation for P(X=1)=0.5, P(X=-1)=0.5
    X[X>0.5] = 1.0
    X[X<=0.5] = -1.0
    return X

def generate_random_walk(num_steps, n):
    """
    Generate a scaled simple random walk:
        W_n(k/n) = S_k / sqrt(n)

    This corresponds to the diffusive scaling used to construct Brownian motion.

    Parameters
    ----------
    num_steps : int
        Total number of increments (S_k) in the walk.
    n : int
        Scaling factor: number of steps per unit time (controls the √n scaling).

    Returns
    -------
    np.ndarray
        Scaled random walk path of length num_steps + 1, starting at zero.
    """
    
    steps = generate_simple_random_walk_steps(num_steps)
    W_n = np.concatenate((np.array([0]), np.cumsum(steps)/np.sqrt(n)))
    return W_n

def brownian_motion_simulate(T, dt):
    """
    Generate a single sample path of Brownian motion using Euler–Maruyama:
        B(t_{k+1}) = B(t_k) + sqrt(dt) * Z_k,    Z_k ~ N(0,1)

    Parameters
    ----------
    T : float
        Final simulation time.
    dt : float
        Time step size.

    Returns
    -------
    np.ndarray
        Brownian motion sample path of length ceil(T/dt)+1, starting at B(0)=0.
    """
    
    num_samples = int(np.ceil(T/dt) + 1)
    samples = np.random.randn(num_samples) * np.sqrt(dt)
    samples[0] = 0.0
    brownian_sample_path = np.cumsum(samples)
    return brownian_sample_path