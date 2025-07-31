import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

def langevin_sampling(mu, Lambda, n_steps=10000, dt=0.01, tau=1):
    """
    Langevin sampling for a multivariate Gaussian distribution.

    Parameters:
    -----------
    mu : array-like, shape (d,)
        Mean of the Gaussian.
    Lambda : array-like, shape (d, d)
        Precision matrix (inverse covariance).
    n_steps : int
        Number of sampling steps.
    dt : float
        Time step.
    tau : float
        Time constant.
    Returns:
    --------
    samples : ndarray, shape (n_steps, d)
        Array of sampled points.
    """

    mu = np.asarray(mu)
    d = mu.shape[0]
    
    rng = np.random.default_rng(42)
    noise = rng.normal(size=(n_steps, d))
    # noise = np.random.randn(n_steps, d)
    x = 0.5 * np.ones_like(mu)  # Initial position
    samples = np.zeros((n_steps, d))
    
    # Precompute for efficiency
    Lambda = np.asarray(Lambda)
    sqrt_2dt_tau = np.sqrt(2 * dt / tau)

    for t in range(n_steps):
        grad_logp = -Lambda @ (x - mu)
        x = x + dt / tau * grad_logp + sqrt_2dt_tau * noise[t,:]
        samples[t] = x

    return samples

def hamiltonian_sampling(mu, Lambda, n_steps=1000, dt=0.01, tau_L=1, tau_H = 1, gamma=0.1, mass = 10):
    """
    Hamiltonian (underdamped Langevin) sampling with friction for a multivariate Gaussian.
    Parameters:
    -----------
    mu : array-like, shape (d,)
        Mean of the Gaussian.
    Lambda : array-like, shape (d, d)
        Precision matrix (inverse covariance).
    n_steps : int
        Number of sampling steps.
    dt : float
        Time step.
    tau : float
        Time constant (mass).
    gamma : float
        Friction coefficient.

    Returns:
    --------
    samples : ndarray, shape (n_steps, d)
        Array of sampled points.
    """
    
    mu = np.asarray(mu)
    d = mu.shape[0]
    
    rng = np.random.default_rng(1)
    noise = rng.normal(size=(n_steps, d, 2))
    # x = rng.normal(d)
    x = 0.5 * np.ones_like(mu)
    momentum = rng.normal(d)  # Initial velocity
    samples = np.zeros((n_steps, d))
    

    Lambda = np.asarray(Lambda)
    sqrt_2gamma_dt_tau = np.sqrt(2 * gamma * dt)
    sqrt_2dt_tau = np.sqrt(2 * dt / tau_L)
    
    for t in range(n_steps):
        # Compute gradient of log probability
        grad_logp = -Lambda @ (x - mu)
        # Update momentum (Ornstein-Uhlenbeck with friction and noise)
        momentum += (dt / tau_H) * grad_logp - gamma/mass * momentum * dt + sqrt_2gamma_dt_tau * noise[t, :, 0]  
        # Update position
        x +=  momentum * dt/ (tau_H * mass) + dt / tau_L * grad_logp + sqrt_2dt_tau * noise[t, :, 1] 
        
        samples[t] = x

    return samples

def levy_sampling(mu, Lambda, n_steps=1000, dt=0.01, tau=1):
    """
    Levy sampling for a multivariate Gaussian distribution.

    Parameters:
    -----------
    mu : array-like, shape (d,)
        Mean of the Gaussian.
    Lambda : array-like, shape (d, d)
        Precision matrix (inverse covariance).
    n_steps : int
        Number of sampling steps.
    dt : float
        Time step.
    tau : float
        Time constant.
    gamma_alpha : float
        Shape parameter for Gamma distribution of time constants
    gamma_beta : float
        Rate parameter for Gamma distribution of time constants
        
    Returns:
    --------
    samples : ndarray, shape (n_steps, d)
        Array of sampled points.
    """
    gamma_alpha = 0.7  # Shape parameter for Gamma distribution
    gamma_beta = 1.0   # Rate parameter for Gamma distribution
    
    mu = np.asarray(mu)
    d = mu.shape[0]
    
    rng = np.random.default_rng(42)
    noise = rng.normal(size=(n_steps, d))
    
    x = 0.5 * np.ones_like(mu)  # Initial position
    samples = np.zeros((n_steps, d))
    
    # Precompute for efficiency
    Lambda = np.asarray(Lambda)
    
    # Generate the Gamma distributed sampling time constant
    rng_tau = np.random.default_rng(1)
    time_constants = rng_tau.gamma(gamma_alpha, 1/gamma_beta, n_steps)
    
    for t in range(n_steps):
        tau1 = tau * time_constants[t]
        grad_logp = -Lambda @ (x - mu)
        x = x + dt / tau1 * grad_logp + np.sqrt(2 * dt / tau1) * noise[t,:]
        
        samples[t] = x
        
    return samples
    

def plot_samples_static(samples, X1, X2, Z):
    # plt.ion()
    fig_combined = plt.figure(figsize=(12, 6))
    ax0 = plt.subplot(1, 2, 1)
    ax1 = plt.subplot(1, 2, 2)
   
    # Left plot: Langevin trajectory
    ax0.contourf(X1, X2, Z, levels=10, cmap='Blues', alpha=0.5)
    ax0.set_xlabel('Stimulus s1')
    ax0.set_xticks([])
    ax0.set_ylabel('Stimulus s2')
    ax0.set_yticks([])
    ax0.set_title('Sampling Trajectory') 
    ax0.axis('equal')
    
    # -------------------------------
    # Right plot: Histogram
    ax1.contourf(X1, X2, Z, levels=10, cmap='Blues', alpha=0.5)
    ax1.set_xlabel('Stimulus s1')
    ax1.set_xticks([])
    ax1.set_ylabel('Stimulus s2')
    ax1.set_yticks([])
    ax1.set_title('2D Histogram of Samples')
    ax1.axis('equal')
    
    # Sampling trajectory
    # ax0.plot(samples[:,0], samples[:,1], lw=2, alpha=0.8)
    ax0.plot(samples[:,0], samples[:,1], 'o-', lw=2, alpha=0.8, markersize=3) 
    
    # Sampling histogram    
    hist, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], 
                                        bins=20, range=[[-1, 1], [-1, 1]], density=True)
    Xc, Yc = np.meshgrid(
        (xedges[:-1] + xedges[1:]) / 2, 
        (yedges[:-1] + yedges[1:]) / 2)
            
    Zc = ndimage.gaussian_filter(hist.T, sigma=1)
    ax1.contour(Xc, Yc, Zc, levels=4, cmap='Oranges', alpha=0.9) 
    
    ax0.set_xlim([-1.2, 1.2])
    ax0.set_ylim([-1.2, 1.2])
    
    return fig_combined

    
def plot_samples_animation(samples, X1, X2, Z):
    # plt.ion()
    fig_combined = plt.figure(figsize=(12, 6))
    ax0 = plt.subplot(1, 2, 1)
    ax1 = plt.subplot(1, 2, 2)

    def update(frame):
        # Get the current samples
        current_samples = samples[:frame+1]
        
        # Left plot: Langevin trajectory
        ax0.clear()
        ax0.contourf(X1, X2, Z, levels=10, cmap='Blues', alpha=0.5)
        ax0.set_xlabel('Stimulus s1', fontsize=12)
        ax0.set_xticks([])
        ax0.set_ylabel('Stimulus s2', fontsize=12)
        ax0.set_yticks([])
        ax0.set_title('Sampling Trajectory', fontsize=16) 
        ax0.set_aspect('equal', adjustable='box')

        # -------------------------------
        # Right plot: Histogram
        ax1.clear()
        ax1.contourf(X1, X2, Z, levels=10, cmap='Blues', alpha=0.5)
        ax1.set_xlabel('Stimulus s1', fontsize=12)
        ax1.set_xticks([])
        ax1.set_ylabel('Stimulus s2', fontsize=12)
        ax1.set_yticks([])
        ax1.set_title('2D Histogram of Samples', fontsize=16)
        ax1.set_aspect('equal', adjustable='box')
        
        # Sampling trajectory
        # ax0.plot(current_samples[:,0], current_samples[:,1], lw=2, alpha=0.8)
        
        #  -------------------------------
        # Using LineCollection for sampling trajectory with gradient color
        if len(current_samples) < 51:
            x = current_samples[:,0]
            y = current_samples[:,1]
        else:
            x = current_samples[-50:, 0]
            y = current_samples[-50:, 1]
            
        # Only create LineCollection if we have at least 2 points
        if len(x) >= 2:
            # Prepare segments
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create gradient alpha values
            alphas = np.linspace(0.1, 1.0, len(segments))

            # Create LineCollection with varying alpha
            lc = LineCollection(segments, colors='tab:blue', alpha=alphas, linewidth=2)
            ax0.add_collection(lc)
        elif len(x) == 1:
            ax0.plot(current_samples[:,0], current_samples[:,1], lw=2, alpha=0.8)
            
        #  -------------------------------
        # Sampling histogram    
        if frame > 10:
            hist, xedges, yedges = np.histogram2d(current_samples[:, 0], current_samples[:, 1], 
                                                bins=20, range=[[-1, 1], [-1, 1]], density=True)
            Xc, Yc = np.meshgrid(
                (xedges[:-1] + xedges[1:]) / 2, 
                (yedges[:-1] + yedges[1:]) / 2)
                    
            Zc = ndimage.gaussian_filter(hist.T, sigma=1)
            ax1.contour(Xc, Yc, Zc, levels=4, cmap='Oranges', alpha=0.9) 
            
        return []

    anim = FuncAnimation(fig_combined, update, 
                        frames=len(samples), interval=5, 
                        blit=True, # Must be False for contour updates
                        repeat=False)
    plt.show()
    return anim, fig_combined


def plot_samples_trials_animation(samples, X1, X2, Z):
    # Samples: [dimension, n_trials, n_steps]
    # plt.ion()
    fig_combined = plt.figure(figsize=(6, 6))
    # ax0 = plt.subplot(1, 2, 1)
    # ax1 = plt.subplot(1, 2, 2)
    
    ax1 = plt.gca()

    def update(frame):
        # Get the current samples
        current_samples = samples[:,:,frame]
        
        # Left plot: Langevin trajectory
        # ax0.clear()
        # ax0.contourf(X1, X2, Z, levels=10, cmap='Blues', alpha=0.5)
        # ax0.set_xlabel('Stimulus s1', fontsize=12)
        # ax0.set_xticks([])
        # ax0.set_ylabel('Stimulus s2', fontsize=12)
        # ax0.set_yticks([])
        # ax0.set_title('Sampling Trajectory', fontsize=16) 

        # -------------------------------
        # Right plot: Histogram
        ax1.clear()
        ax1.contourf(X1, X2, Z, levels=10, cmap='Blues', alpha=0.5)
        ax1.set_xlabel('Stimulus s1', fontsize=12)
        ax1.set_xticks([])
        ax1.set_ylabel('Stimulus s2', fontsize=12)
        ax1.set_yticks([])
        ax1.set_title('2D Histogram of Samples', fontsize=16)
        ax1.set_aspect('equal', adjustable='box')
        
        # Samples over trials
        # ax0.scatter(current_samples[0,:], current_samples[1,:], alpha=0.8)
        
        # ax0.set_xlim([-1.2, 1.2])
        # ax0.set_ylim([-1.2, 1.2])
        # ax0.set_aspect('equal', adjustable='box')
        
        #  -------------------------------
        # Sampling histogram    
        hist, xedges, yedges = np.histogram2d(current_samples[0,:], current_samples[1,:], 
                                            bins=20, range=[[-1, 1], [-1, 1]], density=True)
        Xc, Yc = np.meshgrid(
            (xedges[:-1] + xedges[1:]) / 2, 
            (yedges[:-1] + yedges[1:]) / 2)
                
        Zc = ndimage.gaussian_filter(hist.T, sigma=1)
        ax1.contour(Xc, Yc, Zc, levels=4, cmap='Oranges', alpha=0.9) 
        ax1.scatter(current_samples[0,:], current_samples[1,:], alpha=0.8)
        ax1.set_xlim([-1.2, 1.2])
        ax1.set_ylim([-1.2, 1.2])
        
        return []

    anim = FuncAnimation(fig_combined, update, 
                        frames=samples.shape[-1], interval=5, 
                        blit=True, # Must be False for contour updates
                        repeat=False)
    plt.show()
    return anim, fig_combined
