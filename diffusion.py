import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_steps = 500     # Number of diffusion steps
num_samples = 100    # Number of trajectories
beta_start = 0.0001  # Starting noise level
beta_end = 0.02      # Ending noise level

# Linear beta schedule for diffusion
betas = np.linspace(beta_start, beta_end, num_steps)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)
alpha_bar_all = np.concatenate(([1], alpha_bars))  # Includes alpha_bar_0 = 1

# Initial distribution: mixture of two Gaussians (bimodal)
def initial_distribution(num_samples):
    mix = np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5])
    means = np.array([-2, 2])
    std = 0.5
    samples = np.random.normal(loc=means[mix], scale=std, size=num_samples)
    return samples

# Forward diffusion process at timestep t
def forward_diffusion(x0, t, betas, alpha_bars):
    noise = np.random.normal(0, 1, size=x0.shape)
    sqrt_alpha_bar = np.sqrt(alpha_bars[t])
    sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bars[t])
    xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    return xt

# Simulate forward diffusion for all samples
def simulate_forward_diffusion(num_samples, num_steps, betas, alpha_bars):
    x0 = initial_distribution(num_samples)
    trajectories = np.zeros((num_steps + 1, num_samples))
    trajectories[0] = x0
    for t in range(num_steps):
        trajectories[t + 1] = forward_diffusion(trajectories[0], t, betas, alpha_bars)
    return trajectories

# Reverse diffusion step
def reverse_step(x_t, t, x_0, betas, alphas, alpha_bar_all):
    beta_t = betas[t - 1]  # t from 1 to num_steps, so betas[t-1] is beta_t
    alpha_t = alphas[t - 1]
    alpha_bar_t = alpha_bar_all[t]
    alpha_bar_t_minus_1 = alpha_bar_all[t - 1]
    coeff1 = np.sqrt(alpha_t) * (1 - alpha_bar_t_minus_1) / (1 - alpha_bar_t)
    coeff2 = np.sqrt(alpha_bar_t_minus_1) * beta_t / (1 - alpha_bar_t)
    mu = coeff1 * x_t + coeff2 * x_0
    sigma = np.sqrt(beta_t)
    z = np.random.normal(0, 1, size=x_t.shape)
    x_t_minus_1 = mu + sigma * z
    return x_t_minus_1

# Simulate reverse diffusion for a single sample
def simulate_reverse_diffusion_single(x_T, x_0, num_steps, betas, alphas, alpha_bar_all):
    reverse_traj = [x_T]
    x_t = x_T
    for t in range(num_steps, 0, -1):
        x_t = reverse_step(x_t, t, x_0, betas, alphas, alpha_bar_all)
        reverse_traj.append(x_t)
    return np.array(reverse_traj[::-1])  # Reverse so index 0 is x_0

# Simulate reverse diffusion for all samples
def simulate_reverse_diffusion(trajectories, num_samples, num_steps, betas, alphas, alpha_bar_all):
    reverse_trajectories = np.zeros((num_steps + 1, num_samples))
    for i in range(num_samples):
        x_T = trajectories[num_steps, i]
        x_0 = trajectories[0, i]
        reverse_trajectories[:, i] = simulate_reverse_diffusion_single(x_T, x_0, num_steps, betas, alphas, alpha_bar_all)
    return reverse_trajectories

# Simulate forward and reverse diffusion
forward_trajectories = simulate_forward_diffusion(num_samples, num_steps, betas, alpha_bars)
reverse_trajectories = simulate_reverse_diffusion(forward_trajectories, num_samples, num_steps, betas, alphas, alpha_bar_all)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Forward Diffusion Plot
timesteps = np.arange(num_steps + 1)
for i in range(num_samples):
    ax1.plot(timesteps, forward_trajectories[:, i], color='black', alpha=0.02, zorder=1)

# Distributions at selected timesteps (forward)
x_range = np.linspace(-5, 5, 200)
dist_timesteps = [0, 100, 200, 300, 400, 500]
colors = ['darkblue', 'lightblue', 'skyblue', 'lavender', 'plum', 'purple']
width_factor = 5.0

for idx, t in enumerate(dist_timesteps):
    kde = gaussian_kde(forward_trajectories[t])
    dist_kde = kde(x_range)
    dist_kde_normalized = dist_kde / np.max(dist_kde)
    ax1.fill_betweenx(x_range, t - dist_kde_normalized * width_factor, t + dist_kde_normalized * width_factor,
                      alpha=0.8, color=colors[idx], zorder=10, label=f'T={t}' if t in [0, 100] else "")

ax1.set_xlabel('Timesteps (Forward)')
ax1.set_ylabel('$x_t$')
ax1.set_title('Forward Diffusion')
for t in dist_timesteps:
    ax1.axvline(t, color='black', linestyle='--', alpha=0.3)

# Reverse Diffusion Plot
reverse_timesteps = np.arange(num_steps, -1, -1)  # From T to 0
for i in range(num_samples):
    ax2.plot(reverse_timesteps, reverse_trajectories[::-1, i], color='black', alpha=0.05, zorder=1)

# Distributions at selected timesteps (reverse)
reverse_dist_timesteps = [0, 100, 200, 300, 400, 500]
for idx, t in enumerate(reverse_dist_timesteps):
    s = t  # Convert t to index in reverse_trajectories
    kde = gaussian_kde(reverse_trajectories[s])
    dist_kde = kde(x_range)
    dist_kde_normalized = dist_kde / np.max(dist_kde)
    ax2.fill_betweenx(x_range, t - dist_kde_normalized * width_factor, t + dist_kde_normalized * width_factor,
                      alpha=0.8, color=colors[idx], zorder=10, label=f'T={t}' if t in [100, 0] else "")

ax2.set_xlabel('Timesteps (Reverse)')
ax2.set_ylabel('$x_t$')
ax2.set_title('Reverse Diffusion')
for t in reverse_dist_timesteps:
    ax2.axvline(t, color='black', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

plt.savefig("diffusion.png")