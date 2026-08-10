import numpy as np
import matplotlib.pyplot as plt

def setup(x_min, x_max, num_pts, t_max, dt):
    dx = (x_max - x_min) / num_pts
    num_steps = int(t_max / dt)
    x = np.linspace(x_min, x_max, num_pts)
    return x, dx, num_steps

def check_cfl(alpha, dt, dx):
    cfl = alpha * dt / dx**2
    if cfl >= 0.5:
        raise ValueError(f"CFL unstable: {cfl:.5f} >= 0.5")
    return cfl

def solve(u, alpha, dt, dx, num_steps, keep_history=True):
    check_cfl(alpha, dt, dx)
    curr_u = u.copy()

    if not keep_history:
        for _ in range(num_steps):
            curr_u[1:-1] = (curr_u[1:-1] +
                            alpha * dt / dx**2 *
                            (curr_u[2:] - 2 * curr_u[1:-1] + curr_u[:-2]))
        return [curr_u]

    all_u = [curr_u.copy()]
    for _ in range(num_steps):
        new_u = curr_u.copy()
        new_u[1:-1] = (curr_u[1:-1] +
                       alpha * dt / dx**2 *
                       (curr_u[2:] - 2 * curr_u[1:-1] + curr_u[:-2]))
        curr_u = new_u
        all_u.append(curr_u.copy())
    return all_u

def plot_solution(x, all_u, timesteps, title='Heat Equation Solution', xlabel='x', ylabel='u(x,t)'):
    plt.figure(figsize=(10, 6))
    for step in timesteps:
        plt.plot(x, all_u[step], label=f't = {step} steps')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/plots/{title.replace(" ", "_")}.png')
    plt.show()