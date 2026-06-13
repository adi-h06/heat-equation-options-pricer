import numpy as np
import matplotlib.pyplot as plt

def setup(x_min, x_max, num_pts, t_max, dt):
    """
    Sets up space and time grid for the heat equation solver. 

    Parameters: 
    x_min: left boundary of space grid
    x_max: right boundary of space grid
    num_pts: number of spacial grid point
    t_max: total time to solve for
    dt: size of time step

    Returns:
    x: spacial grid array
    dx: size of spacial step
    num_steps: number of time steps

    """
    dx = (x_max - x_min) / num_pts
    num_steps = int(t_max / dt)
    x = np.linspace(x_min, x_max, num_pts)
    return x, dx, num_steps

def check_cfl(alpha, dt, dx):
    """
    Checks the CFL stability when using explicit finite differences
    and raises ValueError if the stability condition is violated.

    Parameters:
    alpha: thermal diffusivity
    dt: size of time step
    dx: size of spacial step

    Return:
    cfl: the CFL number
    """

    cfl = alpha * dt / dx**2
    if cfl >= 0.5:
        raise ValueError(f"CFL unstable: {cfl:.5f} >= 0.5")
    return cfl

def solve(u, alpha, dt, dx, num_steps):
    """
    First calls check_cfl and then solves the 1D heat equation
    using the explicit finite difference method.

    Parameters:
    u: initial condition array
    alpha: thermal diffusivity
    dt: size of time step
    dx: size of spacial step
    num_steps: number of time steps to solve for 

    Returns:
    all_u: list of solution arrays at each time step
    """
    check_cfl(alpha, dt, dx)

    all_u = [u.copy()]
    curr_u = u.copy()

    for _ in range(num_steps):
        new_u = curr_u.copy()

        new_u[1:-1] = (curr_u[1:-1] + 
                       alpha * dt / dx**2 * 
                       (curr_u[2:] - 2 * curr_u[1:-1] + curr_u[:-2]))

        curr_u = new_u
        all_u.append(curr_u.copy())

    return all_u

def plot_solution(x, all_u, timesteps, title='Heat Equation Solution', xlabel='x', ylabel='u(x,t)'):
    """
    Plots the heat equation solution at the specified time steps

    Parameters:
    x: spatial grid array
    all_u: list of solution arrays at each time step
    timesteps: list of time step indices to plot
    title: plot title
    xlabel: x axis label
    ylabel: y axis label
    """
    plt.figure(figsize=(10,6))
    
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