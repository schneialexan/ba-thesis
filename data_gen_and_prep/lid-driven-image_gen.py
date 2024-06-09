import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
from numba import jit

def find_max_absolute_u(u, imax, jmax):
    return np.max(np.abs(u[1:imax + 1, 1:jmax + 2]))

def find_max_absolute_v(v, imax, jmax):
    return np.max(np.abs(v[1:imax + 2, 1:jmax + 1]))

def select_dt_according_to_stability_condition(Re, dx, dy, tau, u, v, imax, jmax):
    left = (Re / 2) * ((1 / dx ** 2) + (1 / dy ** 2)) ** -1
    middle = dx / (find_max_absolute_u(u, imax, jmax)+.000001)
    right = dy / (find_max_absolute_v(v, imax, jmax)+.000001)
    return tau * min(left, middle, right)

@jit(fastmath=True)      
def set_boundary_conditions(u, v, G, F, jmax, imax):
    # Set boundary conditions for u
    for j in range(jmax + 3):
        u[0, j] = 0.0
        u[imax, j] = 0.0

    for i in range(imax + 2):
        u[i, 0] = -u[i, 1]
        u[i, jmax + 1] = 2.0 - u[i, jmax]

    # Set boundary conditions for v
    for j in range(jmax + 2):
        v[0, j] = -v[1, j]
        v[imax + 1, j] = -v[imax, j]

    for i in range(imax + 3):
        v[i, 0] = 0.0
        v[i, jmax] = 0.0
        
    # Set boundary conditions for F
    for j in range(jmax + 3):
        F[0, j] = u[0, j]
        F[imax, j] = u[imax, j]
    
    for i in range(imax + 2):
        F[i, 0] = u[i, 0]
        F[i, jmax + 1] = u[i, jmax + 1]
    
    # Set boundary conditions for G
    for i in range(imax + 3):
        G[i, 0] = v[i, 0]
        G[i, jmax] = v[i, jmax]
        
    for j in range(jmax + 2):
        G[0, j] = v[0, j]
        G[imax + 1, j] = v[imax + 1, j]


@jit(fastmath=True)
def set_boundary_conditions_p(p, jmax, imax):
    for i in range(imax + 2):
        p[i, 0] = p[i, 1]
        p[i, jmax + 1] = p[i, jmax]

    for j in range(jmax + 2):
        p[0, j] = p[1, j]
        p[imax + 1, j] = p[imax, j]

# ME_X namespace
# Folien 5, Slide 15
@jit(fastmath=True)
def uu_x(u, dx, i, j, alpha):
    return (
        (1 / dx) * ((0.5 * (u[i, j] + u[i + 1, j])) ** 2 - (0.5 * (u[i - 1, j] + u[i, j])) ** 2)
        + (alpha / dx)
        * (
            abs(0.5 * (u[i, j] + u[i + 1, j])) * (0.5 * (u[i, j] - u[i + 1, j])) / 4
            - abs(0.5 * (u[i - 1, j] + u[i, j])) * (0.5 * (u[i - 1, j] - u[i, j])) / 4
        )
    )

@jit(fastmath=True)
def uv_y(u, v, dy, i, j, alpha):
    return (
        (1 / dy) * (
            (0.25 * (v[i, j] + v[i + 1, j]) * (u[i, j] + u[i, j + 1]))
            - (0.25 * (v[i, j - 1] + v[i + 1, j - 1]) * (u[i, j - 1] + u[i, j])) 
        )
        + (alpha / dy)
        * (
            abs(0.5 * (v[i, j] + v[i + 1, j])) * (0.5 * (u[i, j] - u[i, j + 1])) / 4
            - abs(0.5 * (v[i, j - 1] + v[i + 1, j - 1])) * (0.5 * (u[i, j - 1] - u[i, j])) / 4
        )
    )

@jit(fastmath=True)
# Folien 5, Slide 16
def uu_xx(u, dx, i, j):
    return (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx ** 2

@jit(fastmath=True)
def uu_yy(u, dy, i, j):
    return (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy ** 2

@jit(fastmath=True)
def p_x(p, dx, i, j):
    return (p[i + 1, j] - p[i, j]) / dx

# ME_Y namespace
# Folien 5, Slide 17
@jit(fastmath=True)
def uv_x(u, v, dx, i, j, alpha):
    return (
        (1 / dx) * (
            (0.25 * (u[i, j] + u[i, j + 1]) * (v[i, j] + v[i + 1, j]))
            - (0.25 * (u[i - 1, j] + u[i - 1, j + 1]) * (v[i - 1, j] + v[i, j]))
        )
        + (alpha / dx)
        * (
            abs(0.5 * (u[i, j] + u[i, j + 1])) * (0.5 * (v[i, j] - v[i + 1, j])) / 4
            - abs(0.5 * (u[i - 1, j] + u[i - 1, j + 1])) * (0.5 * (v[i - 1, j] - v[i, j])) / 4
        )
    )

@jit(fastmath=True)
def vv_y(v, dy, i, j, alpha):
    return (
        (1 / dy) * ((0.5 * (v[i, j] + v[i, j + 1])) ** 2 - (0.5 * (v[i, j - 1] + v[i, j])) ** 2)
        + (alpha / dy)
        * (
            abs(0.5 * (v[i, j] + v[i, j + 1])) * (0.5 * (v[i, j] - v[i, j + 1])) / 4
            - abs(0.5 * (v[i, j - 1] + v[i, j])) * (0.5 * (v[i, j - 1] - v[i, j])) / 4
        )
    )

# Folien 5, Slide 18
@jit(fastmath=True)
def vv_xx(v, dx, i, j):
    return (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx ** 2

@jit(fastmath=True)
def vv_yy(v, dy, i, j):
    return (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / dy ** 2

@jit(fastmath=True)
def p_y(p, dy, i, j):
    return (p[i, j + 1] - p[i, j]) / dy

# CE namespace
# Folien 5, Slide 19
@jit(fastmath=True)
def u_x(u, dx, i, j):
    return (u[i, j] - u[i - 1, j]) / dx

@jit(fastmath=True)
def v_y(v, dy, i, j):
    return (v[i, j] - v[i, j - 1]) / dy


@jit(fastmath=True)
def compute_f(Re, F, u, v, dx, dy, dt, imax, jmax, alpha):
    for j in range(1, jmax + 2):
        for i in range(1, imax + 1):
            F[i, j] = u[i, j] + dt * (
                (1 / Re) * (uu_xx(u, dx, i, j) + uu_yy(u, dy, i, j))
                - uu_x(u, dx, i, j, alpha)
                - uv_y(u, v, dy, i, j, alpha)
            )

@jit(fastmath=True)
def compute_g(Re, G, u, v, dx, dy, dt, imax, jmax, alpha):
    for i in range(1, imax + 2):
        for j in range(1, jmax + 1):
            G[i, j] = v[i, j] + dt * (
                (1 / Re) * (vv_xx(v, dx, i, j) + vv_yy(v, dy, i, j))
                - uv_x(u, v, dx, i, j, alpha)
                - vv_y(v, dy, i, j, alpha)
            )

@jit(fastmath=True)
def compute_rhs(RHS, F, G, dx, dy, dt, imax, jmax):
    for i in range(1, imax + 1):
        for j in range(1, jmax + 1):
            RHS[i, j] = (1 / dt) * (
                (F[i, j] - F[i - 1, j]) / dx + 
                (G[i, j] - G[i, j - 1]) / dy
            )

@jit(fastmath=True)
def update_step_lgls(RHS, p, dx, dy, imax, jmax):
    for i in range(1, imax + 1):
        for j in range(1, jmax + 1):
            p[i, j] = (
                (1 / (-2 * dx ** 2 - 2 * dy ** 2))
                * (
                    RHS[i, j] * (dx ** 2) * (dy ** 2)
                    - (dy ** 2) * (p[i + 1, j] + p[i - 1, j])
                    - (dx ** 2) * (p[i, j + 1] + p[i, j - 1])
                )
            )

@jit(fastmath=True)
def compute_residual(p, po):
    return np.linalg.norm(p - po)

@jit(fastmath=True)
def compute_u(u, F, p, dx, dt, imax, jmax):
    for i in range(1, imax + 1):
        for j in range(1, jmax + 2):
            u[i, j] = F[i, j] - (dt / dx) * (p[i + 1, j] - p[i, j])

@jit(fastmath=True)
def compute_v(v, G, p, dy, dt, imax, jmax):
    for i in range(1, imax + 2):
        for j in range(1, jmax + 1):
            v[i, j] = G[i, j] - (dt / dy) * (p[i, j + 1] - p[i, j])
            
@jit(fastmath=True)
def prep_data_for_plotting(u, v, p):
    data_u = u.copy()
    data_v = v.copy()
    data_p = p.copy()
    data_interpolated_u = np.zeros((data_u.shape[0], data_v.shape[1]))
    data_interpolated_v = np.zeros((data_u.shape[0], data_v.shape[1]))

    for i in range(data_interpolated_u.shape[0]-1):
        for j in range(data_interpolated_u.shape[1]-1):
            # Siehe Bild im Slack von stagered grid
            data_interpolated_u[i, j]  = (data_u[i, j-1] + data_u[i, j+1])/2
            data_interpolated_v[i, j]  = (data_v[i-1, j] + data_v[i+1, j])/2

    # rotate the data to the left
    data_interpolated_u = np.rot90(data_interpolated_u, 1)[1:-1, 1:-1]
    data_interpolated_v = np.rot90(data_interpolated_v, 1)[1:-1, 1:-1]
    p_rot = np.rot90(data_p, 1)[1:-1, 1:-1]
    return data_interpolated_u, data_interpolated_v, p_rot

def save_matrix(filename, matrix):
    with open(filename, 'w') as file:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                file.write(f"{matrix[i, j]:.5f} ")
            file.write("\n")
    print(f"Matrix saved to {filename}")

if __name__ == "__main__":
    # Constants
    imax = 128
    jmax = 128
    xlength = 1.0
    ylength = 1.0
    dx = xlength / imax
    dy = ylength / jmax
    t_end = 10.0
    tau = 0.5
    eps = 1e-6
    itermax = max(imax, jmax)
    alpha = 0.5
    REs = np.arange(100., 4000., 25.)
    
    plot = True
    
    # Apply tqdm to the outer loop
    with tqdm.tqdm(total=len(REs)) as pbar_re:
        for Re in REs:
            path = f"src/data/{Re}"
            if not os.path.exists(path):
                os.makedirs(path)    

            p = np.zeros((imax + 2, jmax + 2))
            po = np.zeros((imax + 2, jmax + 2))
            RHS = np.zeros((imax + 2, jmax + 2))
            u = np.zeros((imax + 2, jmax + 3))
            F = np.zeros((imax + 2, jmax + 3))
            v = np.zeros((imax + 3, jmax + 2))
            G = np.zeros((imax + 3, jmax + 2))
            
            # Variables
            t = 0
            dt = select_dt_according_to_stability_condition(Re, dx, dy, tau, u, v, imax, jmax)
            with tqdm.tqdm(total=int(t_end/dt)) as pbar:
                while t < t_end:
                    pbar.set_description(f"Re: {Re}, t: {t:.4f}, dt: {dt:.7f}")
                    n = 0
                    res = 99999
                    set_boundary_conditions(u, v, G, F, jmax, imax)
                    # plot the data
                    plot_freq = 0.05
                    if (plot and t % plot_freq < dt):
                        data_interpolated_u, data_interpolated_v, p = prep_data_for_plotting(u, v, p)
                        # save the data with PIL
                        current_t = f'{t:.2f}'
                        plt.imsave(f'{path}/u_{current_t}.png', data_interpolated_u, cmap='gray')
                        plt.imsave(f'{path}/v_{current_t}.png', data_interpolated_v, cmap='gray')
                        plt.imsave(f'{path}/p_{current_t}.png', p, cmap='gray')
                    compute_f(Re, F, u, v, dx, dy, dt, imax, jmax, alpha)
                    compute_g(Re, G, u, v, dx, dy, dt, imax, jmax, alpha)
                    set_boundary_conditions(u, v, G, F, jmax, imax)
                    compute_rhs(RHS, F, G, dx, dy, dt, imax, jmax)
                    p = np.zeros((imax + 2, jmax + 2))
                    while (res > eps or res == 0) and n < itermax:
                        set_boundary_conditions_p(p, jmax, imax)
                        update_step_lgls(RHS, p, dx, dy, imax, jmax)
                        res = compute_residual(p, po)
                        po = p.copy()
                        n += 1
                    compute_u(u, F, p, dx, dt, imax, jmax)
                    compute_v(v, G, p, dy, dt, imax, jmax)
                    t += dt
                    dt = select_dt_according_to_stability_condition(Re, dx, dy, tau, u, v, imax, jmax)
                    pbar.update(1)
            # Save the final state (steady state = ss)
            if plot:
                data_interpolated_u, data_interpolated_v, p_rot = prep_data_for_plotting(u, v, p)
                # save the data as grayscale images
                plt.imsave(f'{path}/u_ss.png', data_interpolated_u, cmap='gray')
                plt.imsave(f'{path}/v_ss.png', data_interpolated_v, cmap='gray')
                plt.imsave(f'{path}/p_ss.png', p_rot, cmap='gray')
                with open(f'{path}/u_ss.png', 'w') as file:
                    for i in range(data_interpolated_u.shape[0]):
                        for j in range(data_interpolated_u.shape[1]):
                            file.write(f"{data_interpolated_u[i, j]:.5f} ")
                        file.write("\n")
                with open(f'{path}/v_ss.png', 'w') as file:
                    for i in range(data_interpolated_v.shape[0]):
                        for j in range(data_interpolated_v.shape[1]):
                            file.write(f"{data_interpolated_v[i, j]:.5f} ")
                        file.write("\n")
            pbar_re.update(1)
        