import numpy as np
from scipy import integrate

def closed_bath_fit(target_dos, nb_bath, density, spectral_tol=1e-5, omega_tol=1e-3, omega_max=20., break_points=None):
    omegas = np.linspace(-omega_max, omega_max, round(2 * omega_max / omega_tol) + 1)
    
    density = density(omegas)
    density[np.abs(target_dos(omegas)) < spectral_tol] = 0.
    
    cum_density = np.cumsum(density)
    pts = np.linspace(cum_density[0], cum_density[-1], 2 * nb_bath + 1)
    omega_pts = np.interp(pts, cum_density, omegas)
    bath_energies = omega_pts[1::2]
    intervals = omega_pts[::2]
    
    bath_couplings = [integrate.quad(target_dos, intervals[i], intervals[i+1], points=break_points) for i in range(len(intervals) - 1)]
    bath_couplings = np.sqrt(np.asarray(bath_couplings)[:, 0])
    
    return bath_energies, bath_couplings
    