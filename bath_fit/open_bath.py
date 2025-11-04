
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import cvxpy as cp
from scipy import optimize


def lorentzian(center, width):
    return lambda w: 1. / ((w - center)**2 + width**2)


def find_best_v(target_func, 
                eps_list, 
                dissip_rate, 
                weight, 
                omega_max=20., 
                break_points=None):
    
    lorentzian_list = [lorentzian(eps, dissip_rate) for eps in eps_list]
    f_vec = np.empty(len(eps_list), dtype=float)
    for k, lor in enumerate(lorentzian_list):
        f_vec[k] = integrate.quad(lambda w: target_func(w) * weight(w) * lor(w), -omega_max, omega_max, points=break_points)[0]
    
    overlap_mat = np.empty((len(f_vec), len(f_vec)), dtype=float)
    for k1, lor1 in enumerate(lorentzian_list):
        overlap_mat[k1, k1] = integrate.quad(lambda w: weight(w) * lor1(w)**2, -omega_max, omega_max)[0]
        for k2 in range(k1):
            lor2 = lorentzian_list[k2]
            overlap_mat[k1, k2] = integrate.quad(lambda w: weight(w) * lor1(w) * lor2(w), -omega_max, omega_max)[0]
            overlap_mat[k2, k1] = overlap_mat[k1, k2]
            
    v = cp.Variable(len(eps_list))
    prob = cp.Problem(cp.Minimize(cp.quad_form(v, cp.psd_wrap(overlap_mat)) - 2 * f_vec @ v), [v >= 0.])
    prob.solve()
    if prob.status != "optimal":
        print(prob.status)
    v_values = v.value
    
    f_norm = integrate.quad(lambda w: weight(w) * target_func(w)**2, -omega_max, omega_max, points=break_points)[0]
    chi_sqr = f_norm + prob.value
    
    # n_neg = sum(v_values < 1e-10)
    # if n_neg > 0:
    #     print(f"/!\ {n_neg} lorentzians were set to zero.")
        
    v_values[v_values < 0.] = 0.
    return v_values, chi_sqr


def find_best_dissip_and_v(target_func,
                          eps_list,
                          weight,
                          dissip_min=1e-2,
                          omega_max=20., 
                          break_points=None, 
                          tol=1e-3):
                          
    res = optimize.minimize_scalar(lambda x: find_best_v(target_func, 
                                                         eps_list, 
                                                         x**2 + dissip_min, 
                                                         weight, 
                                                         omega_max=omega_max, 
                                                         break_points=break_points)[-1],
                                   tol=tol)

    dissip_rate = res.x ** 2 + dissip_min
    v_values, chi_sqr = find_best_v(target_func, 
                                    eps_list, 
                                    dissip_rate, 
                                    weight, 
                                    omega_max=omega_max, 
                                    break_points=break_points)
    
    return v_values, dissip_rate, chi_sqr
    
    
def find_bath_energies(target_dos, nb_bath, density, spectral_tol=1e-5, omega_tol=1e-3, omega_max=20.):
    
    omegas = np.linspace(-omega_max, omega_max, round(2 * omega_max / omega_tol) + 1)
    
    density = density(omegas)
    density[np.abs(target_dos(omegas)) < spectral_tol] = 0.
    
    cum_density = np.cumsum(density)
    pts = np.linspace(cum_density[0], cum_density[-1], 2 * nb_bath + 1)
    bath_energies = np.interp(pts, cum_density, omegas)[1::2]
    return bath_energies
    
    
def fit_open_bath_physm(target_grea, nb_bath, spectral_tol, omega_tol=1e-3, omega_max=20., break_points=None, tol=1e-3):
    assert nb_bath % 2 == 0
    density = lambda w: 0*w + 1.
    # density = lambda w: 100 * target_dos(eps)**2 / ((w - eps)**2 + 100 * target_dos(eps)**2)
    weight = lambda w: 0 * w + 1.
    # weight = lambda w: 3 * target_dos(eps)**2 / ((w - eps)**2 + 3 * target_dos(eps)**2)
    eps_list = find_bath_energies(target_grea, 
                                  nb_bath // 2, 
                                  density, 
                                  spectral_tol=spectral_tol)
    v_list, dissip_rate, chi_sqr = find_best_dissip_and_v(target_grea, 
                                                          eps_list, 
                                                          weight, 
                                                          omega_max=omega_max, 
                                                          break_points=break_points, 
                                                          tol=tol)
    eps_absorb = eps_list.copy()
    eps_emitt = -eps_list
    v_absorb = np.sqrt(v_list / (2 * dissip_rate))
    v_emitt = v_absorb
    
    return eps_absorb, eps_emitt, v_absorb, v_emitt, dissip_rate, chi_sqr

    
def plot_fit(target_func, eps_list, v_list, dissip_rate, omega_max=20.):
    lorentzian_list = [lorentzian(eps, dissip_rate) for eps in eps_list]

    x = np.linspace(-omega_max, omega_max, 300)
    plt.plot(x, target_func(x), 'k-')
    plt.plot(x, sum(v * lor(x) for v, lor in zip(v_list, lorentzian_list)), 'r-')
    for lor, v in zip(lorentzian_list, v_list):
        plt.plot(x, v * lor(x), 'r-', lw=1, alpha=0.3)
    plt.plot(eps_list, np.zeros_like(eps_list), '^k', alpha=0.3)

    plt.show()
    
