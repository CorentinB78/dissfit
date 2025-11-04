from qat.core import Observable as Obs
from qat.lang import Program, X, QRoutine
from qat.fermion import ElectronicStructureHamiltonian
from qat.hardware import HardwareModel, GatesSpecification, DefaultGatesSpecification, DefaultHardwareModel
from qat.quops import ParametricAmplitudeDamping, ParametricGateNoise
from qat.qpus import NoisyQProc

import numpy as np
import toolbox as tb
# from matplotlib import pyplot as plt
# from datetime import datetime

from quantum_algo.correlation_circuit import circuits_for_correlator, evaluate_circuits
from quantum_algo.lindblad_noisy_algo import make_hardware_model, noise_as_a_resource_simulation
from quantum_algo.resonnant_model import make_resonnant_model_ham, make_resonnant_model_hpq
# from bath_fit.closed_bath import closed_bath_fit
from bath_fit.open_bath import fit_open_bath_physm#, plot_fit

from toolbox.free_fermions import SlaterDetState, free_fermions_observable, free_fermions_greater, free_fermions_observable_lindblad, free_fermions_greater_lindblad
# from toolbox.free_fermions import find_gibbs_state_rdm_free_fermions

### Parameters

coupling = 0.6
eps = 0.5
beta = 1.0
half_bandwidth = 10.

### target hybridization functions

def target_dos_unnorm(omega):
    if abs(omega) < half_bandwidth:
        return 1. / (2 * half_bandwidth)
    else:
        return 0.0

def target_dos(omega):
    return coupling**2 * target_dos_unnorm(omega)

target_dos = np.vectorize(target_dos)

def target_reta(omega):
    """correct in the limit half_bandwidth -> +infty"""
    return -1j * np.pi * target_dos(omega)
    
def target_less(omega):
    return 2 * np.pi * tb.fermi(omega, 0., beta) * target_dos(omega)

def target_grea(omega):
    return 2 * np.pi * tb.fermi(omega, 0., -beta) * target_dos(omega)

# omegas = np.linspace(-15, 15, 300)

# plt.plot(omegas, target_dos(omegas))
# plt.show()
# plt.plot(omegas, target_less(omegas))
# plt.plot(omegas, target_grea(omegas))
# plt.show()


nb_bath = 8
tprime = 30.

### open bath fit

fit = fit_open_bath_physm(target_grea, 
                          nb_bath, 
                          spectral_tol=0.1 * coupling**2,
                          omega_max=half_bandwidth,
                          break_points=[-half_bandwidth, half_bandwidth],
                         )

eps_absorb, eps_emitt, v_absorb, v_emitt, dissip_rate, chi_sqr = fit
print("unused bath sites:", sum(np.abs(v_absorb) < 1e-10))
print("dissip rate =", dissip_rate)
print("chi^2 =", chi_sqr)

# plt.axvline(eps, c='k', ls=':', alpha=0.3)
# plot_fit(target_dos, np.append(eps_absorb, eps_emitt), np.append(v_absorb, v_emitt)**2 * dissip_rate / np.pi, dissip_rate)

# plt.axvline(eps, c='k', ls=':', alpha=0.3)
# plot_fit(target_less, eps_emitt, 2 * v_emitt**2 * dissip_rate, dissip_rate)

hpq_open = make_resonnant_model_hpq(eps, np.append(v_emitt, v_absorb), np.append(eps_emitt, eps_absorb))
ham_es = ElectronicStructureHamiltonian(hpq=hpq_open)

### preparing circuit for impurity + open bath

trotter_time_step = 0.3
T_1 = 3e4

gs = DefaultGatesSpecification()
for key in gs.gate_times.keys():
    if key not in ['I', 'C-I', 'D-I']:
        gs.gate_times[key] = 1.
gs.gate_times['CNOT'] = 10.
gs.gate_times['SWAP'] = 10.
gs.gate_times['CSIGN'] = 10.
gs.gate_times['C-Y'] = 10.
# gs.gate_times['Noise'] = lambda t: 10.
# gs.quantum_channels['Noise'] = ParametricAmplitudeDamping(1.)

modes_crea = list(range(1, len(eps_emitt) + 1))
modes_anni = list(range(len(eps_emitt) + 1, len(eps_emitt) + len(eps_absorb) + 1))
# print(modes_crea, modes_anni)
nb_ancillas = max(0, (nb_bath // 3) - 1)

out = noise_as_a_resource_simulation(ham_es, 
                                     modes_crea, 
                                     modes_anni, 
                                     dissip_rate, 
                                     trotter_time_step, 
                                     T_1, 
                                     gs.gate_times, 
                                     nb_ancillas=nb_ancillas)
evol, encoding, qubit_mapping, gate_times, _ = out

nb_qubits = evol(1e-10).arity
print("nb qubits:", nb_qubits)
hm = make_hardware_model(nb_qubits + 1, T_1, [qubit_mapping[m] for m in modes_crea + modes_anni], gate_times)
print("waiting time:", gate_times["Wait"])

crea = 0.5 * (Obs.x(0, nb_qubits) - 1j * Obs.y(0, nb_qubits))
anni = 0.5 * (Obs.x(0, nb_qubits) + 1j * Obs.y(0, nb_qubits))

state_prep = QRoutine(arity=nb_qubits)
for i in modes_crea:
    state_prep.apply(X, qubit_mapping[i])
state_prep.apply(encoding(), range(nb_qubits))

### run noisy simulation
with tb.walltime() as wt:

    time_list_noisy = np.linspace(0, 60., 21)[1::2]

    gf_grea_arr_noisy = np.empty((len(time_list_noisy),), dtype=complex)
    gf_grea_error_re = np.empty_like(time_list_noisy)
    gf_grea_error_im = np.empty_like(time_list_noisy)
    runtime_list = np.empty_like(time_list_noisy)

    # qpu = NoisyQProc(hardware_model=hm, sim_method='stochastic', use_GPU=True,
    #                  n_samples=100)
    qpu = NoisyQProc(hardware_model=hm, sim_method='deterministic-vectorized', use_GPU=True)

    for k, time in enumerate(time_list_noisy):
        print(f"#### time={time} ####", flush=True)
        with tb.walltime() as wt_2:

            circuit_list = circuits_for_correlator(1j * anni, crea, 
                                                   time + tprime, tprime, 
                                                   evol, state_prep, 
                                                   real_part_only=True, verbose=False)

            print(f"#qubits = {circuit_list[0][1].nbqbits}, #gates = {len(circuit_list[0][1])}, #circuits = {len(circuit_list)}", flush=True)
            corr, err_re, err_im = evaluate_circuits(circuit_list, nb_qubits, qpu=qpu, verbose=True)

        gf_grea_arr_noisy[k] = -1 * corr
        gf_grea_error_re[k] = err_re
        gf_grea_error_im[k] = err_im
        runtime_list[k] = wt_2.time.total_seconds()
        print(f"walltime t={time}:", wt_2.time, flush=True)

    del qpu  # cause lagging?

print()
print("walltime:", wt.time)

tb.save_1Darrays_txt(f"data/Nbath={nb_bath}_Nanc={nb_ancillas}_T1={T_1}/noisy_algo_step={trotter_time_step}_tprime={tprime}.dat",
                     "time; Re[G^>(t)]; error; runtime",
                     time_list_noisy, gf_grea_arr_noisy.real, gf_grea_error_re, runtime_list,
                     create_dir=True)