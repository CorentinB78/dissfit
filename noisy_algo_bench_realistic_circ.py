from qat.core import Observable as Obs
from qat.lang import Program, X, QRoutine, CSIGN, AbstractGate, RX, RZ
from qat.fermion import ElectronicStructureHamiltonian
from qat.hardware import (
    HardwareModel,
    GatesSpecification,
    DefaultGatesSpecification,
    DefaultHardwareModel,
)
from qat.quops import ParametricAmplitudeDamping, ParametricGateNoise
from qat.qpus import NoisyQProc
import time as time_module
from qat.noisy.noisy_circuit import total_duration
from qat.plugins import Nnizer, QuameleonPlugin, Remap, Display
import numpy as np

# import toolbox as tb
# from matplotlib import pyplot as plt
# from datetime import datetime

from quantum_algo.correlation_circuit_circ import (
    circuits_for_correlator,
    evaluate_circuits,
)
from quantum_algo.lindblad_noisy_algo_circuit import (
    make_hardware_model,
    noise_as_a_resource_simulation,
)
from quantum_algo.resonnant_model import (
    make_resonnant_model_ham,
    make_resonnant_model_hpq,
)
from qat.qpus import IQMQPU
from qat.iqmqpu.interop.converters import qlm_to_iqm

# from bath_fit.closed_bath import closed_bath_fit
from bath_fit.open_bath import fit_open_bath_physm  # , plot_fit

import sys
# IMPORTS TO BE FIXED
sys.path.append("/home/mplazanet/qat/qat-fermion/qat/plugins/")
from zero_noise_extrapolator import ZeroNoiseExtrapolator
from zero_noise_extrapolator import insert_ids

sys.path.append("/home/mplazanet/qat/qat-fermion/misc/")
sys.path.append("/home/mplazanet/qat/qat-iqmqpu/notebooks/")
sys.path.append("/home/mplazanet/qat/qat-iqmqpu/python/qat/iqmqpu/interop/")

sys.path.append(
    "home/mplazanet/qat/qat-fermion/misc/scripts/dissipative_fit/garnet-benchmarking-data.json"
)
from iqm_hwmodel import get_Garnet_hwmodel
from iqm_compiler import get_IQM_Compiler
from iqm_compiler_2 import get_IQM_Compiler as get_IQM_Compiler_2
from free_fermions import (
    SlaterDetState,
    free_fermions_observable,
    free_fermions_greater,
    free_fermions_observable_lindblad,
    free_fermions_greater_lindblad,
    fermi,
)

# from toolbox.free_fermions import find_gibbs_state_rdm_free_fermions

### Parameters

coupling = 0.6
eps = 0.5
beta = 1.0
half_bandwidth = 10.0

### target hybridization functions

"""
def target_dos_unnorm(omega):
    if abs(omega) < half_bandwidth:
        return 1.0 / (2 * half_bandwidth)
    else:
        return 0.0


"""




"""

def target_dos(omega):
    return coupling/ ( coupling**2 + omega**2)
target_dos = np.vectorize(target_dos)


def target_reta(omega):
    correct in the limit half_bandwidth -> +infty
    return -1j * np.pi * target_dos(omega)


def target_less(omega):
    return 2 * np.pi * fermi(omega, 0.0, beta) * target_dos(omega)


def target_grea(omega):
    return 2 * np.pi * fermi(omega, 0.0, -beta) * target_dos(omega)

"""
# omegas = np.linspace(-15, 15, 300)

# plt.plot(omegas, target_dos(omegas))
# plt.show()
# plt.plot(omegas, target_less(omegas))
# plt.plot(omegas, target_grea(omegas))
# plt.show()


nb_bath = 1
tprime = 0.0
### open bath fit
"""
fit = fit_open_bath_physm(
    target_grea,
    nb_bath,
    spectral_tol=0.1 * coupling**2,
    omega_max=half_bandwidth,
    break_points=[-half_bandwidth, half_bandwidth],
)

eps_absorb, eps_emitt, v_absorb, v_emitt, dissip_rate, chi_sqr = fit"""
eps_emitt = np.array([-0.5])
eps_absorb = np.array([])
v_absorb = np.array([])
v_emitt = np.array([1])
dissip_rate = 0.2
print("unused bath sites:", sum(np.abs(v_absorb) < 1e-10))
print("dissip rate =", dissip_rate)
# print("chi^2 =", chi_sqr)

# plt.axvline(eps, c='k', ls=':', alpha=0.3)
# plot_fit(target_dos, np.append(eps_absorb, eps_emitt), np.append(v_absorb, v_emitt)**2 * dissip_rate / np.pi, dissip_rate)

# plt.axvline(eps, c='k', ls=':', alpha=0.3)
# plot_fit(target_less, eps_emitt, 2 * v_emitt**2 * dissip_rate, dissip_rate)

hpq_open = make_resonnant_model_hpq(
    eps, np.append(v_emitt, v_absorb), np.append(eps_emitt, eps_absorb)
)
ham_es = ElectronicStructureHamiltonian(hpq=hpq_open)

### preparing circuit for impurity + open bath

trotter_time_step = 0.3
T_1 = 50.86

"""gs = DefaultGatesSpecification()
for key in gs.gate_times.keys():
    if key not in ["I", "C-I", "D-I"]:
        gs.gate_times[key] = 1.0
gs.gate_times["CNOT"] = 10.0
gs.gate_times["SWAP"] = 10.0
gs.gate_times["CSIGN"] = 10.0
gs.gate_times["C-Y"] = 10.0"""
# gs.gate_times['Noise'] = lambda t: 10.
# gs.quantum_channels['Noise'] = ParametricAmplitudeDamping(1.)

modes_crea = list(range(1, len(eps_emitt) + 1))
modes_anni = list(range(len(eps_emitt) + 1, len(eps_emitt) + len(eps_absorb) + 1))
# print(modes_crea, modes_anni)
nb_ancillas = 0  # max(0, (nb_bath // 3) - 1)
ideal = False
PureDephasing = True
measurement_error = True
DepolarizingNoise = True
UnwantedAmplitudeDamping = True
perfect_wait = False
ZNE = False
real = True
use_dd = True
hm, garnet_hw, shifted_topology, get_wait_channels = get_Garnet_hwmodel(
    qubits=[15, 19, 20],
    return_shifted_topology=True,
    ideal=ideal,
    PureDephasing=PureDephasing,
    measurement_error=measurement_error,
    DepolarizingNoise=DepolarizingNoise,
    UnwantedAmplitudeDamping=UnwantedAmplitudeDamping,
    perfect_wait=perfect_wait,
)


"""gs = DefaultGatesSpecification()
for key in gs.gate_times.keys():
    if key not in ["I", "C-I", "D-I"]:
        gs.gate_times[key] = 1.0
gs.gate_times["CNOT"] = 10.0
gs.gate_times["SWAP"] = 10.0
gs.gate_times["CSIGN"] = 10.0
gs.gate_times["C-Y"] = 10.0"""


# compilers to be passed to linblad_noisy_algo_circuit in order to compile between barriers, and to the correlation circuit generator
# This one is for the circuit without Hadamard qubit (shifted topology) and to be passed to noise_as_a_resource_simulation
compiler1 = None
# this one is for circuits_for_correlators
compiler2 = None
out = noise_as_a_resource_simulation(
    ham_es,
    modes_crea,
    modes_anni,
    dissip_rate,
    trotter_time_step,
    T_1,
    # gs.gate_times,
    hm.gates_specification.gate_times,
    nb_ancillas=nb_ancillas,
    compiler=compiler1,
)
evol, encoding, qubit_mapping, gate_times, _ = out
hm.gates_specification.gate_times = gate_times
nb_qubits = evol(1e-10).nbqbits
print("nb qubits:", nb_qubits)


crea = 0.5 * (Obs.x(0, nb_qubits) - 1j * Obs.y(0, nb_qubits))
anni = 0.5 * (Obs.x(0, nb_qubits) + 1j * Obs.y(0, nb_qubits))
prog = Program()
qbits = prog.qalloc(nb_qubits)
for i in modes_crea:
    prog.apply(X, qubit_mapping[i])
state_prep = prog.to_circ()
state_prep += encoding()
# in case compilation is not done at the end of the circuit generation
if compiler1 is not None:
    state_prep = state_prep.compile(compiler1).circuit
### run noisy simulation
# with tb.walltime() as wt:

time_list_noisy = np.linspace(0, 6.0, 31)
# time_list_noisy = [0.1]
gf_grea_arr_noisy = np.empty((len(time_list_noisy),), dtype=complex)
gf_grea_error_re = np.empty_like(time_list_noisy)
gf_grea_error_im = np.empty_like(time_list_noisy)
runtime_list = np.empty_like(time_list_noisy)

# Putting no wait time
gate_times.update({"Wait": 0})

# Generating the wait noise channels corresponding to Wait gates with the given gate times
Wait_channels = get_wait_channels(gate_times)
hm.gate_noise.update({"Wait": Wait_channels})


qpu =NoisyQProc(
    hardware_model=hm,
    sim_method="deterministic-vectorized",
    use_GPU=False,
)
if real:
    qpu = IQMQPU(
        token="X3s2R9mOOyiaeWg8n9iQRNu94lEczscXnCF89JaGX/QGhmbpifN8eYAAj4Ju7WMU",
        wait_time=0, # If there are waits this should be set to gate_times["Wait"]
        mapping=[15, 19, 20],
        use_dd=use_dd,
    )
if ZNE: # only for ZNE extrapolations without keeping the trace
    def PRX_matrix(theta, phi):
        return np.array(
            [
                [np.cos(theta / 2), -np.exp(-1j * phi) * np.sin(theta / 2)],
                [np.exp(1j * phi) * np.sin(theta / 2), np.cos(theta / 2)],
            ]
        )

    PRX = AbstractGate("PRX", [float, float], arity=1, matrix_generator=PRX_matrix)
    ZNE_plugin = ZeroNoiseExtrapolator(
        n_ins=3, extrap_gates=[CSIGN], extrap_method="quadratic"
    )
    qpu = ZNE_plugin |qpu

# Since we want to perform ZNE on each component we keep each one and their corresponding coefficients (to reconstruct the Green's function at the end)
gf_grea_arr_noisy_1 = np.empty((len(time_list_noisy)))
gf_grea_arr_noisy_2 = np.empty((len(time_list_noisy)))
gf_grea_arr_noisy_3 = np.empty((len(time_list_noisy)))
gf_grea_arr_noisy_4 = np.empty((len(time_list_noisy)))
coeffs_1 = []
coeffs_2 = []
coeffs_3 = []
coeffs_4 = []

# Number of insertions for ZNE
n_ins = 0

compiler = get_IQM_Compiler(qubits=[15, 19, 20])
for k, time in enumerate(time_list_noisy):
    print(f"#### time={time} ####", flush=True)
    # with tb.walltime() as wt_2:

    circuit_list = circuits_for_correlator(
        1j * anni,
        crea,
        time + tprime,
        tprime,
        evol,
        state_prep,
        real_part_only=True,
        verbose=False,
        compiler=compiler2,
    )
    if n_ins > 0:
        circuit_list_ZNE = [
            (coeff, insert_ids(circ=circ.compile(compiler).circuit, gates=[CSIGN], n_ins=n_ins))
            for coeff, circ in circuit_list
        ]
    else:
        circuit_list_ZNE = [(coeff, circ.compile(compiler).circuit) for coeff, circ in circuit_list]
    print("wait time:", gate_times["Wait"])
    circ = circuit_list[0][1].compile(compiler).circuit
    print(
        f"#qubits = {circuit_list[0][1].nbqbits}, #gates = {len(circ)}, #circuits = {len(circuit_list)}",
        flush=True,
    )
    p_list, coeffs, err_re, err_im = evaluate_circuits(
        circuit_list_ZNE,
        nb_qubits,
        qpu=qpu,
        compiler=None,
        verbose=False,
    )
    gf_grea_arr_noisy_1[k] = p_list[0]
    gf_grea_arr_noisy_2[k] = p_list[1]
    gf_grea_arr_noisy_3[k] = p_list[2]
    gf_grea_arr_noisy_4[k] = p_list[3]
    coeffs_1.append(coeffs[0])
    coeffs_2.append(coeffs[1])
    coeffs_3.append(coeffs[2])
    coeffs_4.append(coeffs[3])

print()
if ideal:
    name = f"/home/mplazanet/qat/qat-fermion/misc/scripts/dissipative_fit/data/ZNE/noisy_algo_step={trotter_time_step}_tprime={tprime}-circ-ideal"
    # name = f"/home/mplazanet/qat/qat-fermion/misc/scripts/dissipative_fit/data/Nbath={nb_bath}/02-06/impurity_Z-ideal"
elif real:
    if use_dd:
        name = f"/home/mplazanet/qat/qat-fermion/misc/scripts/dissipative_fit/data/ZNE/noisy_algo_step={trotter_time_step}_tprime={tprime}-circ-real-dd"
        # name = f"/home/mplazanet/qat/qat-fermion/misc/scripts/dissipative_fit/data/Nbath={nb_bath}/02-06/impurity_Z-real-dd"
    else:
        name = f"/home/mplazanet/qat/qat-fermion/misc/scripts/dissipative_fit/data/ZNE/noisy_algo_step={trotter_time_step}_tprime={tprime}-circ-real"
        # name = f"/home/mplazanet/qat/qat-fermion/misc/scripts/dissipative_fit/data/Nbath={nb_bath}/02-06/impurity_Z-real"
else:
    name = f"/home/mplazanet/qat/qat-fermion/misc/scripts/dissipative_fit/data/ZNE/noisy_algo_step={trotter_time_step}_tprime={tprime}-circ-iqmhw"
    # name = f"/home/mplazanet/qat/qat-fermion/misc/scripts/dissipative_fit/data/Nbath={nb_bath}/02-06/impurity_Z-iqmhw"
    if perfect_wait:
        name += "-perfect_wait"
    if not measurement_error:
        name += "-no_meas_error"
    if not DepolarizingNoise:
        name += "-no_depo"
    if not PureDephasing:
        name += "-no_PD"
    if not UnwantedAmplitudeDamping:
        name += "-no_AD"
    if ZNE:
        name += "-ZNE"
name += f"-n_ins={n_ins}.dat"
# print("walltime:", wt.time)
np.savetxt(
    name,
    np.array(
        [
            time_list_noisy,
            gf_grea_arr_noisy_1,
            gf_grea_arr_noisy_2,
            gf_grea_arr_noisy_3,
            gf_grea_arr_noisy_4,
            np.array(coeffs_1),
            np.array(coeffs_2),
            np.array(coeffs_3),
            np.array(coeffs_4),
        ]
    ).T,
    header="time; Re[G^>(t)]_1; Re[G^>(t)]_2;Re[G^>(t)]_3;Re[G^>(t)]_4;coeff_1; coeff_2; coeff_3; coeff_4",
)


