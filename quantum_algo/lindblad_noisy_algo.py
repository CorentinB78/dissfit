from math import ceil
from typing import List, Dict, Tuple, Optional
import numpy as np
from qat.core import Observable as Obs, Term
from qat.lang import AbstractGate, qrout, X, Y, Z, H, S, RX, RY, RZ, CNOT, CSIGN, PH, SWAP, QRoutine
from qat.hardware import HardwareModel, GatesSpecification, DefaultGatesSpecification
from qat.quops import QuantumChannelKraus, ParametricAmplitudeDamping, ParametricGateNoise
from qat.noisy.noisy_circuit import total_duration
from qat.fermion import ElectronicStructureHamiltonian
# from qat.fermion.trotterisation import _number_operator_jw, _excitation_operator_jw  # TODO use these instead?
from math import pi


def _number_op_jw(energy: float, delta_t: float, qubit: int, nbqubits: int):
    qr = QRoutine(arity=nbqubits)
    if energy != 0.:
        qr.apply(PH(-energy * delta_t), qubit)
    return qr

def _excitation_op_jw(hopping: float, delta_t: float, qubit0: int, qubit1: int, nbqubits: int):
    qr = QRoutine(arity=nbqubits)

    if hopping == 0.:
        return qr

    if qubit0 > qubit1:
        qubit0, qubit1 = qubit1, qubit0
        
    if qubit0 == qubit1:
        raise ValueError

    j, i = qubit0, qubit1
    qr.apply(H, j)
    qr.apply(H, i)
    for k in range(i - j):
        qr.apply(CNOT, i - k, i - k - 1)
    qr.apply(RZ(delta_t * hopping), j)
    for k in range(i - j):
        qr.apply(CNOT, j + k + 1, j + k)
    qr.apply(H, j)
    qr.apply(H, i)
    qr.apply(RX(-pi / 2), j)
    qr.apply(RX(-pi / 2), i)
    for k in range(i - j):
        qr.apply(CNOT, j + k, j + k + 1)
    qr.apply(RZ(delta_t * hopping), i)
    for k in range(i - j):
        qr.apply(CNOT, i - k - 1, i - k)
    qr.apply(RX(pi / 2), j)
    qr.apply(RX(pi / 2), i)

    return qr


def trotter_lindblad_noise(hamiltonian: ElectronicStructureHamiltonian,
                           jump_crea_coeffs: Dict[int, float],
                           jump_anni_coeffs: Dict[int, float],
                           trotter_time_step: float,
                           nb_ancillas: int,
                           with_barriers: bool = True,
                           with_noise_gates: bool = False,
                           with_outer_encoding: bool= True,
                           with_wait_gates: bool = True,
                          ) -> Tuple[QRoutine, QRoutine, List[int]]:
    """ does the Trotterization """

    # check there creation and annihilation jump ops are not applied on the same modes -> not supported yet
    assert len(set(jump_anni_coeffs.keys()).intersection(set(jump_crea_coeffs.keys()))) == 0
    crea_indices = list(jump_crea_coeffs.keys())
    anni_indices = list(jump_anni_coeffs.keys())

    assert np.all(hamiltonian.hpqrs == 0.)  # interaction not supported yet

    nb_modes_original = hamiltonian.nbqbits
    nb_imp_modes = 1
    nb_bath_modes = nb_modes_original - nb_imp_modes
    assert 0 <= nb_ancillas < nb_bath_modes
    nb_qubits = nb_imp_modes + nb_bath_modes + nb_ancillas
    
    ### blocks
    nb_blocks = nb_ancillas + 1  # last block has no ancilla
    block_sizes = [nb_bath_modes // nb_blocks + (1 if i < nb_bath_modes % nb_blocks else 0) for i in range(nb_blocks)]
    block_list = []  # bath indices grouped by block
    idx = 0
    for block_size in block_sizes:
        block_list.append([i + idx for i in range(block_size)])
        idx += block_size
        
    # print("block_list:", block_list)
    # print("size blocks:", block_sizes)
    assert sum(block_sizes) == nb_bath_modes        
    
    ### mappings
    qb = nb_imp_modes
    imp_qubit_map = [0]  # original impurity index -> impurity qubit
    bath_qubit_map = []  # original bath index -> bath qubit
    ancilla_qubit_map = []  # ancilla index -> ancilla qubit
    bath_ancilla_map = {}  # bath index -> ancilla qubit
    for block_idx, block in enumerate(block_list):
        for _ in block:
            bath_qubit_map.append(qb)
            qb += 1
        ancilla_qubit_map.append(qb if qb < nb_qubits else -1)
        for bath_idx in block:
            bath_ancilla_map[bath_idx] = ancilla_qubit_map[-1]
        qb += 1  # ancilla
        
    # print("bath_qubit_map:", bath_qubit_map)
    # print("ancilla_qubit_map:", ancilla_qubit_map)
    # print("bath_ancilla_map:", bath_ancilla_map)
    qubit_mapping = imp_qubit_map + bath_qubit_map
    
    block_qubit_list = [[bath_qubit_map[bath_idx] for bath_idx in block] + [bath_ancilla_map[block[0]]] for block in block_list]  # list of qubits grouped by block
    del block_qubit_list[-1][-1]  # remove last ancilla
    block_ancilla = [bath_ancilla_map[block[0]] for block in block_list]  # block index -> ancilla qubit
    
    imp_qubits = imp_qubit_map
    all_qubits = range(nb_qubits)
    assert(list(all_qubits) == sorted(imp_qubit_map + bath_qubit_map + ancilla_qubit_map[:-1]))
    
    hpq = hamiltonian.hpq.real
    
    # @qrout
    # def fSWAP():
    #     """Must be fermionic neighbors"""
    #     SWAP(0, 1)
    #     CSIGN(0, 1)

    def encoding(orig_mode: int, anc_qubit: int, bath_qubit: int, nb_qubits: int = nb_qubits):
        """"""
        assert -1 <= anc_qubit < nb_qubits
        assert 0 <= bath_qubit < nb_qubits
        assert anc_qubit != bath_qubit
        qr = QRoutine(arity=nb_qubits)
        
        if anc_qubit == -1:
            rng = range(bath_qubit + 1, nb_qubits)
        elif anc_qubit < bath_qubit:
            rng = range(anc_qubit + 1, bath_qubit)
        else:
            rng = range(bath_qubit + 1, anc_qubit)
        
        if orig_mode in crea_indices:
            if anc_qubit != -1:
                qr.apply(CNOT, bath_qubit, anc_qubit)
            for k in rng:
                qr.apply(CSIGN, bath_qubit, k)
            qr.apply(X, bath_qubit)
        elif orig_mode in anni_indices:
            if anc_qubit != -1:
                qr.apply(CNOT, bath_qubit, anc_qubit)
            for k in rng:
                qr.apply(CSIGN, bath_qubit, k)
        return qr
    
    def encoding_block(block_idx: int):
        qr = QRoutine(arity=nb_qubits)
        for bath_idx in block_list[block_idx]:
            bath_mode = bath_idx + nb_imp_modes
            qr.apply(encoding(bath_mode, bath_ancilla_map[bath_idx], qubit_mapping[bath_mode]), all_qubits)
        return qr
    
    def encoding_all():
        qr = QRoutine(arity=nb_qubits)
        for block_idx in range(len(block_sizes)):
            qr.apply(encoding_block(block_idx), all_qubits)
        return qr

    ### trotter step
    if with_barriers:
        barrier = [AbstractGate(f"B{n}", [], arity=n) for n in range(nb_qubits+1)]
    if with_noise_gates:
        noise_gate = AbstractGate("Noise", [float], arity=1)
    if with_wait_gates:
        wait_gate = AbstractGate("Wait", [], arity=1)

    def evolve_block(delta_t: float, block_idx: int):
        """
        """
        assert nb_imp_modes == 1
        block_qubits = block_qubit_list[block_idx]
        block = block_list[block_idx]
        first_bath_idx = block[0]
        qr = QRoutine(arity=nb_qubits)
        
        if with_barriers:
            qr.apply(barrier[nb_qubits](), all_qubits)
        qr.apply(encoding_block(block_idx).dag(), all_qubits)
        
        for bath_idx in block:
            bath_mode = bath_idx + nb_imp_modes
        
            if hpq[bath_mode, bath_mode] != 0.0:
                qr.apply(_number_op_jw(hpq[bath_mode, bath_mode], delta_t, 0, 1), [bath_qubit_map[bath_idx]])
            
            if hpq[0, bath_mode] != 0.0:
                qr.apply(_excitation_op_jw(hpq[0, bath_mode],
                                           delta_t,
                                           0,
                                           bath_idx - first_bath_idx + nb_imp_modes,
                                           len(block_qubits) + nb_imp_modes),
                         imp_qubits + block_qubits)
            
        qr.apply(encoding_block(block_idx), all_qubits)
        
        return qr
    
    def evolve_impurity(delta_t):
        assert nb_imp_modes == 1
        qr = QRoutine(arity=nb_qubits)
        if hpq[0, 0] != 0.0:
            qr.apply(_number_op_jw(hpq[0, 0], delta_t, 0, 1), imp_qubits)
            
        return qr
    
    def artificial_noise(delta_t):
        qr = QRoutine(arity=nb_qubits)
        
        for mode, gamma in jump_crea_coeffs.items():
            qr.apply(noise_gate(delta_t * gamma), qubit_mapping[mode])
        
        for mode, gamma in jump_anni_coeffs.items():
            qr.apply(noise_gate(delta_t * gamma), qubit_mapping[mode])
        
        return qr
    
    
    def wait():
        qr = QRoutine(arity=nb_qubits)
        
        for qb in range(nb_qubits):
            qr.apply(wait_gate(), qb)
        
        return qr
    
    def trotter_step_first_order(delta_t):
        qr = QRoutine(arity=nb_qubits)
        
        qr.apply(evolve_impurity(delta_t), all_qubits)
        
        for block_idx in range(len(block_list)):
            qr.apply(evolve_block(delta_t, block_idx), all_qubits)
            if block_idx < len(block_list) - 1:  # if not the last one
                qr.apply(CSIGN, [imp_qubits[0], block_ancilla[block_idx]])
                
        for block_idx in range(len(block_list)-2, -1, -1):
            qr.apply(CSIGN, [imp_qubits[0], block_ancilla[block_idx]])
        
        return qr
    
    def trotter_step_second_order(delta_t):
        qr = QRoutine(arity=nb_qubits)
        
        qr.apply(evolve_impurity(delta_t / 2), all_qubits)
        
        for block_idx in range(len(block_list) - 1):
            qr.apply(evolve_block(delta_t / 2, block_idx), all_qubits)
            qr.apply(CSIGN, [imp_qubits[0], block_ancilla[block_idx]])
                
        qr.apply(evolve_block(delta_t, len(block_list) - 1), all_qubits)
                
        for block_idx in range(len(block_list)-2, -1, -1):
            qr.apply(CSIGN, [imp_qubits[0], block_ancilla[block_idx]])
            qr.apply(evolve_block(delta_t / 2, block_idx), all_qubits)
        
        qr.apply(evolve_impurity(delta_t / 2), all_qubits)
        
        return qr
    
    trotter_step = trotter_step_first_order
    # trotter_step = trotter_step_second_order

    # hopping_step(0.1).display()

    ## TODO: interaction

    def full_evol(time: float) -> QRoutine:
        qr = QRoutine(arity=nb_qubits)
        n_trotter_steps = max(1, ceil(float(time) / trotter_time_step))
        dt = time / n_trotter_steps
        
        if with_outer_encoding:
            qr.apply(encoding_all(), all_qubits)
            
        for _ in range(n_trotter_steps):
            qr.apply(trotter_step(dt), all_qubits)
            if with_noise_gates:
                qr.apply(artificial_noise(dt), all_qubits)
            if with_wait_gates:
                qr.apply(wait(), all_qubits)
        
        if with_outer_encoding:
            if with_barriers:
                qr.apply(barrier[nb_qubits](), all_qubits)
            qr.apply(encoding_all().dag(), all_qubits)
        
        return qr
    
    return full_evol, encoding_all, qubit_mapping



def tensor_prod_quantum_channels(qchan1, qchan2):
    kraus_operators = [np.kron(k1, k2) for k1 in qchan1.kraus_operators for k2 in qchan2.kraus_operators]
    return QuantumChannelKraus(kraus_operators)

class TensorParamQuantumChannel:
    
    def __init__(self, pqchan1, pqchan2):
        self.pqchan1 = pqchan1
        self.pqchan2 = pqchan2
        
    def __call__(self, *args, **kwargs):
        return tensor_prod_quantum_channels(self.pqchan1(*args, **kwargs), self.pqchan2(*args, **kwargs))


def make_hardware_model(nb_qubits: int, T_ampl_damping: float, noisy_qubits: List[int], gate_times: Dict[str, float]):
    """ Creates the HW model with amplitude damping only on some qubits """

    gs = DefaultGatesSpecification(gate_times)
    gs.quantum_channels['Wait'] = QuantumChannelKraus([np.eye(2)], name='Identity')
    
    one_qb_noiseless = ParametricAmplitudeDamping(np.inf)
    two_qb_noiseless = TensorParamQuantumChannel(one_qb_noiseless, one_qb_noiseless)
    one_qb_ad = ParametricAmplitudeDamping(T_ampl_damping)
    two_qb_ad_10 = TensorParamQuantumChannel(one_qb_ad, one_qb_noiseless)
    two_qb_ad_01 = TensorParamQuantumChannel(one_qb_noiseless, one_qb_ad)
    two_qb_ad_11 = TensorParamQuantumChannel(one_qb_ad, one_qb_ad)
    
    idle_noise = {qb: [one_qb_noiseless] for qb in range(nb_qubits)}
    for qb in noisy_qubits:
        idle_noise[qb] = [one_qb_ad]
        
    one_qb_gates = ['H', 'X', 'Y', 'Z', 'S', 'RX', 'RY', 'RZ', 'PH', 'Wait']
    two_qb_gates = ['CNOT', 'CSIGN', 'C-Y', 'SWAP']
    all_gates = one_qb_gates + two_qb_gates
    gate_noise = {gate: {} for gate in all_gates}
    
    for qb in range(nb_qubits):
        if qb in noisy_qubits:
            for gate in one_qb_gates:
                gate_noise[gate][qb] = ParametricGateNoise(gs, gate, [one_qb_ad])
        else:
            for gate in one_qb_gates:
                gate_noise[gate][qb] = ParametricGateNoise(gs, gate, [one_qb_noiseless])
    
    for qb1 in range(nb_qubits):
        for qb2 in range(qb1):
            if qb1 in noisy_qubits:
                if qb2 in noisy_qubits:
                    for gate in two_qb_gates:
                        gate_noise[gate][(qb1, qb2)] = ParametricGateNoise(gs, gate, [two_qb_ad_11])
                        gate_noise[gate][(qb2, qb1)] = ParametricGateNoise(gs, gate, [two_qb_ad_11])
                else:
                    for gate in two_qb_gates:
                        gate_noise[gate][(qb1, qb2)] = ParametricGateNoise(gs, gate, [two_qb_ad_10])
                        gate_noise[gate][(qb2, qb1)] = ParametricGateNoise(gs, gate, [two_qb_ad_01])
            else:
                if qb2 in noisy_qubits:
                    for gate in two_qb_gates:
                        gate_noise[gate][(qb1, qb2)] = ParametricGateNoise(gs, gate, [two_qb_ad_01])
                        gate_noise[gate][(qb2, qb1)] = ParametricGateNoise(gs, gate, [two_qb_ad_10])
                else:
                    for gate in two_qb_gates:
                        gate_noise[gate][(qb1, qb2)] = ParametricGateNoise(gs, gate, [two_qb_noiseless])
                        gate_noise[gate][(qb2, qb1)] = ParametricGateNoise(gs, gate, [two_qb_noiseless])

    return HardwareModel(gates_specification=gs, gate_noise=gate_noise, idle_noise=idle_noise)

def noise_as_a_resource_simulation(hamiltonian: ElectronicStructureHamiltonian, 
                                   jump_crea_modes: List[int],
                                   jump_anni_modes: List[int],
                                   logic_noise_rate: float,
                                   trotter_time_step: float,
                                   T_ampl_damping: float,
                                   gate_times: Dict[str, float],
                                   nb_ancillas: int):
    """Adapts the waiting time to get the right noise


    Args:
        hamiltonian: input Hamiltonian

    Returns:
    """
    
    gate_times = gate_times.copy()
    gate_times['Wait'] = 0.0
    for k in range(len(hamiltonian.hpq) + nb_ancillas + 1):
        gate_times[f"B{k}"] = 1e-4
    
    hm = HardwareModel(GatesSpecification(gate_times, {}))
    
    evol_qr, encoding, qubit_mapping = trotter_lindblad_noise(hamiltonian, 
                                                              {i: logic_noise_rate for i in jump_crea_modes},
                                                              {i: logic_noise_rate for i in jump_anni_modes},
                                                              trotter_time_step=trotter_time_step,
                                                              nb_ancillas=nb_ancillas,
                                                              with_barriers=True,
                                                              with_noise_gates=False,
                                                              with_outer_encoding=False,
                                                              with_wait_gates=True)
    
    phys_time_5 = total_duration(evol_qr(5 * trotter_time_step).to_circ(), hm)
    phys_time_10 = total_duration(evol_qr(10 * trotter_time_step).to_circ(), hm)
    
    trotter_step_phys_time = (phys_time_10 - phys_time_5) / 5.
    print("Trotter step phys time:", trotter_step_phys_time)
    min_logic_noise_rate = trotter_step_phys_time / (T_ampl_damping * trotter_time_step)
    
    if logic_noise_rate < min_logic_noise_rate:
        raise ValueError(f"Logical noise rate {logic_noise_rate} is too low to achieve. With given input, it must be at least {min_logic_noise_rate}.")
        
    waiting_time = (logic_noise_rate - min_logic_noise_rate) * T_ampl_damping * trotter_time_step
    gate_times['Wait'] = waiting_time
    
    return evol_qr, encoding, qubit_mapping, gate_times, trotter_step_phys_time
    
