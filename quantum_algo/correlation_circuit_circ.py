from qat.core import Observable as Obs, Term, Circuit
from qat.lang import Program, X, Y, Z, H, S, RX, QRoutine, AbstractGate
from qat.qpus import LinAlg
from qat.plugins import ZeroNoiseExtrapolator, Display
from qat.lang import CSIGN
from itertools import product
from typing import List, Tuple, Optional
from math import sqrt
import numpy as np
from datetime import datetime

pauli_gate = {'X': X, 'Y': Y, 'Z': Z}

def wait_matrix():
    return np.eye(2)
wait_gate = AbstractGate("Wait", [], arity=1, matrix_generator = wait_matrix)
barrier = [AbstractGate(f"B{n}", [], arity=n) for n in range(3)]

def _circuit_for_simple_correlator(n_qubits: int, 
                                   u1: Term, 
                                   u2: Term, 
                                   t1: float, 
                                   t2: float, 
                                   time_evol_circ, 
                                   state_prep = None,
                                   real_part_only: bool = False,compiler = None) -> List[Tuple[complex, Circuit]]:
    """
    Prepare circuits for computing Tr[U2 E_{t2}[ U1 E_{t1}[rho_0]]]
    with E_t the evolution superop given by `time_evol_qroutine`,
    with rho_0 the state produced by `state_prep`,
    assuming t1, t2, >= 0.


    (Hadamard test)
    """
    assert t1 >= 0
    assert t2 >= 0
        
    coeff = complex(u1.coeff * u2.coeff)
    
    if len(u1.qbits) == 0 and len(u2.qbits) == 0:
        if real_part_only:
            return [(complex(coeff.real), None)]
        else:
            return [(coeff, None)]
    
    circuit_list = []
    
    # print(u2, '\t', u1)
    
    for real_part in [True, False]:
        coeff_out = coeff if real_part else -1j * coeff
        if real_part_only:
            coeff_out = complex(coeff_out.real)
            
        if coeff_out == 0.0:
            continue
            
        
        if state_prep is not None:
            if compiler is None : 
                prog = Program()
                prog.qalloc(n_qubits)
                circ1 = prog.to_circ()
                circ = circ1 +  state_prep
                circ.shift_qbits(1)
            else : 
                circ = state_prep.compile(compiler).circuit
        prog = Program()
        qubits = prog.qalloc(n_qubits + 1)
        prog.apply(H, qubits[0])
        if compiler is None : 
            circ += prog.to_circ()
        else : 
            circ += prog.to_circ().compile(compiler).circuit
        if t1 > 0.:
            circ2 = time_evol_circ(t1)
            circ2.shift_qbits(1)
            circ += circ2
        prog = Program()
        qubits = prog.qalloc(n_qubits + 1)
        for idx, pauli in zip(u1.qbits, u1.op):
            prog.apply(pauli_gate[pauli].ctrl(), qubits[0], qubits[idx+1])
        if compiler is None : 
            circ += prog.to_circ()
        else : 
            circ += prog.to_circ().compile(compiler).circuit
        if len(u2.qbits) > 0:
            if t2 > 0.:
                circ2 = time_evol_circ(t2)
                circ2.shift_qbits(1)
                circ += circ2
            prog = Program()
            qubits = prog.qalloc(n_qubits + 1)
            prog.apply(X, qubits[0])
            for idx, pauli in zip(u2.qbits, u2.op):
                prog.apply(pauli_gate[pauli].ctrl(), qubits[0], qubits[idx+1])
            prog.apply(X, qubits[0])

        if not real_part:
            prog.apply(S, qubits[0])

        prog.apply(H, qubits[0])
        if compiler is None : 
                circ += prog.to_circ()
        else : 
            circ += prog.to_circ().compile(compiler).circuit
        circuit_list.append((coeff_out, circ))
    return circuit_list


def circuits_for_correlator(op_a: Obs, 
                            op_b: Obs, 
                            t_a: float, 
                            t_b: float, 
                            time_evol_circ, 
                            state_prep: Optional[Circuit] = None,
                            real_part_only: bool = False,
                            verbose: bool = False, 
                            compiler = None) -> List[Tuple[complex, Circuit]]:
    """
    Produce a list of circuits with complex coefficients that can be used to compute
    <psi| op_a(t_a) op_b(t_b) |psi>
    
    If `time_evol_qroutine` yields a quantum channel E_t, this corresponds to computing
    Tr[op_a E_{t_a - t_b}[ op_b E_{t_b}[rho_0]]]               for t_a >= t_b
    Tr[op_b^dag E_{t_b - t_a}[ op_a^dag E_{t_a}[rho_0]]]^*     for t_a <= t_b

    Args:
        time_evol_qroutine (Callable float -> QRoutine): the float parameter represents time
    """
    assert op_a.nbqbits == op_b.nbqbits
    n_qubits = op_a.nbqbits

    if t_b > t_a:
        #  <phi|AB|phi> = <phi|B^dag A^dag |phi>^*
        circuit_list = circuits_for_correlator(op_b.dag(), op_a.dag(), t_b, t_a, time_evol_circ, state_prep, real_part_only)
        for i in range(len(circuit_list)):  # conjugate the coefficients
            circuit_list[i] = (circuit_list[i][0].conjugate(), circuit_list[i][1])
        return circuit_list

    circuit_list = []

    op_a_terms = list(op_a.terms)
    op_b_terms = list(op_b.terms)
    if op_a.constant_coeff:
        op_a_terms.append(Term(op_a.constant_coeff, "", []))
    if op_b.constant_coeff:
        op_b_terms.append(Term(op_b.constant_coeff, "", []))

    assert t_a >= t_b >= 0.0

    for term_a, term_b in product(op_a_terms, op_b_terms):
        if verbose:
            print("Circuit for corr:", term_b, term_a)
        
        circuit_list.extend(_circuit_for_simple_correlator(n_qubits,
                                                           term_b,
                                                           term_a,
                                                           t_b,
                                                           t_a - t_b,
                                                           time_evol_circ,
                                                           state_prep,
                                                           real_part_only,
                                                           compiler=compiler))

    return circuit_list

def evaluate_circuits(circuit_list, n_qubits: int, qpu=None, compiler = None, verbose=False):
    out = 0.0
    err_re = 0.0
    err_im = 0.0
    start = datetime.now()

    if qpu is None:
        qpu = LinAlg()

    # simu_time = 0.
    p_list = []
    coeffs = []
    for coeff, circ in circuit_list:
        if circ is None:
            out += coeff
        else:
            nbshots = 5000
            #res = qpu.submit(circ.to_job('OBS', observable=Obs.z(0, n_qubits + 1), nbshots = nbshots))
            
            if compiler is not None:
                circ_compiled = circ.compile(compiler).circuit
            else:
                circ_compiled = circ
                #circ_compiled = circ.compile(compiler).circuit
            res = qpu.submit(circ_compiled.to_job('SAMPLE', qubits=[0], nbshots = nbshots))
            coeffs.append(coeff)
            if res[0].state.int == 0:
                p = res[0].probability
            else:
                p = res[1].probability
            p_list.append(p)
            obs = 2 * p - 1
            shot_noise = 2 * sqrt(p * (1 - p))
            #obs = res.value
            # print(coeff * res.value, '\t', coeff * res.error)
            out += coeff * obs
            err_re += (coeff.real * shot_noise)**2
            err_im += (coeff.imag * shot_noise)**2
            # simu_time += float(res.meta_data['simulation_time'])
    if verbose:
        print(f"Walltime: {datetime.now() - start}")#, Simulation time: {simu_time} s")
    return p_list, coeffs, sqrt(err_re), sqrt(err_im)

    


if __name__ == '__main__':
    from math import cos, sin
    from pytest import approx
    import numpy as np
    from scipy import linalg

    def time_evol_qrout(time):
        qr = QRoutine(arity=1)
        qr.apply(RX(0.1 * time), 0)
        return qr

    X_mat = np.array([[0, 1], [1, 0]], dtype=float)

    def test_1(t1, t2):
        circuit_list = circuits_for_correlator(Obs.z(0, 1), Obs.z(0, 1), t1, t2, time_evol_qrout)

        res, _, _ = evaluate_circuits(circuit_list, 1)
        # print(res)
        ref_res = cos(0.1 * (t1 - t2))
        # print(ref_res)
        assert res == approx(ref_res)

    def test_2(t1, t2):
        circuit_list = circuits_for_correlator(1j * Obs.z(0, 1), Obs.z(0, 1), t1, t2, time_evol_qrout)

        res, _, _ = evaluate_circuits(circuit_list, 1)
        # print(res)
        ref_res = 1.j * cos(0.1 * (t1 - t2))
        # print(ref_res)
        assert res == approx(ref_res)

    def test_3(t1, t2):
        A = Obs.x(0, 1) + (0.8 + 0.2j) * Obs.y(0, 1) - 0.1 * Obs.z(0, 1) + 0.02
        B = 0.2 * Obs.x(0, 1) - (0.3 + 0.5j) * Obs.y(0, 1) - 0.9 * Obs.z(0, 1) - 0.03j
        circuit_list = circuits_for_correlator(A, B, t1, t2, time_evol_qrout)

        # for (coeff, circ) in circuit_list:
        #     print()
        #     print(coeff)
        #     circ.display()

        res, _, _ = evaluate_circuits(circuit_list, 1)
        print(res)
        ref_res = linalg.expm(0.05j * t1 * X_mat) @ A.to_matrix(False) @ \
            linalg.expm(-0.05j * (t1 - t2) * X_mat) @ B.to_matrix(False) @ linalg.expm(-0.05j * t2 * X_mat)
        ref_res = ref_res[0, 0]
        print(ref_res)
        assert res == approx(ref_res)

    test_1(t1=2.0, t2=0.5)
    test_2(t1=2.0, t2=0.5)
    test_3(t1=2.0, t2=0.5)
    test_1(t1=0.5, t2=2.0)
    test_2(t1=0.5, t2=2.0)
    test_3(t1=0.5, t2=2.0)
