from qat.core import Term
from qat.fermion import FermionHamiltonian
from typing import List
import numpy as np

def make_resonnant_model_ham(eps: float, v_list: List[complex], eps_list: List[float]):
    n_modes = 1 + len(v_list)
    assert len(v_list) == len(eps_list)
    
    term_list = [Term(eps, "Cc", [0, 0])]
    for i in range(len(v_list)):
        mode = i + 1
        term_list.append(Term(v_list[i], "Cc", [0, mode]))
        term_list.append(Term(np.conj(v_list[i]), "Cc", [mode, 0]))
        term_list.append(Term(eps_list[i], "Cc", [mode, mode]))
    
    return FermionHamiltonian(n_modes, term_list)

def make_resonnant_model_hpq(eps: float, v_list: List[complex], eps_list: List[float]):
    n_modes = 1 + len(v_list)
    assert len(v_list) == len(eps_list)
    
    hpq = np.zeros((n_modes, n_modes), dtype=complex)
    hpq[0, 0] = eps
    for i in range(1, len(v_list) + 1):
        hpq[i, i] = eps_list[i - 1]
        hpq[i, 0] = v_list[i - 1]
        hpq[0, i] = np.conj(v_list[i - 1])
    
    return hpq
