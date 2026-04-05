import numpy as np
from hmardic.hmm import PoissonHMM

def test_hmm_runs():
    counts = np.array([0,0,1,0,10,12,9,0,0], dtype=float)
    lambdas = np.ones_like(counts) * 1.0
    hmm = PoissonHMM(counts, lambdas, max_iter=5)
    hmm.baum_welch_train()
    states = hmm.viterbi()
    assert states.shape[0] == counts.shape[0]
