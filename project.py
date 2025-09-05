
from __future__ import annotations

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import csv


def beta_from_T(T_K: float) -> float:
    """Return beta = 1/(k_B T) in 1/eV."""
    k_B_J_per_kelvin = constants.Boltzmann      # J/K
    K_B_eV_per_kelvin = k_B_J_per_kelvin * 6.242e18  # eV/K
    return 1.0 / (K_B_eV_per_kelvin * T_K)

def rate_from_deltaG(delta_G_eV: float, beta: float, k_h: float) -> float:
    """
    Detailed balance:
    - downhill (ΔG <= 0): k = k_h
    - uphill   (ΔG > 0):  k = k_h * exp(-β ΔG)
    """
    return k_h if delta_G_eV <= 0.0 else (k_h * np.exp(-beta * delta_G_eV))



def adjacency_matrix(edges_csv: str, T_K: float = 300.0, tau_ps: float = 10.0):
    """
    This function loads information about the system.The information is contained within a single 
    CSV file that contains the donor and acceptor nodes and also delta G values or the rate of transfer.
    The output of the function is the weighted A matrix.
    """
    beta = beta_from_T(T_K)
    k_h = 1.0 / tau_ps  # ps^-1

    edges = []          # list of (i, j, rate)
    nodes = -1 # could be any negative number

    with open(edges_csv, 'r', newline="") as f:
        reader = csv.DictReader(f)
        required = {"donor", "acceptor", "delta_G_eV", "rate"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV must have columns: donor, acceptor, delta_G_eV, rate")
        for row in reader:
            i = int((row["donor"]).strip())
            j = int((row["acceptor"]).strip())
            raw_rate = (row.get("rate") or "").strip()
            raw_dG   = (row.get("delta_G_eV") or "").strip()

            if raw_rate != "":
                r = float(raw_rate)
            elif raw_dG != "":
                r = rate_from_deltaG(float(raw_dG), beta, k_h)
            else:
                raise ValueError("Each row needs either rate or delta_G_eV")

            if i == j:
                raise ValueError(f"Self-loop not allowed at node {i}")
            if r < 0:
                raise ValueError(f"Rate must be non-negative for edge {i}->{j}")

            edges.append((i, j, r))
            nodes = max(nodes, i, j) 

    if nodes < 0:
        raise ValueError("No edges found in CSV")

    N = nodes + 1  
    A = np.zeros((N, N), dtype=float)
    for i, j, r in edges:
        A[i, j] += r  

    labels = [f"P{k+1}" for k in range(N)]
    return A, labels

def make_K_matrix(A: np.ndarray) -> np.ndarray:
    """
    Getting K from A:
    - First take the transpose this shows the influx to each node.
    - Then to calculate the outflow take the sums of each of the rows and place 
    them on the diagonals of a matrix (every other entry = 0)
    - then this outflow matrix from the transposed A matrix.
    """
    K_influx = np.transpose(A)
    row_sum = np.sum(A, axis = 1)
    K_out = np.diag(row_sum)
    K = K_influx - K_out
    return K

def calc_evolution(K: np.ndarray, P0: np.ndarray, t_ps: np.ndarray) -> np.ndarray:
    """
    Esssentially doing  P(t) = V exp(Λ t) V^{-1} P0. as per previous versions but now in a  
    slightly better way. (much more memory efficient)
    Breakdown:
    - convert the vector P0 into eigenbasis (eigencoordinates)
    - Then e^eigenvalues * t
    - Convert back to our original basis 
    Returns array of shape (len(t_ps), N).
    """
    w, V = np.linalg.eig(K)
    Vinv = np.linalg.inv(V)
    c0 = Vinv @ P0  # this puts P0 into the eigenbasis

    T = len(t_ps)
    N = K.shape[0]
    out = np.zeros((T, N), dtype=float)
    for k, t in enumerate(t_ps):
        ew = np.exp(w * t) # no need to regenerate the matrix exponential per timestep
        Pt = V @ (ew * c0) # converts back into the original basis
        out[k] = Pt.real
    return out

def graphing(t_ps: np.ndarray, P_t: np.ndarray, labels=None) -> None:
    """
    This function graphs the cal_evolution one.
    the input is an array where each time point corresponds to a new row, with the amount of nodes
    being the number of columns. (This is a far easier way of doing things than was previously implemented)
    """
    if labels is None:
        labels = [f"P{i + 1}" for i in range(P_t.shape[1])] # this is a failsafe
    for i in range(P_t.shape[1]):  # this just indexes each of the columns (rows would have been .shape[0])
        # The x axis is time in picoseconds and will be determined in another function
        # the number of rows in P_t must match the number of time points otherwise the function will not run
        # we then want to iterate over each of the columns 
        plt.plot(t_ps, P_t[:, i], label=labels[i])
    plt.xlabel("Time (ps)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True) # good to have a grid instead of blank space
    plt.tight_layout()
    plt.show()


def main():
    """
     reads edges from 'edges.csv' in the current folder (or change the filename below). 
     The function then constructs P0 and creates the timestamps, then calls the relevant functions

    """
    edges_path = "edges.csv"   # change this line if your CSV is named differently

    # Build A, K
    A, labels = adjacency_matrix(edges_path, T_K=300.0, tau_ps=10.0)
    K = make_K_matrix(A)

    # Initial condition (one-hot at node 0)
    N = A.shape[0] # number of rows of A so we can think of this as the number of nodes
    P0 = np.zeros(N, dtype=float)
    P0[0] = 1.0

    # Time grid
    t_ps = np.linspace(0.0, 100.0, 1000)

    # Evolve and plot
    P_t = calc_evolution(K, P0, t_ps)
    graphing(t_ps, P_t, labels=labels)

if __name__ == "__main__":
    main()
