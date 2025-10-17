
from __future__ import annotations

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import csv
import math
from scipy import linalg

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

def FWHM() -> float:
    """
    Calculate the full width half max value. The reason for doing this is so that
    we can calculate w (or the standard deviation in for A_p(lambda))

    the file named T01.absorption_Qyonly.txt is data for the Qy absorption band of chlorophyll A.
    """
    x, y = np.loadtxt("T01.absorption_Qyonly.txt", skiprows=1,unpack=True)

    peak = np.argmax(y) # This is for indexing
    # the interpolate function is going to look like this:
    # np.interp(new_y, y, x)
    new_y = np.max(y) / 2 # half max
    left = np.interp(new_y, y[:peak+1], x[:peak+1]) # interpolation
    right = np.interp(new_y, y[peak:][::-1], x[peak:][::-1]) # interpolation

    fwhm = right - left
    return fwhm

def A_calc() -> np.ndarray:
    """
    Small function that calculates A_p(lambda). See below for the equation (maybe at later date include 
    latex code for the formula)
    """
    fwhm = FWHM()
    x, y = np.loadtxt("T01.absorption_Qyonly.txt", skiprows=1,unpack=True)
    w = fwhm / 2.35482
    peak_y = np.argmax(y)
    peak = x[peak_y]
    list_A_vals = []

    for wavelengths in x: # not sure how good it is to use the loop when dealing with vectorized quantities but it works
                          # for now
        first_part = 1 / (w * math.sqrt(2*math.pi))
        exponetial = np.exp(-(wavelengths - peak)**2 / (2*w**2))
        A = first_part * exponetial
        list_A_vals.append(A)

    array_A = np.asarray(list_A_vals)
    return array_A   


def check(A:np.ndarray) -> None:

    """
    A_p(lambda) must integrate to have an overall area with 1. A_p(lambda) is dimensionless because it is expressed in nm^-1
    and therefore when we integrate it is being multiplied by d lambda and thus the units cancel.
    However this function is essentially doing nothing at the moment, i am not even sure if it checks the validity 
    in the correct manner.
    """

    x, _ = np.loadtxt("T01.absorption_Qyonly.txt",skiprows=1, unpack=True)
    integrated_A = np.trapezoid(A, x)

    if np.isclose(integrated_A, 1.0 , rtol=1e-2) == True:
        print("This is normalized and ready to use")
        return None
    else:
        raise ValueError("The value for A when integrated is not 1")
        
    

def Gamma(A: np.ndarray, h: float=constants.h, c: float=constants.c, N: int=100, sigma: float=1e-20) -> float: 
    
   """
   Units:
   
    - cross section (chlorophyll) = 10^-20 m^2
    - spectral irradience = Wm^-2 nm^-1 -> Js^-1 m^-2 nm^-1
    - lambda / h*C = J^-1
    - A_p(lambda) = dimensionless
    - Number of pigments = dimensionless 

   """ 
   check(A)
   lam_pig, _ = np.loadtxt("T01.absorption_Qyonly.txt",unpack=True, skiprows=1)
   lam_star, f = np.loadtxt("5800K.txt", unpack=True)

   lam_pig = np.asarray(lam_pig, dtype=float).ravel()
   lam_star = np.asarray(lam_star, dtype=float).ravel()
   f = np.asarray(f, dtype=float).ravel()  # make shape (n,)
   
   lam_star_m = lam_star * 1e-9
   A_star = np.interp(lam_star,lam_pig,A)
   photon_eq = lam_star_m / (h*c)
   integrand = photon_eq * f * A_star
   photons = np.trapezoid(integrand, lam_star)
   gamma = float((N) * sigma * photons)
   return gamma

def adjacency_matrix(edges_csv: str, T_K: float = 300.0, tau_s: float = 1e-11):
    """
    This function loads information about the system.The information is contained within a single 
    CSV file that contains the donor and acceptor nodes and also delta G values or the rate of transfer.
    The output of the function is the weighted A matrix.
    """
    beta = beta_from_T(T_K)
    k_h = 1.0 / tau_s  # s^-1
    gamma_s = Gamma(A_calc(),h=constants.h, c=constants.c, N=100, sigma=1e-20) # units of gamma are s^-1
    

    edges = []          # list of (i, j, rate)
    nodes = -1 # could be any negative number

    with open(edges_csv, 'r', newline="") as f:
        reader = csv.DictReader(f)
        required = {"donor", "acceptor", "delta_G_eV", "rate"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV must have columns: donor, acceptor, delta_G_eV, rate, gamma")
        for row in reader:
            i = int((row["donor"]).strip())
            j = int((row["acceptor"]).strip())
            raw_rate = (row.get("rate") or "").strip()
            raw_dG   = (row.get("delta_G_eV") or "").strip()
            gamma    = (row.get("gamma") or "").strip()

            if raw_rate != "":
                r = float(raw_rate)
            elif raw_dG != "":
                r = rate_from_deltaG(float(raw_dG), beta, k_h)
            elif gamma != "":
                r = gamma_s
            else:
                raise ValueError("Each row needs either rate or delta_G_eV or gamma value")

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

    labels = [f"P{k}" for k in range(N)]
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
    
    out = []
    for i in t_ps:
        evolve = linalg.expm(K*i) @ P0
        out.append(evolve)
    out_array = np.asarray(out)
    return out_array
        

def graphing(t_ps: np.ndarray, P_t: np.ndarray, labels=None) -> None:
    """
    This function graphs the cal_evolution one.
    the input is an array where each time point corresponds to a new row, with the amount of nodes
    being the number of columns. (This is a far easier way of doing things than was previously implemented)
    """
    if labels is None:
        labels = [f"P{i}" for i in range(P_t.shape[1])] # this is a failsafe
    for i in range(P_t.shape[1]):  # this just indexes each of the columns (rows would have been .shape[0])
        # The x axis is time in picoseconds and will be determined in another function
        # the number of rows in P_t must match the number of time points otherwise the function will not run
        # we then want to iterate over each of the columns 
        plt.plot(t_ps, P_t[:, i], label=labels[i])
    plt.xlabel("Time in seconds")
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
    edges_path = "edges.csv"   # This line depends on the name of CSV file -> would CLI path be better?

    # Build A, K
    A, labels = adjacency_matrix(edges_path, T_K=300.0, tau_s=1e-11)
    K = make_K_matrix(A)

    # Initial condition (one-hot at node 0)
    N = A.shape[0] # number of rows of A so we can think of this as the number of nodes
    P0 = np.zeros(N, dtype=float)
    P0[0] = 1.0

    # Time grid
    t_s = np.linspace(0, 1, num=10000)

    # Evolve and plot
    P_t = calc_evolution(K, P0, t_s)
    graphing(t_s, P_t, labels=labels)

if __name__ == "__main__":
    main()
