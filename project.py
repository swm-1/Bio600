
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import constants
import matplotlib.pyplot as plt
import csv
import math
from scipy import linalg
import argparse

def beta_from_T(T_K: float) -> float:
    """Return beta = 1/(k_B T) in 1/eV."""
    k_B_J_per_kelvin = constants.Boltzmann      # J/K
    K_B_eV_per_kelvin = k_B_J_per_kelvin * 6.242e18  # eV/K
    return 1.0 / (K_B_eV_per_kelvin * T_K)

def reverse_rate(k_f: float, KBT: float) -> float:
    '''
    The purpose of this function is to calculate the reverse rate in a way that satisfies detailed balance.
    Detailed balance definition:
    k_r / k_f = exp(-deltaE * beta)
    '''
    k_reverse = (k_f * np.exp(-KBT))
    return k_reverse



def FWHM(file: str) -> float:
    """
    Calculate the full width half max value. The reason for doing this is so that
    we can calculate w (or the standard deviation in for A_p(lambda))

    the file named T01.absorption_Qyonly.txt is data for the Qy absorption band of chlorophyll A.
    """
    x, y = np.loadtxt(file, skiprows=1,unpack=True)

    peak = np.argmax(y) # This is for indexing
    # the interpolate function is going to look like this:
    # np.interp(new_y, y, x)
    new_y = np.max(y) / 2 # half max
    left = np.interp(new_y, y[:peak+1], x[:peak+1]) # interpolation
    right = np.interp(new_y, y[peak:][::-1], x[peak:][::-1]) # interpolation

    fwhm = right - left
    return fwhm

def A_calc(file_path: str) -> np.ndarray:
    """
    Small function that calculates A_p(lambda). See below for the equation (maybe at later date include 
    latex code for the formula)
    """
    fwhm = FWHM(file_path)
    x, y = np.loadtxt(file_path, skiprows=1,unpack=True)
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


def check(A:np.ndarray,file_path: str) -> None:

    """
    A_p(lambda) must integrate to have an overall area with 1. A_p(lambda) is dimensionless because it is expressed in nm^-1
    and therefore when we integrate it is being multiplied by d lambda and thus the units cancel.
    However this function is essentially doing nothing at the moment, i am not even sure if it checks the validity 
    in the correct manner.
    """

    x, _ = np.loadtxt(file_path, skiprows=1, unpack=True)
    integrated_A = np.trapezoid(A, x)

    if np.isclose(integrated_A, 1.0 , rtol=1e-2) == True:
        print("This is normalized and ready to use")
        return None
    else:
        raise ValueError("The value for A when integrated is not 1")
        
    

def Gamma(A: np.ndarray,file_path_pig: str,  filepath: str, h: float=constants.h, c: float=constants.c, N: int=100, sigma: float=1e-20) -> float: 
    
   """
   Units:
   
    - cross section (chlorophyll) = 10^-20 m^2
    - spectral irradience = Wm^-2 nm^-1 -> Js^-1 m^-2 nm^-1
    - lambda / h*C = J^-1
    - A_p(lambda) = dimensionless
    - Number of pigments = dimensionless 

   """ 
   check(A, file_path_pig)
   lam_pig, _ = np.loadtxt(file_path_pig,unpack=True, skiprows=1)
   lam_star, f = np.loadtxt(filepath, unpack=True)

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

def adjacency_matrix(gamma_s: float, edges_csv: str, T_K: float = 300.0
                     , KBT_switcher: Optional[Dict[Tuple[int,int],float]] = None) -> np.ndarray:
    """
    This function loads information about the system.The information is contained within a single 
    CSV file that contains the donor and acceptor nodes and also delta G values or the rate of transfer.
    The output of the function is the weighted A (Adjacency) matrix.
    """ 
    
    
    edges = []          # list of (i, j, rate)
    nodes = -1 # could be any negative number

    with open(edges_csv, 'r', newline="") as f:
        reader = csv.DictReader(f)
        required = {"donor", "acceptor", "KBT", "rate", "labels"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV must have columns: donor, acceptor, KBT, rate, gamma")
        for row in reader:
            i = int((row["donor"]).strip())
            j = int((row["acceptor"]).strip())
            raw_rate = (row.get("rate") or "").strip()
            KBT   = (row.get("KBT") or "").strip()
            gamma    = (row.get("gamma") or "").strip()

            # This will allow us to sweep through values of KBT and essentially 
            # ignore the value of KBT in the CSV file
            if KBT_switcher is not None:
                key = (i,j)  # define which edge we care about
                if key in KBT_switcher:
                    KBT = str(KBT_switcher[key]) 
        

            # exitation
            if gamma != "":
                r = float(gamma_s)
            # calculate the backwards rate
            elif KBT and raw_rate != "":
                r = reverse_rate(float(raw_rate), float(KBT))
            # forward rate
            elif raw_rate != "":
                r = float(raw_rate)
            else:
                raise ValueError("You did something wrong")


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

    return A

def make_K_matrix(A: np.ndarray) -> np.ndarray:
    """/home/sea/Desktop/Bio600/absorption_spectras/Chla.absorption.txt
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
        

def graphing(t_ps: np.ndarray, P_t: np.ndarray, edgefile: str, labels=None) -> None:
    """
    This function graphs the cal_evolution one.
    the input is an array where each time point corresponds to a new row, with the amount of nodes
    being the number of columns. (This is a far easier way of doing things than was previously implemented)
    """
    
    with open( edgefile, 'r', newline="") as f:
        reader = csv.DictReader(f)
        labels = []
        for row in reader:
            j = str((row["labels"]).strip())
            labels.append(j)

    if labels is None:
        labels = [f"P{i}" for i in range(P_t.shape[1])] # this is a failsafe
    for i in range(P_t.shape[1]):  # this just indexes each of the columns (rows would have been .shape[0])
        # The x axis is time in seconds and will be determined in another function
        # the number of rows in P_t must match the number of time points otherwise the function will not run
        # we then want to iterate over each of the columns 
        plt.plot(t_ps, P_t[:, i], label=[f"{labels[i]}"])
    plt.xlabel("Time in seconds")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True) # good to have a grid instead of blank space
    plt.tight_layout()
    plt.show()

def electron_out_put_for_heat (KBT_switcher: Optional[Dict[Tuple[int, int], float]], gamma: float, 
                               edges_path: str,   t_s: np.ndarray, P0: np.ndarray) -> float:
    """
    This function will return the electron output for a given edge or iteration of an edge. This is crucial for the heat map as this is what
    we are using as the fitness (not sure that is the correct word)
    """
    A = adjacency_matrix(
        gamma_s=gamma,
        edges_csv=edges_path,
        T_K=300.0,
        KBT_switcher=KBT_switcher
        )
   
    K = make_K_matrix(A)

    P_t = calc_evolution(K, P0, t_s)

    electron_output = P_t[-1, -1] * 1e3 # The rate can perhaps not be hardcoded in the future?
    return electron_output


def graphing_heat_map (  edges_path: str, gamma: float, group_CS_R_edges: list[tuple[int, int]],
                       group_oxidation_edges: list[tuple[int, int]],KBT_CS_R_values: np.ndarray,KBT_oxidation_values: np.ndarray) -> None:
    """
    This is the function where we are going to be producing the heat map. The namings and this docstrig also need to be updated
    """
    A_for_size = adjacency_matrix(gamma, edges_path, T_K=300.0)
    N = A_for_size.shape[0]

    P0 = np.zeros(N, dtype=float)
    P0[0] = 1.0

    t_s = np.linspace(0, 10, num=1000)

    # 2. Allocate 2D array for fitness values
    Z = np.zeros((len(KBT_CS_R_values), len(KBT_oxidation_values)), dtype=float)

    # 3. Double loop over parameter grid
    for i, kbt_CS_R in enumerate(KBT_CS_R_values):      # y-axis
        for j, kbt_ox in enumerate(KBT_oxidation_values):  # x-axis

            overrides: dict[tuple[int, int], float] = {}

            # Set KBT for all edges in group A
            for edge in group_CS_R_edges:
                overrides[edge] = kbt_CS_R

            # Set KBT for all edges in group B
            for edge in group_oxidation_edges:
                overrides[edge] = kbt_ox

            # Now compute fitness for this combination
            electron_output = electron_out_put_for_heat(
                KBT_switcher=overrides,
                gamma=gamma,
                edges_path=edges_path,
                t_s=t_s,
                P0=P0,
            )
            Z[i, j] = electron_output

    # 4. Plot heatmap
    plt.figure()
    plt.imshow(
        Z,
        origin="lower",
        extent =(
            float(KBT_oxidation_values[0]), float(KBT_oxidation_values[-1]),
            float(KBT_CS_R_values[0]), float(KBT_CS_R_values[-1])
        ),
        aspect="auto",
    )
    plt.colorbar(label="Electron output (arb. units)")
    plt.xlabel("ΔEox (in KBT)")
    plt.ylabel("ΔE (in KBT)")
    plt.title("Optimal energy gaps for a simple anoxygenic photosystem")
    plt.tight_layout()
    plt.show()



def probability_check(P_t: np.ndarray)-> None:
    
    sum_probs = np.sum(P_t, axis= 1)
    if np.isclose(sum_probs, 1, rtol=1e-2).all():
        print("probailities are conserved :)")
        return None
    else:
        raise ValueError("Probabilities do not sum to 1")
    

def main():
    """
     reads edges from 'edges.csv' in the current folder (or change the filename below). 
     The function then constructs P0 and creates the timestamps, then calls the relevant functions

    """
    edges_path = "edges.csv"   # This line depends on the name of CSV file -> would CLI path be better?

    # need to construct a CLI parser for the filepath of the solar spectrum and pigment absorption spectra
    parser = argparse.ArgumentParser(description="Variables that can change in the file (maybe a better description is needed).")
    parser.add_argument("File_name_star", type=str,nargs='?', help="filepath for the solar spectrum data", default="5800K.txt")
    parser.add_argument("File_name_pig", type=str, nargs='?', help="provide the filepath for the absorption spectrum of the pigment", default="NCHL261(bchla_Qy).absorption.txt")
    parser.add_argument('-s','--sigma', help='input what value the absorption cross-section is.', type=float, nargs='?', default= 1e-20)
    args = parser.parse_args()
    
    
    # Calculate Gamma
    A = A_calc(args.File_name_pig)
    gamma = Gamma(A,args.File_name_pig,args.File_name_star,  N=100, sigma=args.sigma) # in s^-1
    # Build A, K
    A = adjacency_matrix(gamma, edges_path, T_K=300.0)
    K = make_K_matrix(A)
    # Initial condition -> probability starts on the ground state (node 0)
    N = A.shape[0] # number of rows of A so we can think of this as the number of nodes
    P0 = np.zeros(N, dtype=float)
    P0[0] = 1.0
    # Time grid
    t_s = np.linspace(0, 1, num=10000)
    # Evolve
    #P_t = calc_evolution(K, P0, t_s)
    #probability_check(P_t)
    #print(f"the rate of excitation is {gamma} per second")
    #print(f"average electron output is {P_t[-1,-1] * 1e3}") # last entry of P_t matrix which is the final P value of the final node
    
    #graphing(t_s, P_t, edges_path)
    cs_r = [
        (2,1),
        (3,2),
        (5,4)
    ]

    oxidation = [
        (4,2),
        (5,3)
        
    ]

    KBT_CSandR_values = np.linspace(0.0, 15, 50)  # ΔE_CS = ΔE_r from 0 → 20 kBT
    KBT_oxidation_values = np.linspace(0.0, 15, 50)  # oxidation energy gap(s) from 0 → 20 kBT

    graphing_heat_map (edges_path=edges_path, gamma=gamma, group_CS_R_edges=cs_r,
                       group_oxidation_edges=oxidation,KBT_CS_R_values=KBT_CSandR_values,KBT_oxidation_values=KBT_oxidation_values,
    )
             


if __name__ == "__main__":
    main()
