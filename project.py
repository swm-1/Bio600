
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import constants
import matplotlib.pyplot as plt
import csv
import gammaCalculation
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


def adjacency_matrix(gamma_s: float, edges_csv: str
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


def probability_check(P_t: np.ndarray)-> None:
    
    sum_probs = np.sum(P_t, axis= 1)
    if np.isclose(sum_probs, 1, rtol=1e-2).all():
        return None
    else:
        raise ValueError("Probabilities do not sum to 1")
        

def electron_out_put_for_heat(KBT_switcher: Optional[Dict[Tuple[int, int], float]], gamma: float, 
                               edges_path: str,   t_s: np.ndarray, P0: np.ndarray) -> float:
    """
    Function parameters:
        - KBT_switcher -> This parameter is a dictionary and will override the KBT values that are found within the CSV file.
                          The only real information needed to produce this dictionary is the lists of edges that we care about. 
                          For the sake of ease it should be kept to two seperate groups of edges and these can be contained within 
                          seperate lists (ΔE & ΔE_ox).
        - gamma        -> This is the manual calculation for the reat of excitation.
        - edges_path   -> This is the file path for the edges.csv or any CSV file that contains correctly formatted information about
                          the system.
        - t_s          -> This parameter is just a 1-D array that provides how long of a timespan the calculation should be run for.
        - P0           -> This is just the initial probability vector.

    Returns:
        The function should return the electron output for a given set of conditions.

    Notes:
        This function should be pretty self explanitory.
    """
    A = adjacency_matrix(
        gamma_s=gamma,
        edges_csv=edges_path,
        KBT_switcher=KBT_switcher
        )
   
    K = make_K_matrix(A)

    P_t = calc_evolution(K, P0, t_s)
    probability_check(P_t=P_t)

    electron_output = P_t[-1, -1] * 1e3 # The rate can perhaps not be hardcoded in the future?
    return electron_output


def graphing_heat_map (  edges_path: str, gamma: float, delta_E_edges: list[tuple[int, int]],
                       delta_E_ox_edges: list[tuple[int, int]],KBT_delta_E_values: np.ndarray,KBT_delta_E_ox_values: np.ndarray) -> None:
    """
    The purpose of this function is to output a heat map that corresponds to the electron output at varying values of KBT.

    Function parameters:
    
    - edges_path      -> this is the csv file where all of the information about the system we are modelling is contained.
    - Gamma           -> The calculation for the rate of excitation, but currently this is not being used.
    - delta_E_edges   -> This parameter corresponds to all of the edges that encode ΔE_CS and ΔE_r. The reason that these two have been grouped
                         together is because they are equal (ΔE_CS = ΔE_r). This parameter must be passed as a list of the edges.
    -delta_E_ox_edges -> This parameter is esentially the same as the one above. The only difference is that the edges contained
                         correspond to ΔE_ox. 
    (A quick thing to note about the above parameters (detla_E_ox_edges and delta_E_edges) these can be changed depending on desired outcome 
    of the function. i.e. they need not just represent ΔE_ox or ΔE.)
   
    - KBT_delta_E_values & KBT_delta_E_ox_values -> these two parameters should be the same. They are a 1-D array of values. This is 
                                                    the range of KBT values you wish to evaluate 
    
    Returns:
        The function in and of itself does not return anything, the whole purpose of it is to generate a heat map.
                                                
    """
    
    # this array does not really do anything, it is just needed to get the correct number of rows for the 
    # intial probability vector
    A_for_size = adjacency_matrix(gamma, edges_path)
    N = A_for_size.shape[0]
    # initial probability vector
    P0 = np.zeros(N, dtype=float)
    P0[0] = 1.0 
    # this is the time that the calculation will be run for
    t_s = np.linspace(0, 10, num=1000)

    # Create an array to store the calculated electron outputs in
    Z = np.zeros((len(KBT_delta_E_values), len(KBT_delta_E_ox_values)), dtype=float)

    # Double loop over parameter grid
    for i, kbt_CS_R in enumerate(KBT_delta_E_values):      # y-axis because axis 0 - corresponds to the rows
        for j, kbt_ox in enumerate(KBT_delta_E_ox_values):  # x-axis because axis 1 - correpsponds to the coloumns of a matrix.

            overrides: dict[tuple[int, int], float] = {}

            # Set KBT for all edges in group A
            for edge in delta_E_edges:
                overrides[edge] = kbt_CS_R

            # Set KBT for all edges in group B
            for edge in delta_E_ox_edges:
                overrides[edge] = kbt_ox

            # Now compute electron output for this combination
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
            float(-KBT_delta_E_ox_values[0]), float(-KBT_delta_E_ox_values[-1]),
            float(-KBT_delta_E_values[0]), float(-KBT_delta_E_values[-1])
        ),
        aspect="auto"
    )
    plt.colorbar(label="Electron output (arb. units)")
    plt.xlabel("ΔEox (in KBT)")
    plt.ylabel("ΔE (in KBT)")
    plt.title("Optimal energy gaps for a simple anoxygenic photosystem")
    plt.tight_layout()
    plt.show()

def oneDimensionalGraph(edges_path: str, delta_E_edges: list[tuple[int,int]], 
                        gamma: float, delta_E_values: np.ndarray):
    """
    I should write a docstring
    """

    A_for_size = adjacency_matrix(gamma_s=gamma,edges_csv=edges_path)
    N = A_for_size.shape[0]
    P0=np.zeros(N,dtype=float)
    P0[0] = 1.0
    t_s = np.linspace(0,10,num=1000)

    overrides = {}
    output = []
    for kbt in delta_E_values:
        for edge in delta_E_edges:
            overrides[edge] = kbt
        output.append(electron_out_put_for_heat(
            KBT_switcher=overrides,
                gamma=gamma,
                edges_path=edges_path,
                t_s=t_s,
                P0=P0
        ))
    plt.plot(-delta_E_values, output)
    plt.gca().invert_xaxis()
    plt.title("Optimal energy gaps for ΔE")
    plt.xlabel("Number of KBT")
    plt.ylabel("Electron output")
    plt.show()
    

def grouped_edge_loader(file_path: str):
    """
    This function will take in the CSV file which contains the information about the system and group the edges depending on whether they are
    the edges associated with ΔE_ox or ΔE_CS/ΔE_r.

    The parameters of the function are:
        - Edges_path -> this is the CSV that contains the info about the system

    The function should return two lists. One being a list of edges associated with oxidation and the other being one associated with non-oxidation

    Extra comments:
        - If all parameters need to be varied the function should be updated to sort each parameter accordingly 
        - The file must be formatted correctly with the reverse of ox, red or CS being labeled as those. If not 
          this function will miss handel the file.

    """

    with open (file_path, 'r', newline="") as f:
        reader = csv.DictReader(f)
        required = {"donor", "acceptor", "labels"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV file must contain a donor, acceptor and labels column")
    
        oxidation_edges = []
        reduction_edges = []
        charge_sep_edges = []
        
        for row in reader:
            i = int((row["donor"]).strip())
            j = int((row["acceptor"]).strip())
            if row["labels"] == "Ox":
                oxidation_edges.append([i,j])
            elif row["labels"] == "CS":
                charge_sep_edges.append([i,j])
            elif row["labels"] == "Red":
                reduction_edges.append([i,j])
        
    non_ox_edges = reduction_edges + charge_sep_edges
    # These lines are needed to format into list[tuple[int,int]]
    group_1 = [tuple(pair) for pair in oxidation_edges] # Group_1 canonically should be for non_ΔE_ox_edges, but can be anything as long as it is specified.
    group_2 = [tuple(pair) for pair in non_ox_edges]    # Group_2 canonically is the ΔE_ox_edges, but again this can be anything as long as it is specified.
    return group_1,group_2

def timeTakenForSteadyState (  edges_path: str, gamma: float, KBT_delta_E_values: np.ndarray) -> list:
    
    A_for_size = adjacency_matrix(gamma, edges_path)
    N = A_for_size.shape[0]
    # initial probability vector
    P0 = np.zeros(N, dtype=float)
    P0[0] = 1.0 
    time = np.linspace(0,1,10000)
   
    delta_E_ox_edges, delta_E_edges = grouped_edge_loader(edges_path)
    all_edges = delta_E_ox_edges + delta_E_edges
    
    time_taken =[]
    for i in KBT_delta_E_values:

        overrides ={}
        for edge in all_edges:
            overrides[edge] = i
            
        A = adjacency_matrix(
        gamma_s=gamma,
        edges_csv=edges_path,
        KBT_switcher=overrides
        )
        K = make_K_matrix(A=A)
        P_t = calc_evolution(P0=P0, t_ps=time, K=K)
        probability_check(P_t=P_t)
        y = P_t[:,-1] #final node
        tolerance = 0.001
        dy = np.diff(y)
        dt = np.diff(time)
        mask = dy / dt <= tolerance

        count = 0
        threshold = 100
        for n, value in enumerate(mask):

            if count >= threshold:
                index = n - threshold
                time_taken.append(time[index])
                break
    
            elif value == True:
                count +=1
    
            else:
                count = 0
        


    
    return time_taken


def graphForTimeSteadyState(times_group_1, KBT_delta_E_values, times_group_2 = None):

    plt.plot(-KBT_delta_E_values, times_group_1, label= "single trap")
    if times_group_2 is not None:
        plt.plot(-KBT_delta_E_values, times_group_2, label="double trap")
        plt.gca().invert_xaxis()
        plt.xlabel("ΔE in KBT")
        plt.ylabel("Time taken to reach steady state [D+PTA-]/[D+PTTA-] in seconds")
        plt.legend()
        plt.show()
    else:
         plt.gca().invert_xaxis()
         plt.show()
    



def main():
    """
     reads edges from 'edges.csv' in the current folder (or change the filename below). 
     The function then constructs P0 and creates the timestamps, then calls the relevant functions

    """
    
    parser = argparse.ArgumentParser(description="Variables that can change in the file and need user specification.")
    parser.add_argument('-2', '--heatmap', help='This will return a heat map', action="store_true")
    parser.add_argument('-1', '--oneDplot', help='This will return the 1-D plot', action="store_true")
    parser.add_argument('-o', '--oxidation', help='Tells the program which edges to use', action="store_true")
    parser.add_argument('-e', '--nonox', help='Essentially the same as above just not the oxidation edges', action="store_true")
    parser.add_argument ('-f', '--filepath', help='Tells the program where the information for the system is')
    parser.add_argument('-t', '--time', help='This will show how long it takes for a given system to reach the steady state', action='store_true')
    args = parser.parse_args()

    
    gamma = gammaCalculation.Gamma(file_path_pigment="NCHL261(bchla_Qy).absorption.txt", 
                                   file_path_star="5800K.txt")

    edges_path = args.filepath
    if edges_path ==  None:
        edges_path = input("Please specify a filepath: ") # Not sure if this should be kept or not.

    oxidation_edges, non_oxidation_edges = grouped_edge_loader(file_path=edges_path)
    
    KBT_non_ox_values = np.linspace(0.0, 25, 50)  # ΔE_CS = ΔE_r from 0 → 20 kBT
    KBT_ox_values = np.linspace(0.0, 10, 20)  # oxidation energy gap(s) from 0 → 20 kBT

    if args.heatmap is True:
        graphing_heat_map (edges_path=edges_path, gamma=gamma, delta_E_edges= non_oxidation_edges,
                      delta_E_ox_edges=oxidation_edges, KBT_delta_E_values=KBT_non_ox_values, KBT_delta_E_ox_values=KBT_ox_values,
        )
    if args.time is True:
        time_group_1 = timeTakenForSteadyState(edges_path=edges_path, gamma=gamma, KBT_delta_E_values= KBT_non_ox_values)
        time_group_2 = timeTakenForSteadyState(edges_path="edges2Trap.csv", gamma=gamma, KBT_delta_E_values=KBT_non_ox_values)
        graphForTimeSteadyState(times_group_1=time_group_1, KBT_delta_E_values=KBT_non_ox_values, times_group_2=time_group_2)
    elif args.oneDplot is True:
        if args.nonox is True:
            oneDimensionalGraph(edges_path=edges_path, delta_E_edges=non_oxidation_edges, gamma=gamma, delta_E_values=KBT_non_ox_values)
        elif args.oxidation is True:
            oneDimensionalGraph(edges_path=edges_path, delta_E_edges=oxidation_edges, gamma=gamma, delta_E_values=KBT_ox_values)
        else:
            raise ValueError("Need to specify which edges you want to vary")
    else:
        raise ValueError("Need to specify what plot you want")


if __name__ == "__main__":
    main()
