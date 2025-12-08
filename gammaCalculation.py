import numpy as np
import math
from scipy import constants






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


def Gamma(A: np.ndarray,file_path_pig: str,  filepath: str, h: float=constants.h, 
          c: float=constants.c, N: int=100, sigma: float=1e-20) -> float: 
    
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