"""
Functions to output data of the code
"""
import numpy as np

import domain

def write_superphoton_data(file,prob_domain: domain,fileflag="w") -> None:
    """
    write out the superphoton x and k values for a domain
    """
    with open(file,fileflag) as fp:
        for superphoton in prob_domain.superphotons:
            fp.write(" ".join(map(str,np.concatenate((superphoton.x_init,superphoton.x,superphoton.k))))+"\n")

def read_superphoton_data(file,nfields=2) -> list:
    """
    read in a text file as written above and return a list of x_init, x and k arrays for each superphoton
    """

    with open(file,"r") as fp:
        lines = fp.readlines()
    return_list = []
    for line in lines:
        split_list = line.strip().split()
        return_list.append([np.array(list(map(eval,split_list[:4]))) for _ in range(nfields)])
    return (return_list)