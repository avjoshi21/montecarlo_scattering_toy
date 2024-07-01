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
            fp.write(" ".join(map(str,np.concatenate((superphoton.x,superphoton.k))))+"\n")

def read_superphoton_data(file) -> list:
    """
    read in a text file as written above and return a list of x and k arrays for each superphoton
    """

    with open(file,"r") as fp:
        lines = fp.readlines()
    return_list = []
    for line in lines:
        split_list = line.strip().split()
        return_list.append([np.array(list(map(eval,split_list[:4]))),np.array(list(map(eval,split_list[4:])))])
    return (return_list)