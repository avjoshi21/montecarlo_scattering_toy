# main script for running test scattering problem


def main():
    # pseudo code for now
    # define scattering domain ()
    # initialize Ninit superphotons from origin as blackbody, each with same weight and bias values
    # parallelize across available cores using jax. Each core works on a superphoton and the recursively scattered ones until completion
        # for each superphoton, have code to integrate out (flat space) by dlambda. Code should still be relativistic, just minkowski metric.
        # compute scattering optical depth at each step
            # 
    pass
if __name__ == "__main__":
    main()