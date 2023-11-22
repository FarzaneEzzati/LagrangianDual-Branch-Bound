import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('ToySolution.pkl', 'rb') as handle:
        x_opt, y_pves, y_dges, y_grides, \
            y_pvl, y_dgl, y_esl, y_gridl, y_l, y_lh, y_ll,\
            y_pvcur,y_dgcur, y_pvgrid, y_dggrid, y_esgrid, \
            y_gridplus, y_gridminus, e, pv, tgs_indices, zstar = pickle.load(handle)
    handle.close()

    plt.bar(range(1, 16), [y_l[(t, 1, 1)] for t in range(1, 16)])
    plt.show()
