import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def geneCases():
    # Load profile of households
    load_profile_1 = pd.read_csv('Load_profile_1.csv')
    load_profile_2 = pd.read_csv('Load_profile_2.csv')

    # PV output for one unit (4kW)
    pv_profile = pd.read_csv('PV_profiles.csv')
    return pv_profile, load_profile_1, load_profile_2


if __name__ == '__main__':
    solar, l1, l2 = geneCases()
    w = 0.4
    s = 5  # Scenario number
    g = 1  # Month number
    Eta_i = 0.9
    with open('Solution.pkl', 'rb') as handle:
        x_opt, y_pves, y_dges, y_grides, \
            y_pvl, y_dgl, y_esl, y_gridl, y_l, y_lh, y_ll,\
            y_pvcur,y_dgcur, y_pvgrid, y_dggrid, y_esgrid, \
            y_gridplus, y_gridminus, e, pv, y_indices, ytg_indices, yh_indices, RNGT, zstar = pickle.load(handle)
    handle.close()
    print(f'Optimal Solution is: {x_opt}')


    fig1, axs = plt.subplots(1, 2, figsize=(20, 7), dpi=200)
    axs[0].bar(RNGT, l1[f'Month {g}'].iloc[0:len(RNGT)], width=w, linestyle='--', edgecolor='black', facecolor='white',
               label='Demand 1')
    axs[0].bar(RNGT, [y_lh[(1, t, g, s)] for t in RNGT], width=w, linestyle='-', facecolor='green',
               label='Served 1')
    axs[0].bar(RNGT, [y_ll[(1, t, g, s)] for t in RNGT], width=w, linestyle='-', facecolor='red',
            label='Lost 1')
    axs[0].set_xlabel('Hour of Week')
    axs[0].set_ylabel('kW')
    axs[0].legend()

    axs[1].bar(RNGT, l2[f'Month {g}'].iloc[0:len(RNGT)], width=w, linestyle='--', edgecolor='black', facecolor='white',
            label='Demand 2')
    axs[1].bar(RNGT, [y_lh[(2, t, g, s)] for t in RNGT], width=w, linestyle='-', facecolor='green',
            label='Served 2')
    axs[1].bar(RNGT, [y_ll[(2, t, g, s)] for t in RNGT], width=w, linestyle='-', facecolor='red',
            label='Lost 2')
    axs[1].set_xlabel('Hour of Week')
    axs[1].set_ylabel('kW')
    axs[1].legend()
    plt.savefig('IMG/SolutionPlotLarge-Demands.jpg', bbox_inches='tight')

    fig2 = plt.figure(figsize=(20, 7), dpi=300)
    plt.bar(RNGT, np.add(l2[f'Month {g}'].iloc[0:len(RNGT)], l1[f'Month {g}'].iloc[0:len(RNGT)]),
            width=w, linestyle='--', edgecolor='black', facecolor='white', label='Demand')
    plt.bar(RNGT, [Eta_i * y_esl[(t, g, s)]for t in RNGT], color='#35824a', width=w, label='ES to Load')

    plt.bar(RNGT, [Eta_i * y_pvl[(t, g, s)]for t in RNGT], color='#c7a644', width=w,
            bottom=[Eta_i * y_esl[(t, g, s)]for t in RNGT], label='PV to Load')
    '''plt.bar(RNGT, [y_pvcur[(t, g, s)] for t in RNGT], color='orange', width=w,
             label='PV Cur')'''

    plt.bar(RNGT, [Eta_i * y_dgl[(t, g, s)]for t in RNGT], color='#3c66b5', width=w,
            bottom=[Eta_i * (y_esl[(t, g, s)] + y_pvl[(t, g, s)]) for t in RNGT], label='DG to Load')

    plt.bar(RNGT, [y_gridl[(t, g, s)]for t in RNGT], color='#9a43a1', width=w,
            bottom=[Eta_i * (y_esl[(t, g, s)] + y_pvl[(t, g, s)] + y_dgl[(t, g, s)]) for t in RNGT], label='Grid to Load')
    plt.bar(RNGT, [-np.sum([y_ll[(h, t, g, s)] for h in range(1, 3)]) for t in RNGT],
            color='red', width=w, label='Load Lost')
    plt.legend()
    plt.xlabel('Hour of Week')
    plt.ylabel('kW')
    plt.title(f'Month {g} - Scenario {s}')
    plt.savefig(f'IMG/SolutionPlotLarge-Devices-({g},{s}).jpg', bbox_inches='tight')


