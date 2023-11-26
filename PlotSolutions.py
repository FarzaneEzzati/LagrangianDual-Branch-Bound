import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def geneCases():
    # Load profile of households
    load_profile_1 = pd.read_csv('Data/Load_profile_1.csv')
    load_profile_2 = pd.read_csv('Data/Load_profile_2.csv')

    # PV output for one unit (4kW)
    pv_profile = pd.read_csv('Data/PV_profiles.csv')
    return pv_profile, load_profile_1, load_profile_2


if __name__ == '__main__':
    solar, l1, l2 = geneCases()
    w = 0.5
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

    with open('Initializations.pkl', 'rb') as handle:
        GridPlus, GridMinus, LoadPrice, GenerPrice, VoLL, PVSellPrice, \
            DGSellPrice, PVCurPrice, DGCurPrice, FuelPrice, load, pv_unit, \
            RNGScen, RNGTime, RNGMonth, RNGHouse, DG_gamma = pickle.load(handle)
    handle.close()

    with open('Data/OutageScenarios.pkl', 'rb') as handle:
        Scenarios, Prob = pickle.load(handle)
    handle.close()

    '''fig1, axs = plt.subplots(1, 2, figsize=(30, 7), dpi=200)
    axs[0].bar(RNGT, l1[f'Month {g}'].iloc[0:len(RNGT)], width=w, facecolor='gray',
               label='Demand 1')
    axs[0].bar(RNGT, [y_lh[(1, t, g, s)] for t in RNGT], width=w, linestyle='-', facecolor='green',
               label='Served 1')
    axs[0].bar(RNGT, [y_ll[(1, t, g, s)] for t in RNGT], width=w, linestyle='-', facecolor='red',
            label='Lost 1')
    axs[0].set_xlabel('Hour of Week')
    axs[0].set_ylabel('kW')
    axs[0].legend()

    axs[1].bar(RNGT, l2[f'Month {g}'].iloc[0:len(RNGT)], width=w, facecolor='gray',
            label='Demand 2')
    axs[1].bar(RNGT, [y_lh[(2, t, g, s)] for t in RNGT], width=w, linestyle='-', facecolor='green',
            label='Served 2')
    axs[1].bar(RNGT, [y_ll[(2, t, g, s)] for t in RNGT], width=w, linestyle='-', facecolor='red',
            label='Lost 2')
    axs[1].set_xlabel('Hour of Week')
    axs[1].set_ylabel('kW')
    axs[1].legend()
    plt.savefig(f'IMG/SolutionPlotLarge-Demands-({g}, {s}).jpg', bbox_inches='tight')'''

    fig2 = plt.figure(figsize=(20, 4), dpi=300)
    plt.bar(RNGT, np.add(l2[f'Month {g}'].iloc[0:len(RNGT)], l1[f'Month {g}'].iloc[0:len(RNGT)]),
            width=w, facecolor='gray', label='Demand')
    plt.bar(RNGT, [Eta_i * y_esl[(t, g, s)]for t in RNGT], color='#35824a', width=w, label='ES to Load')

    plt.bar(RNGT, [Eta_i * y_pvl[(t, g, s)]for t in RNGT], color='#c7a644', width=w,
            bottom=[Eta_i * y_esl[(t, g, s)]for t in RNGT], label='PV to Load')

    plt.bar(RNGT, [Eta_i * y_dgl[(t, g, s)]for t in RNGT], color='#3c66b5', width=w,
            bottom=[Eta_i * (y_esl[(t, g, s)] + y_pvl[(t, g, s)]) for t in RNGT], label='DG to Load')

    plt.bar(RNGT, [y_gridl[(t, g, s)]for t in RNGT], color='#9a43a1', width=w,
            bottom=[Eta_i * (y_esl[(t, g, s)] + y_pvl[(t, g, s)] + y_dgl[(t, g, s)]) for t in RNGT], label='Grid to Load')
    plt.bar(RNGT, [-np.sum([y_ll[(h, t, g, s)] for h in range(1, 3)]) for t in RNGT],
            color='red', width=w, label='Load Lost')
    plt.xlim([0, 169])
    plt.legend()
    plt.xlabel('Hour of Week')
    plt.ylabel('kW')
    plt.title(f'Month {g} - Scenario {s} (prob: {Prob[s-1]})')
    plt.savefig(f'IMG/SolutionPlotLarge-Devices-({g},{s}).jpg', bbox_inches='tight')

    answer = input('Get optimal expected cost: y/n')
    if answer == 'y':
        C1 = np.sum(
            [Prob[s-1] * np.sum([PVCurPrice * (y_pvcur[(t, g, s)] + y_dgcur[(t, g, s)]) for t in RNGTime for g in RNGMonth])
             for s in RNGScen])
        C2 = np.sum(
            [Prob[s-1] * np.sum([VoLL[h - 1] * (load[(h, t, g)] - y_lh[(h, t, g, s)]) for h in RNGHouse for t in RNGTime for g in RNGMonth])
             for s in RNGScen])
        C3 = np.sum(
            [Prob[s-1] * FuelPrice * DG_gamma * np.sum([y_dgl[(t, g, s)] + y_dggrid[(t, g, s)] + y_dgcur[(t, g, s)] +
                                                      y_dges[(t, g, s)] for t in RNGTime for g in RNGMonth])
             for s in RNGScen])
        C4 = np.sum(
            [Prob[s-1] * np.sum([GridPlus * y_gridplus[(t, g, s)] -
                                 GridMinus * y_gridminus[(t, g, s)] -
                                 GenerPrice * x_opt[1] * pv_unit[(t, g)] - np.sum([LoadPrice * y_lh[(h, t, g, s)] for h in RNGHouse])
                            for t in RNGTime for g in RNGMonth])
             for s in RNGScen])

        print(f'Expected Cur Cost: {12 * 4 * C1: 0.3f}')
        print(f'Expected Load Lost Cost: {12 * 4 * C2: 0.3f}')
        print(f'Expected Fuel Cost: {12 * 4 * C3: 0.3f}')
        print(f'Expected Transaction with Grid: {12 * 4 * C4: 0.3f}')
