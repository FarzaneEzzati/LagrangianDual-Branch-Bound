import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def geneCases():
    # Load profile of households
    load_profile_1 = pd.read_csv('Data/Load_profile_1.csv')
    load_profile_2 = pd.read_csv('Data/Load_profile_2.csv')

    # PV output for one unit (4kW)
    pv_profile = pd.read_csv('Data/PV_profiles.csv')
    return pv_profile, load_profile_1, load_profile_2


if __name__ == '__main__':
    pv, l1, l2 = geneCases()

    fig, axs = plt.subplots(1, 2, figsize=(20, 7), dpi=200)
    axs[0].plot(l1['Month 1'], label='Load 1 - Januray')
    axs[0].plot(l2['Month 1'], label='Load 2 - January')
    axs[0].set_xlabel('Hour')
    axs[0].set_ylabel('Load Demand (kW)')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(pv['Month 1'], label='January')
    axs[1].plot(pv['Month 2'], label='June')
    axs[1].set_xlabel('Hour')
    axs[1].set_ylabel('PV Generation (kW)')
    axs[1].grid(True)
    axs[1].legend()

    plt.savefig('IMG/PV_Load_profiles.jpg', bbox_inches='tight')
