import pickle
import random

from distfit import distfit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as ss
import scipy.stats as st
import pickle as pkl

class Outage:
    def __init__(self):
        pass

    def dataPreparation(self):
        # Import Data 1
        data1 = pd.read_csv('2000_2016_purdue.csv')
        mask1 = data1['POSTAL.CODE'] == 'TX'
        data1_dropped = data1[mask1].dropna()
        XD = data1_dropped['OUTAGE.DURATION.HOUR']
        XS = data1_dropped['OUTAGE.START.HOUR']
        XW = data1_dropped['Week Day']

        # Import Data 2
        data2 = pd.read_csv('Texas_2002_2023.csv')
        mask2 = data2['Year'] > 2016
        data2_dropped = data2[mask2].dropna()
        YD = data2_dropped['Outage Duration']  # no need to divide by 60 since it is in hour unit
        YS = data2_dropped['Outage Hour']
        YW = data2_dropped['Week Day']

        # Remove outliers and combine the outage durations
        XYS = np.concatenate((XS, YS))
        XYD = np.concatenate((XD, YD))
        XYW = np.concatenate((XW, YW))
        Q1 = np.percentile(XYD, 25, method='midpoint')
        Q3 = np.percentile(XYD, 75, method='midpoint')
        IQR = Q3 - Q1
        mask3 = XYD <= Q3 + 1.5 * IQR
        OutDur = XYD[mask3]
        OutStrt = XYS[mask3]
        OutDay = XYW[mask3]

        fig, axs = plt.subplots(1, 2, figsize=(20, 7))
        axs[0].hist(OutStrt, bins=24, color='#697cb8', edgecolor='black', alpha=1, density=True)
        axs[0].set_xlabel('Hour of Day')
        axs[0].set_ylabel('Frequency of Outage Events')
        axs[0].set_xticks(range(24), range(24))

        axs[1].hist(OutDay, bins=7, width=0.5, edgecolor='black', color='orange', density=True)
        axs[1].set_xlabel('Week Day of Events')
        axs[1].set_ylabel('Frequency of Outage Events')
        axs[1].set_xticks(range(1, 8), ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])

        plt.savefig('IMG/outage_start.jpg', dpi=300, bbox_inches='tight')

        with open('Outage.pkl', 'wb') as handle:
            pickle.dump([OutDur, OutStrt], handle)
        handle.close()


    def fitDistribution(self):

        with open('Outage.pkl', 'rb') as handle:
            OutDur, OutStrt = pickle.load(handle)
        handle.close()

        dist = distfit(smooth=5, bins=10000)
        dist.fit_transform(OutDur)
        pd.DataFrame(dist.summary).to_csv('PDFs.csv')

        fig, axs = plt.subplots(2, 1, figsize=(20, 14))
        dist.plot(chart='pdf', n_top=10, ax=axs[0])
        axs[0].set_ylim([0, 1.1])
        axs[0].set_xlabel('Outage Duration')

        dist.plot(chart='cdf', n_top=10, ax=axs[1])
        axs[1].set_ylim([0, 1.1])
        axs[1].set_xlabel('Outage Duration')

        plt.savefig('IMG/DistFit.jpg', dpi=300, bbox_inches='tight')

        figure1 = plt.figure(figsize=(20, 7))
        dist.plot_summary()
        plt.ylabel('RSS')
        plt.xlabel('Popular Probability Density Functions (PDF)')
        plt.savefig('IMG/PDFs.jpg', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    task = eval(input('What to do: 1) outage data, 2) distfit, 3) generate data '))
    outage = Outage()
    if task == 1:
        outage.dataPreparation()
    elif task == 2:
        outage.fitDistribution()
    elif task == 3:
        shape, scale = 0.5539, 13.36

        CDF = st.norm.cdf([-3, -2, -1, 0, 1, 2, 3])
        Scens = st.gamma.ppf(CDF, shape, loc=0.02, scale=scale).round()

        # Generate scenarios
        ScenProbs = [i for i in CDF]
        for index in range(1, 6):
            ScenProbs[index] = CDF[index] - CDF[index-1]
        ScenProbs[6] = CDF[6] - CDF[5]

        # Reduce count of scenarios
        RScens = [Scens[0]]
        RProbs = [ScenProbs[0]]
        temp = 0
        for index in range(len(Scens)-1):
            if Scens[index] == Scens[index + 1]:
                RProbs[-1] += ScenProbs[index + 1]
            else:
                RScens.append(Scens[index+1])
                RProbs.append(ScenProbs[index+1])

        print(RScens)
        with open('OutageScenarios.pkl', 'wb') as handle:
            pickle.dump([RScens, RProbs], handle)
        handle.close()


