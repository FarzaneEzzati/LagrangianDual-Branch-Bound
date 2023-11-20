import copy
import pickle
import random
import math
import numpy as np
import pandas as pd
import gurobipy
from gurobipy import quicksum, GRB
env = gurobipy.Env()
env.setParam('OutputFlag', 0)

def geneCases():
    # Each element represents the scenarios for a month (1, ..., 12)
    with open('OutageScenarios.pkl', 'rb') as handle:
        scens, probs = pickle.load(handle)
    handle.close()

    # There are 5 scenarios.

    # Load profile of households
    load_profile_1 = pd.read_csv('Load_profile_1.csv')
    load_profile_2 = pd.read_csv('Load_profile_2.csv')

    # PV output for one unit (4kW)
    pv_profile = pd.read_csv('PV_profiles.csv')
    return scens, probs, load_profile_1, load_profile_2, pv_profile


class BuildModel:
    def __init__(self):
        pass
    def Build(self):
        # =============================================  PARAMETERS ==================================================
        # Read the data about scenarios, load profiles, and pv profiles
        scenarios, prob, load1, load2, pv = geneCases()

        # Initialization of parameters
        Budget = 1000000
        Horizon = 20
        interest_rate = 0.08
        operational_rate = 0.01
        PA_factor = ((1 + interest_rate) ** Horizon - 1) / (interest_rate * (1 + interest_rate) ** Horizon)
        self.C = {1: 600, 2: 2780, 3: 150}
        self.O = {i: operational_rate * PA_factor * self.C[i] for i in (1, 2, 3)}
        self.CO = {i: (self.C[i] + self.O[i]) / (365*24) for i in (1, 2, 3)}
        UB = [60, 30, 10]
        LB = [10, 10, 2]
        self.FuelPrice = 3.7
        alpha, beta = 0.5, 0.2
        self.GridPlus = 0.1497
        self.GridMinus = alpha * self.GridPlus
        self.LoadPrice = self.GridPlus
        self.GenerPrice = beta * self.GridPlus
        self.VoLL = np.array([1.4, 1.8, 2.1]) * self.GridPlus
        self.PVSellPrice = (alpha + beta) * self.GridPlus
        self.DGSellPrice = self.PVSellPrice
        self.PVCurPrice = (alpha + beta) * self.GridPlus
        self.DGCurPrice = (alpha + beta) * self.GridPlus
        SOC_UB, SOC_LB = 0.9, 0.1
        ES_gamma = 0.85
        Eta_c = 0.8
        Eta_i = 0.9

        # Ranges need to be used
        T = 168
        SCount = len(scenarios)
        DVCCount = 3
        MCount = 3
        HCount = 2
        OutageStart = 15
        RNGDvc = range(1, DVCCount+1)
        RNGTime = range(1, T + 1)
        RNGTimeMinus = range(1, T)
        RNGMonth = range(1, MCount+1)
        RNGScen = range(1, SCount+1)
        RNGScenMinus = range(1, SCount)
        RNGHouse = range(1, HCount+1)

        # Define the load profiles and PV profiles
        load = [load1, load2]
        Load = {(h, t, g): load[h-1][f'Month {g}'].iloc[t - 1] for h in RNGHouse for t in RNGTime for g in RNGMonth}
        PV_unit = {(t, g): pv[f'Month {g}'].iloc[t - 1] for t in RNGTime for g in RNGMonth}
        Out_Time = {(g, s): 0 for s in RNGScen for g in RNGMonth}

        for s in RNGScen:
            if scenarios[s-1] != 0:
                for g in RNGMonth:
                    Out_Time[(g, s)] = [OutageStart + i for i in range(int(scenarios[s - 1]))]

        Prob = {s: prob[s - 1] for s in RNGScen}
        # =============================================  Make Model  ==================================================
        # Generate the Model (with Lagrangian Dual)
        model = gurobipy.Model('MIP', env=env)
        X_indices = [(s, j) for s in RNGScen for j in RNGDvc]  # 1: ES, 2: PV, 3: DG
        X = model.addVars(X_indices, vtype=GRB.INTEGER, name='X')

        # Bounds on X decisions
        for index in RNGScen:
            model.addConstr(X[(index, 1)] <= UB[0], name='ES UB')
            model.addConstr(X[(index, 2)] <= UB[1], name='PV UB')
            model.addConstr(X[(index, 3)] <= UB[2], name='DG UB')
            model.addConstr(X[(index, 1)] >= LB[0], name='ES LB')

        # First stage constraint
        model.addConstr(quicksum([X[(s, j)] * self.C[j] for s in RNGScen for j in RNGDvc]) <= Budget, name='budget constraint')

        # Second Stage Variables
        Y_indices = [(t, g, s) for t in RNGTime for g in RNGMonth for s in RNGScen]
        Yh_indices = [(h, t, g, s) for h in RNGHouse for t in RNGTime for g in RNGMonth for s in RNGScen]
        Ytg_indices = [(t, g) for t in RNGTime for g in RNGMonth]

        Y_PVES = model.addVars(Y_indices, name='Y_PVES')
        Y_DGES = model.addVars(Y_indices, name='Y_DGES')
        Y_GridES = model.addVars(Y_indices, name='Y_GridES')

        Y_PVL = model.addVars(Y_indices, name='Y_PVL')
        Y_DGL = model.addVars(Y_indices, name='Y_DGL')
        Y_ESL = model.addVars(Y_indices, name='Y_ESL')
        Y_GridL = model.addVars(Y_indices, name='Y_GridL')

        Y_L = model.addVars(Y_indices, name='Y_L')
        Y_LH = model.addVars(Yh_indices, name='Y_LH')
        Y_LL = model.addVars(Yh_indices, name='Y_LL')

        Y_PVCur = model.addVars(Y_indices, name='Y_PVCur')
        Y_DGCur = model.addVars(Y_indices, name='Y_DGCur')

        Y_PVGrid = model.addVars(Y_indices, name='Y_DGGrid')
        Y_DGGrid = model.addVars(Y_indices, name='Y_DGGrid')
        Y_ESGrid = model.addVars(Y_indices, name='Y_ESGrid')

        Y_GridPlus = model.addVars(Y_indices, name='Y_GridPlus')
        Y_GridMinus = model.addVars(Y_indices, name='Y_GridMinus')

        E = model.addVars(Y_indices, name='E')

        PV = model.addVars(Ytg_indices, name='PV')

        u = model.addVars(Y_indices, name='u')

        # Lambda definition
        lmda = [[0 for _ in RNGDvc] for _ in range(len(scenarios)-1)]

        # Second stage constraints
        # Energy storage level
        model.addConstrs(E[(1, g, s)] == SOC_UB * X[(s, 1)] for s in RNGScen for g in RNGMonth)
        model.addConstrs(E[(1, g, s)] == E[(T, g, s)] for s in RNGScen for g in RNGMonth)
        model.addConstrs(SOC_LB * X[(s, 1)] <= E[(t, g, s)] for t in RNGTime for s in RNGScen for g in RNGMonth)
        model.addConstrs(E[(t, g, s)] <= SOC_UB * X[(s, 1)] for t in RNGTime for s in RNGScen for g in RNGMonth)
        # Balance of power flow
        model.addConstrs(E[(t + 1, g, s)] == E[(t, g, s)] +
                         ES_gamma * (Y_PVES[(t, g, s)] + Y_DGES[(t, g, s)] + Eta_c * Y_GridES[(t, g, s)]) -
                         (Y_ESL[(t, g, s)] + Y_ESGrid[(t, g, s)]) / ES_gamma
                         for t in RNGTimeMinus for s in RNGScen for g in RNGMonth)
        # The share of Load
        model.addConstrs(Y_L[(t, g, s)] == Eta_i * (Y_ESL[(t, g, s)] + Y_DGL[(t, g, s)] + Y_PVL[(t, g, s)]) +
                         Y_GridL[(t, g, s)]
                         for t in RNGTime for s in RNGScen for g in RNGMonth)

        model.addConstrs(quicksum(Y_LH[(h, t, g, s)] for h in RNGHouse) <= Y_L[(t, g, s)]
                         for t in RNGTime for s in RNGScen for g in RNGMonth)

        model.addConstrs(Y_LH[(h, t, g, s)] + Y_LL[(h, t, g, s)] == Load[(h, t, g)]
                         for h in RNGHouse for t in RNGTime for s in RNGScen for g in RNGMonth)

        model.addConstrs(PV[(t, g)] == X[(s, 2)] * PV_unit[(t, g)]
                         for t in RNGTime for s in RNGScen for g in RNGMonth)

        model.addConstrs(Y_PVL[(t, g, s)] + Y_PVES[(t, g, s)] + Y_PVCur[(t, g, s)] + Y_PVGrid[(t, g, s)] == PV[(t, g)]
                         for t in RNGTime for s in RNGScen for g in RNGMonth)

        model.addConstrs(Y_GridPlus[(t, g, s)] == Eta_c * Y_GridES[(t, g, s)] + Y_GridL[(t, g, s)]
                         for t in RNGTime for s in RNGScen for g in RNGMonth)

        model.addConstrs(Y_GridMinus[(t, g, s)] == Eta_i * (Y_ESGrid[(t, g, s)] + Y_PVGrid[(t, g, s)] + Y_DGGrid[(t, g, s)])
                         for t in RNGTime for s in RNGScen for g in RNGMonth)

        model.addConstrs(Y_DGL[(t, g, s)] + Y_DGES[(t, g, s)] + Y_DGGrid[(t, g, s)] + Y_DGCur[(t, g, s)] == X[(s, 3)]
                         for t in RNGTime for s in RNGScen for g in RNGMonth)

        model.addConstrs(Y_ESL[(t, g, s)] + Y_ESGrid[(t, g, s)] <= UB[0] * u[(t, g, s)]
                         for t in RNGTime for s in RNGScen for g in RNGMonth)

        model.addConstrs(Y_PVES[(t, g, s)] + Y_GridES[(t, g, s)] + Y_DGES[(t, g, s)] <= UB[0] * (1 - u[(t, g, s)])
                         for t in RNGTime for s in RNGScen for g in RNGMonth)

        for s in RNGScen:
            for g in RNGMonth:
                if Out_Time[(g, s)] != 0:
                    model.addConstrs(Y_GridPlus[(t, g, s)] == 0 for t in Out_Time[(g, s)])
                    model.addConstrs(Y_GridMinus[(t, g, s)] == 0 for t in Out_Time[(g, s)])

        # Save variables as self for later use
        self.X = X
        self.model = model
        self.Prob = Prob
        self.Y_PVES = Y_PVES
        self.Y_DGES = Y_DGES
        self.Y_GridES = Y_GridES
        self.Y_PVL = Y_PVL
        self.Y_DGL = Y_DGL
        self.Y_ESL = Y_ESL
        self.Y_GridL = Y_GridL
        self.Y_L = Y_L
        self.Y_LH = Y_LH
        self.Y_LL = Y_LL
        self.Y_PVCur = Y_PVCur
        self.Y_DGCur = Y_DGCur
        self.Y_PVGrid = Y_PVGrid
        self.Y_DGGrid = Y_PVGrid
        self.Y_ESGrid = Y_PVGrid
        self.Y_GridPlus = Y_PVGrid
        self.Y_GridMinus = Y_PVGrid
        self.E = Y_PVGrid
        self.PV = PV

        self.RNGDvc = RNGDvc
        self.RNGScen = RNGScen
        self.RNGScenMinus = RNGScenMinus
        self.RNGTime = RNGTime
        self.RNGMonth = RNGMonth
        self.RNGHouse = RNGHouse
        self.SetObjective(lmda)

    def Solve(self):
        self.model.optimize()
        OutPut = []
        if self.model.status == 2:
            OutPut.append(self.ReturnSolutionValue(self.X))
            OutPut.append(self.model.ObjVal)
        else:
            OutPut.append('Not Feasible')
        return OutPut

    def SetObjective(self, cur_lambda):
        X = self.X
        Y_PVES = self.Y_PVES
        Y_DGES = self.Y_DGES
        Y_GridES = self.Y_GridES
        Y_GridL = self.Y_GridL
        Y_LH = self.Y_LH
        Y_LL = self.Y_LL
        Y_PVCur = self.Y_PVCur
        Y_DGCur = self.Y_DGCur
        Y_PVGrid = self.Y_PVGrid
        Y_DGGrid = self.Y_DGGrid
        Y_DGL = self.Y_DGL
        Y_ESGrid = self.Y_ESGrid
        Y_GridPlus = self.Y_GridPlus
        Y_GridMinus = self.Y_GridMinus
        PV = self.PV
        Prob = self.Prob
        RNGDvc = self.RNGDvc
        RNGScen = self.RNGScen
        RNGScenMinus = self.RNGScenMinus
        RNGTime = self.RNGTime
        RNGMonth = self.RNGMonth
        RNGHouse = self.RNGHouse
        PVCurPrice = self.PVCurPrice
        FuelPrice = self.FuelPrice
        GridPlus = self.GridPlus
        GridMinus = self.GridMinus
        LoadPrice = self.LoadPrice
        GenerPrice = self.GenerPrice
        VoLL = self.VoLL
        CO = self.CO

        Cost1 = quicksum([Prob[s] * quicksum([X[(s, j)] * (CO[j]) for j in RNGDvc]) for s in RNGScen])
        Cost2 = quicksum([Prob[s] * quicksum([PVCurPrice * (Y_PVCur[(t, g, s)] + Y_DGCur[(t, g, s)]) for t in RNGTime for g in RNGMonth]) for s in RNGScen])
        Cost3 = quicksum([Prob[s] * quicksum([VoLL[h - 1] * Y_LL[(h, t, g, s)] for h in RNGHouse for t in RNGTime for g in RNGMonth]) for s in RNGScen])
        Cost4 = quicksum([Prob[s] * FuelPrice * quicksum([Y_DGL[(t, g, s)] + Y_DGGrid[(t, g, s)] + Y_DGCur[(t, g, s)] +
                                                 Y_DGES[(t, g, s)] for t in RNGTime for g in RNGMonth]) for s in RNGScen])
        Cost5 = quicksum([Prob[s] * quicksum([GridPlus * Y_GridPlus[(t, g, s)] - GridMinus * Y_GridMinus[(t, g, s)] -
                                              GenerPrice * PV[(t, g)] - quicksum([LoadPrice * Y_LH[(h, t, g, s)] for h in RNGHouse])
                                              for t in RNGTime for g in RNGMonth]) for s in RNGScen])
        Cost6 = quicksum(cur_lambda[s-1][d - 1] * (X[(s, d)] - X[(s+1, d)]) for s in RNGScenMinus for d in RNGDvc)
        self.model.setObjective(Cost1 + Cost2 + Cost3 + Cost4 + Cost5 + Cost6, sense=GRB.MINIMIZE)
        self.model.update()

    def UpdateObjective(self, cur_lambda):
        self.SetObjective(cur_lambda)

    def UpdateLmda(self, iteration, solution, old_lambda, step_size):
        sub_gradient = [[solution[0][d-1] - solution[1][d-1] for d in self.RNGDvc],
                   [solution[1][d-1] - solution[2][d-1] for d in self.RNGDvc]]
        return old_lambda - np.multiply(sub_gradient, step_size ** iteration)

    def GetXBar(self, X):
        return np.multiply(self.Prob[1], X[0]) + np.multiply(self.Prob[2], X[1]) + np.multiply(self.Prob[3], X[2])

    def ReturnSolutionValue(self, solution):
        return [[solution[(s, d)].x for d in self.RNGDvc] for s in self.RNGScen]

    def FixVars(self, XFixed):
        for s in self.RNGScen:
            for d in self.RNGDvc:
                self.model.addConstr(self.X[(s, d)] == XFixed[d-1], name=f'Fixed({s},{d})')
        self.model.update()

        output = self.Solve()

        for s in self.RNGScen:
            for d in self.RNGDvc:
                self.model.remove(self.model.getConstrByName(f'Fixed({s},{d})'))
        self.model.update()
        return output

    def BranchFloor(self, x_index, bound):
        self.model.addConstrs(self.X[(s, x_index)] <= bound for s in self.RNGScen)
        self.model.update()

    def BranchCeiling(self, x_index, bound):
        self.model.addConstrs(self.X[(s, x_index)] >= bound for s in self.RNGScen)
        self.model.update()

    def CheckIdent(self, solution):
        return np.sum([int(np.array_equal(solution[s], solution[s+1])) for s in self.RNGScenMinus])


if __name__ == '__main__':

    scenarios, prob, load1, load2, pv = geneCases()

    # The original model is used to evaluate generated XBarR, as the average of sub-problems first stage decisions
    OriginalModel = BuildModel()
    OriginalModel.Build()

    model = BuildModel()
    model.Build()

    # Writing the algorithm
    Z_LB = -float('inf')
    IdenticalsFound = False

    # Set PP = {MIP}
    PSet = [model]
    PObjectives = [0]
    PSelected = model
    XSelected = 0
    NodeItr = 0

    # START THE ALGORITHM
    while len(PSet) > 0:
        NodeItr += 1
        print(f'\n{20*"="} Node Iteration {NodeItr} {20*"="}')
        print(f'Best lower bound so far {Z_LB}')
        print(PSet)

        # This two empty sets are intended to save new branched models and add it to PSet when the P is removed.
        PSetTemp = []
        PObjectivesTemp = []

        # NODE SELECTION STEP
        PIndex = random.choice(range(len(PSet)))
        Pmodel = PSet[PIndex]

        # Solve it and then give it to the sub-gradient method if feasible
        OutPut = Pmodel.Solve()
        PSet.remove(PSet[PIndex])
        PObjectives.remove(PObjectives[PIndex])

        if OutPut == ['Not Feasible']:
            print('Selected node not feasible, hence removed.')
        else:
            print('Selected node feasible, Lagrangian Dual is applied.')
            X_values = OutPut[0]
            Objective = OutPut[1]
            print(f'Z_LD before applying Subgradient is: {Objective}')

            if NodeItr == 1:
                # Lambda has size (#scenarios -1) * (#devices)
                RootLmda = [[0 for _ in range(3)] for _ in range(len(scenarios)-1)]
            elif NodeItr == 2:
                RootLmda = copy.copy(lmdaOld)

            lmda = RootLmda
            lmdaOld = copy.copy(lmda)
            itr = 0
            ContinueWhile = True
            while ContinueWhile:
                lmda = Pmodel.UpdateLmda(iteration=itr, solution=X_values, old_lambda=lmda, step_size=0.6)
                Pmodel.UpdateObjective(cur_lambda=lmda)
                itr += 1
                if np.array_equal(lmda, lmdaOld):
                    ContinueWhile = False
                else:
                    OutPut = Pmodel.Solve()
                    X_values = OutPut[0]
                    Objective = OutPut[1]
                lmdaOld = copy.copy(lmda)
            SelectedPObj = Objective
            print(f'Z_LD after applying Subgradient is: {Objective}')

            # BOUNDING STEP
            if SelectedPObj < Z_LB:  # Z_LD < Z_LB =>  True: remove P from PSet, False: Continue for branching if needed
                print('Pmodel is removed from PSet due to Z_LD < Z_LB')
            else:
                print('Pmodel is checked for identicality of the solutions')
                CheckIden = Pmodel.CheckIdent(X_values)
                if CheckIden == len(lmda):  # Solutions are identical
                    print(f'Solutions are identical. Lower bound updated. Selected X {X_values[0]}')
                    IdenticalsFound = True
                    # Update zLowerBound if required
                    if SelectedPObj >= Z_LB:
                        Z_LB = SelectedPObj
                        PSelected = Pmodel
                        XSelected = X_values[0]
                    # Remove all models with objective value < zLowerBound
                    for index in range(len(PSet)):
                        if PObjectives[index] < Z_LB:
                            PSet.remove(PSet[index])
                            PObjectives.remove(PObjectives[index])

                else:   # Solutions differ
                    XBar = Pmodel.GetXBar(X_values)
                    XBarR = [math.ceil(x) for x in XBar]

                    # update Z_LB only if you have found an identical solution before. Otherwise, ignore it.
                    if IdenticalsFound:
                        print(f'Solutions are not identical. XBarR is {XBarR}.')
                        # Get cx + sum(pqy) for XR. Note that it must be applied on the primal problem without any bounds added.
                        OutPut = OriginalModel.FixVars(XBarR)
                        print('Output', OutPut)
                        if OutPut == ['Not Feasible']:
                            print('Model with XBarR is not feasible.')
                        else:
                            Z_XBarR = OutPut[1]
                            print(f'Model with XBarR is feasible. Z_XBarR {Z_XBarR} and Z_LB {Z_LB}. Selected X {OutPut[0]}')
                            if Z_XBarR > Z_LB:
                                Z_LB = Z_XBarR
                                PSelected = Pmodel
                                XSelected = OutPut[0]
                            # BRANCHING STEP
                            # Remove all models with objective value <= zLowerBound
                            for index in range(len(PSet)):
                                if PObjectives[index] < Z_LB:
                                    PSet.remove(PSet[index])
                                    PObjectives.remove(PObjectives[index])

                    print(f'Solutions are not identical. Branching Started for XBar {XBar}')
                    flag = True
                    for element in range(len(XBar)):
                        if math.floor(XBar[element]) != XBar[element]:
                            model1 = BuildModel()
                            model1.Build()
                            model1.UpdateObjective(RootLmda)
                            model1.BranchFloor(x_index=element + 1, bound=math.floor(XBar[element]))
                            PSet.append(model1)
                            PObjectives.append(0)

                            model2 = BuildModel()
                            model2.Build()
                            model2.UpdateObjective(RootLmda)
                            model2.BranchCeiling(x_index=element + 1, bound=math.floor(XBar[element])+1)
                            PSet.append(model2)
                            PObjectives.append(0)

                            flag = False
                            break
                    if flag:
                        print('All X variables obtained by one of the model in the tree are integer. NO BRANCHING')
    print(f'Optimal Solution is: {XSelected}')
