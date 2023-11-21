import gurobipy as gp
from gurobipy import GRB

model = gp.Model()
model.addVars([(1, 1), (1, 2)], name='X')
model.update()
print(model.getVarByName('X[1,1]'))
