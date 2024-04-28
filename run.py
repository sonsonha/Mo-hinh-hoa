import math
import pandas as pd
import numpy as np
from gamspy import *
from scipy.stats import binom

k_values = list(range(11))
distribution = []
for k in k_values:
    distribution.append(binom.pmf(k, 10, 0.5))

require_data_frame = []
lines = open("require_matrix.txt").readlines()
for line in lines:
    num = line.split()
    arr = []
    for s in num:
        arr.append(int(s))
    require_data_frame.append(arr)

require_data = []
for i in range(8):
    for j in range(5):
        prod_str = "product_" + str(i+1)
        part_str = "part_" + str(j+1)
        require_data.append([prod_str, part_str, require_data_frame[i][j]])

require_matrix = pd.DataFrame(
    require_data,
    columns=["product", "part", "require"]
).set_index(["product", "part"])

demand = []

demand.append(
    pd.DataFrame(
        [["product_1", np.random.choice(np.arange(0,11), p=distribution)],
        ["product_2", np.random.choice(np.arange(0,11), p=distribution)],
        ["product_3", np.random.choice(np.arange(0,11), p=distribution)],
        ["product_4", np.random.choice(np.arange(0,11), p=distribution)],
        ["product_5", np.random.choice(np.arange(0,11), p=distribution)],
        ["product_6", np.random.choice(np.arange(0,11), p=distribution)],
        ["product_7", np.random.choice(np.arange(0,11), p=distribution)],
        ["product_8", np.random.choice(np.arange(0,11), p=distribution)]],
        columns=["product", "demand_1"]
    ).set_index("product")
)

demand.append(
    pd.DataFrame(
        [["product_1", np.random.choice(np.arange(0,11), p=distribution)],
        ["product_2", np.random.choice(np.arange(0,11), p=distribution)],
        ["product_3", np.random.choice(np.arange(0,11), p=distribution)],
        ["product_4", np.random.choice(np.arange(0,11), p=distribution)],
        ["product_5", np.random.choice(np.arange(0,11), p=distribution)],
        ["product_6", np.random.choice(np.arange(0,11), p=distribution)],
        ["product_7", np.random.choice(np.arange(0,11), p=distribution)],
        ["product_8", np.random.choice(np.arange(0,11), p=distribution)]],
        columns=["product", "demand_2"]
    ).set_index("product")
)

product_cost_data = []
product_price_data = []


lines = open("product.txt").readlines()
for i in range (1, 9):
    line = lines[i].split()
    product_cost_data.append(["product_" + str(i), int(line[0])])
    product_price_data.append(["product_" + str(i), int(line[1])])


product_cost = pd.DataFrame(
    product_cost_data,
    columns=["product", "cost\n(l_i)"]
).set_index("product")


product_selling_price = pd.DataFrame(
    product_price_data,
    columns=["product", "selling price\n(q_i)"]
).set_index("product")


part_price_data = []
preorder_cost_part_data = []



lines = open("part.txt").readlines()
for j in range (1,6):
    part_price_data.append(["part_"+str(j), int(lines[j].split()[0])])
    preorder_cost_part_data.append(["part_"+str(j), int(lines[j].split()[1])])


part_selling_price = pd.DataFrame(
    part_price_data,
    columns=["part", "selling price\n(s_j)"]
).set_index("part")

preorder_cost = pd.DataFrame(
    preorder_cost_part_data,
    columns=["part", "preorder cost\n(b_j)"]
).set_index("part")

S = len(demand)


m = Container()

i = Set(m, "i", description="product", records=product_selling_price.index)
j = Set(m, "j", description="part", records=part_selling_price.index)

A = Parameter(
    container=m,
    name="A",
    description="require matrix",
    domain=[i, j],
    records=require_matrix.reset_index(),
)
d = [None]*S
for scenerio in range(S):
    d[scenerio] = Parameter(m, "d_" + str(scenerio), domain=i, description="demand", records=demand[scenerio].reset_index())

l = Parameter(m, "l", domain=i, description="product cost", records=product_cost.reset_index())
q = Parameter(m, "q", domain=i, description="product selling price", records=product_selling_price.reset_index())
s = Parameter(m, "s", domain=j, description="part selling price", records=part_selling_price.reset_index())
b = Parameter(m, "b", domain=j, description="preorder cost per part", records=preorder_cost.reset_index())

x = Variable(m, "x", type="Positive", domain=j)
y = [None]*S
z = [None]*S

require = [None]*S
demand_constraint = [None]*S

obj = Sum(j, x[j]*b[j])

for scenerio in range(S):
    y[scenerio] = Variable(m, "y" + str(scenerio), type="Positive", domain=j)
    z[scenerio] = Variable(m, "z" + str(scenerio), type="Positive", domain=i)
    require[scenerio] = Equation(
        m, "require" + str(scenerio),
        domain=j, description="require of part j to product i"
    )
    require[scenerio][j] = y[scenerio][j] == x[j] - Sum(i, A[i,j]*z[scenerio][i])

    demand_constraint[scenerio] = Equation(
        m, "demand" + str(scenerio),
        domain=i, description="Demand for each product"
    )
    demand_constraint[scenerio][i] = z[scenerio][i] <= d[scenerio][i]

    obj += 0.5*Sum(i, (l[i]-q[i])*z[scenerio][i]) - 0.5*Sum(j, s[j]*y[scenerio][j])

modelTransport = Model(
    m, "modelTransport",
    problem="LP", equations=m.getEquations(),
    sense=Sense.MIN, objective=obj
)

modelTransport.solve(solver="CPLEX")

print("x:\n", x.records)
print("//-------------------Scenerio 1--------------------//")
print("y1:\n", y[0].records)
print("//------------------------------------------------//")
print("z1:\n", z[0].records)
print("//-------------------Scenerio 2--------------------//")
print("y2:\n", y[1].records)
print("//------------------------------------------------//")
print("z2:\n", z[1].records)
print("//------------------------------------------------//")
print("objective result:", modelTransport.objective_value)
