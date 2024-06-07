from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import dccp

n=100 #number of points
radius=0.08 #typical distance between particles
r = np.random.normal(loc=1.0, scale=0.01, size=n) #radii

c = Variable(shape=(n,2))
constr = []
for i in range(n-1):
    for j in range(i+1,n):
        constr.append(norm(c[i,:]-c[j,:])>=r[i]+r[j])
prob = Problem(Minimize(max(max(abs(c),axis=1)+r)), constr)
#prob = Problem(Minimize(max_entries(normInf(c,axis=1)+r)), constr)
prob.solve(method = 'dccp', ccp_times = 1)

l = max(max(abs(c),axis=1)+r).value*2
pi = np.pi
ratio = pi*sum(square(r)).value/square(l).value
print("ratio =", ratio)
print(prob.status)


xs = c[:, 0].value
ys = c[:, 1].value
pos = np.array([xs, ys]).reshape(-1, 2)
np.save(f'../lattices/random_packing_n{n}.npy', pos, allow_pickle=False)
