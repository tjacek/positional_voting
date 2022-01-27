import numpy as np
from scipy.optimize import differential_evolution

class GenAlg(object):
    def __init__(self,maxiter=10,init_type='latinhypercube',pop_size=15):
        self.maxiter=maxiter
        self.init_type=init_type
        self.pop_size=pop_size
 
    def __call__(self,loss_fun,n_cand):
        init=init_population(self.init_type,n_cand,self.pop_size)
        bound_w = [(0.0, n_cand)  for _ in range(n_cand)]
        result = differential_evolution(loss_fun, bound_w, 
            init=init,
            maxiter=self.maxiter, tol=1e-7)
        return result['x']

def init_population(init_type,n_cand,pop_size=15):
    if(init_type=="random"):
        return np.random.uniform(0,n_cand,(pop_size,n_cand))
    if(init_type=="borda"):
        return [[n_cand-j
                    for j in range(n_cand)]
                        for _ in range(pop_size)]
    if(init_type=="borda_mixed"):
        pop=[]
        for i in range(pop_size):
            if( (i%2)==0):
                pop.append([n_cand-j for j in range(n_cand)])
            else:
                pop.append( np.random.uniform(low=0.0, 
                    high=n_cand, size=(pop_size,)))
        return pop
    return 'latinhypercube'