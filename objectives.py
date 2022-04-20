from typing import Any, Callable, NamedTuple, Tuple, Union
Step = int
Schedule = Callable[[Step], float]

import jax
from jax import jit, vmap, random, grad, value_and_grad, hessian
from jax.experimental import optimizers
from jax.experimental.optimizers import optimizer
from jax import numpy as jnp

from functools import partial

import numpy as np
import utils

def init_problem(A, E_0, D, v, C):
    def f(X):
        return jnp.trace(jnp.inner(X, A_x@X + 2*E_0))
    def normg(X):
        return v.T@X
    def normh(X):
        return X.T@X - C
    
    return f, g, h

@jit
def f(X, A_x, A_y, E_0):
    obj= X[:,0].T@A_x@X[:,0] + X[:,1].T@A_y@X[:,1] + jnp.trace(jnp.inner(X, 2*E_0))
    return obj

@jit
def foc_sqp(X, L, C, A, E_0):
    obj = A@X + E_0 + X@L
    return jnp.linalg.norm(obj)

def g(X, v, c):
    return np.array([v.T@X[:,0], v.T@X[:,1]]) - c

def h(X, D, c1, c2, c3):
    return np.array([X[:,0].T@D@X[:,0]- c1, 
                     X[:,1].T@D@X[:,1] - c2, 
                     2*(X[:,0].T@D@X[:,1] - c3)])

@jit
def L_init(X_k, C, A, E_0):
    return (jnp.linalg.inv(C)@X_k.T@(A@X_k+E_0))