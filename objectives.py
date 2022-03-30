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

@jit # A_x = A_y
def f(X, A_x, A_y, b_x, b_y):
    obj= X[:,0].T@A_x@X[:,0] + X[:,1].T@A_y@X[:,1] + 2*b_x.T@X[:,0] + 2*b_y.T@X[:,1]
    return obj, obj

@jit
def f_l(X, L, C, A_x, A_y, b_x, b_y):
    obj = jnp.trace(jnp.inner(X, A_x@X + 2*jnp.stack([b_x,b_y],axis=1))) + jnp.trace(jnp.inner(L, X.T@X - C))
    return obj

@jit
def foc_pgd(X, L, C, A, b_x, b_y):
    obj = jnp.linalg.norm((A + L[0,0]*jnp.eye(A.shape[0]))@X[:,0] + L[1,0]*X[:,1] + b_x) + \
    jnp.linalg.norm((A + L[1,1]*jnp.eye(A.shape[0]))@X[:,1] + L[1,0]*X[:,0] + b_y)
    return obj

@jit
def foc_sqp(X, L, C, A, E_0):
    obj = A@X + E_0 + X@L
    return jnp.linalg.norm(obj)

def soc(L, P):
    pass

def g(X, v, c):
    return np.array([v.T@X[:,0], v.T@X[:,1]]) - c

def h(X, D, c1, c2, c3, c=jnp.array([0,0])):
    return np.array([(X[:,0]-c[0]).T@D@(X[:,0]-c[0]) - c1, 
                     (X[:,1]-c[1]).T@D@(X[:,1]-c[1]) - c2, 
                     2*((X[:,0]-c[0]).T@D@(X[:,1]-c[1]) - c3)])

@jit
def L_init(X_k, C, A, E_0):
    return (jnp.linalg.inv(C)@X_k.T@(A@X_k+E_0))