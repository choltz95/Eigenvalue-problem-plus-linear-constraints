from typing import Any, Callable, NamedTuple, Tuple, Union
Step = int
Schedule = Callable[[Step], float]

from collections import namedtuple

import time

from scipy import sparse as sp

import jax
from jax import jit, vmap, random, grad, value_and_grad
from jax.experimental import optimizers, sparse
from jax.experimental.optimizers import optimizer
from jax import numpy as jnp

from functools import partial
import itertools

import math
import numpy as np
import numpy.random as npr

from utils import *

@jit
def PX(v, X):
    return X - v@(v.T@X)

@jit
def XP(X,v):
    return X - (X@v)@v.T

#@jit
def subspace(X_k, Z, v, A, E_0, C):
    v_s = jnp.ones((A.shape[0],1))/np.sqrt(A.shape[0])
    X_k = X_k - v_s@(v_s.T@X_k)
    AX = A@X_k
    AX = AX - v_s@(v_s.T@AX)
    
    AXE = AX + E_0
    Z = Z - v_s@(v_s.T@Z)
    Q, _ = jnp.linalg.qr(jnp.concatenate([X_k, Z, v, AXE],axis=-1), mode='reduced')
    
    PQ = Q - v_s@(v_s.T@Q)
    B=PQ.T@(A@PQ)
    w,v = nonzero_eig(B)
    v = Q@v[:,:2] 
    
    return Q,v 

@jit
def project(X1, C, E_0, c=jnp.array([0,0])):
    C1 = X1.T@X1
    C1sqrt = utils._sqrtm(C1)
    Csqrt = utils._sqrtm(C)
    U,s,V = jnp.linalg.svd(Csqrt@C1sqrt)
    X = X1@jnp.linalg.inv(C1sqrt)@U@V.T@Csqrt
    
    U_E, _, V_E = jnp.linalg.svd(X.T@E_0)
    X = X@(-U_E@V_E.T)
    return X.real

@partial(jit, static_argnums=(3,))
def step(i, opt_state, Z, opt_update):
    """Perform a single descent + projection step with arbitrary descent direction."""
    return opt_update(i, Z, opt_state)

def _D_Z(X, A, d, e):
    I = jnp.eye(A.shape[0])
    Ad = A + d*I
    
    Del = jnp.linalg.solve(X.T@jnp.linalg.solve(Ad, X), X.T)@jnp.linalg.solve(Ad, e)
    Z = jnp.linalg.solve(Ad, -X@Del + e)
    
    return Del, Z

@jit
def sqp(A, L, E_0, X):
    """Perform an iteration of SQP.""" 
    w = jnp.linalg.eigvals(L)
    idx = w.argsort() 
    w = w[idx]
    E = -E_0 - (A@X + X@L)

    Del_0, Z_0 = _D_Z(X, A, w[0], E[:,0])
    Del_1, Z_1 = _D_Z(X, A, w[1], E[:,1])
    
    Z = jnp.stack([Z_0, Z_1], axis=1)
    Del = jnp.stack([Del_0, Del_1], axis=1)
    
    return Z, Del  

def scipy_sqp(X, A, P, L, E_0, I):
    """Perform an iteration of SQP.""" 
    w = jnp.linalg.eigvals(L)
    idx = w.argsort()  
    w = w[idx]
    v = np.ones((A.shape[0],1))/np.sqrt(A.shape[0])
    AX = A@(X - v@(v.T@X))
    AX = AX - v@(v.T@AX)
    E = -E_0 - (AX + X@L)
    Del_0, Z_0 = scipy_D_Z(X, A, P, w[0], E[:,0], I)
    Del_1, Z_1 = scipy_D_Z(X, A, P, w[1], E[:,1], I)
    
    Z = jnp.stack([Z_0, Z_1], axis=1)
    Del = jnp.stack([Del_0, Del_1], axis=1)
    
    return Z, Del  

def bicg_solve(A, B, M):
    if len(B.shape) > 1 and B.shape[1] > 1:
        X, info = zip(*(sp.linalg.bicgstab(A, b, M=M, tol=1e-6) for b in B.T))
    else:
        X, info = sp.linalg.bicgstab(A,B, M=M, tol=1e-6)
    return np.transpose(X), info

def scipy_D_Z(X, A, P, d, e, I):
    Ad = A + d*I
    Ad = sp.csc_matrix((A.data, (A.indices[:,0], A.indices[:,1])))
        
    sp_solve = lambda A, b:jnp.array(sp.linalg.spsolve(A.astype(np.float64),b.astype(np.float64)))
   
    Del = jnp.linalg.solve(X.T@(sp_solve(Ad,X)),X.T)@sp_solve(Ad,e)
    
    v_s = np.ones((A.shape[0],1))/np.sqrt(A.shape[0])
    XDE = (-X@Del + e)

    PXDE = XDE - (XDE@v_s)@v_s.T
    ADinvP = sp_solve(Ad, PXDE)
    Z = ADinvP - v_s@(v_s.T@ADinvP)
    
    return Del, Z