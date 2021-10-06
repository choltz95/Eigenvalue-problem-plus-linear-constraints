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
from objectives import *


def constant(step_size) -> Schedule:
  def schedule(i):
    return step_size
  return schedule

def make_schedule(scalar_or_schedule: Union[float, Schedule]) -> Schedule:
  if callable(scalar_or_schedule):
    return scalar_or_schedule
  elif jnp.ndim(scalar_or_schedule) == 0:
    return constant(scalar_or_schedule)
  else:
    raise TypeError(type(scalar_or_schedule))
    
    
@optimizer
def psgd(projector):
  """Construct optimizer triple for stochastic gradient descent.
  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  def init(x0):
    return x0
  def update(a, g, x):
    return projector(x - a * g)
  def get_params(x):
    return x
  return init, update, get_params

@optimizer
def padam(step_size, projector, b1=0.9, b2=0.999, eps=1e-8):
  """Construct optimizer triple for Adam.
  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).
  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    m0 = jnp.zeros_like(x0)
    v0 = jnp.zeros_like(x0)
    return x0, m0, v0
  def update(i, g, state):
    x, m, v = state
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * jnp.square(g) + b2 * v  # Second moment estimate.
    mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
    vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
    x = projector(x - step_size(i) * mhat / (jnp.sqrt(vhat) + eps))
    return x, m, v
  def get_params(state):
    x, _, _ = state
    return x
  return init, update, get_params

def pgd(X_k, A_x, A_y, b_x, b_y, C, c=jnp.array([0,0])):
    """Perform iterations of PGD, without autograd."""
    loss = []
    param_hist  = []
    for k in tqdm(range(1000)):
        X_k_x = X_k[:,0] - alpha*A_x@X_k[:,0]
        X_k_y = X_k[:,1] - alpha*A_y@X_k[:,1]
        X_k = np.vstack([X_k_x,X_k_y]).T - alpha*np.vstack([b_x,b_y]).T
        X_k = project(X_k, C, c)
        param_hist.append(X_k)
        loss.append(f(X_k, A_x, A_y, b_x, b_y))   
    return {'x':X_k, 'lossh':loss, 'sln_path':param_hist}

@jit
def step(i, opt_state, A_x, A_y, b_x, b_y):
    """Perform a single gradient (using autograd) + projection step with adaptive momentum."""
    p = get_params(opt_state)
    g = grad(f)(p, A_x, A_y, b_x, b_y)
    return opt_update(i, g, opt_state)

def pgd_autograd(opt_params, A_x, A_y, b_x, b_y, C, convergence_criterion, maxiters=1000):
    """Perform iterations of PGD, with autograd """
    opt_state, opt_update, get_params = opt_params
    E_0 = np.stack([b_x, b_y], axis=1)
    X_k = get_params(opt_state)
    loss = [np.array(f(X_k, A_x, A_y, b_x, b_y))]
    Lh = [np.eye(2)]
    param_hist  = [X_k]
    grad_hist= []
    hess_hist = []
    for k in tqdm(range(maxiters)):
        opt_state = step(k, opt_state, A_x, A_y, b_x, b_y)
        X_k = get_params(opt_state)
        param_hist.append(X_k)
        l = np.array(f(X_k, A_x, A_y, b_x, b_y))
        
        assert not np.isnan(l)
        
        if len(loss) > 1 and np.abs(l - loss[-1]) <= convergence_criterion:
            break
        loss.append(l)

        L = -jnp.linalg.inv(C)@(X_k.T@(A_x@X_k+E_0))
        Lh.append(L)
        
        gr = grad(f_l)(X_k, L, C, A_x, A_y, b_x, b_y)
        grad_hist.append(np.linalg.norm(gr[:,0]) + np.linalg.norm(gr[:,1]))
    return {'x':X_k, 'lossh':loss, 'sln_path':param_hist, 'foc':grad_hist, 'hess':hess_hist, 'L':Lh}

@jit
def _step(i, opt_state, Z):
    """Perform a single descent + projection step with arbitrary descent direction."""
    return opt_update(i, Z, opt_state)

@jit
def project(X1, C, E_0, c=jnp.array([0,0])):
    C1 = X1.T@X1
    C1sqrt = utils._sqrtm(C1)
    Csqrt = utils._sqrtm(C)
    U,s,V = jnp.linalg.svd(Csqrt@C1sqrt)
    X = X1@jnp.linalg.inv(C1sqrt)@U@V.T@Csqrt

    # default to this unless not improve cost
    # normalized v as Q
    negdef = jnp.all(jnp.linalg.eigvals(X.T@E_0) <= 0)
    U_E, _, V_E = jnp.linalg.svd(X.T@E_0)
    X = jax.lax.cond(negdef,
                     lambda _ : X@(-U_E@V_E.T),
                     lambda _ : X,
                     operand=None
                    )
    return X.real

@jit
def sqp(A, L, E_0, X):
    """Compute the newton direction and lagrangian multipliers"""
    I = jnp.eye(A.shape[0])
    w, v = jnp.linalg.eig(L)
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]

    D = jnp.diag(w)
    E = -E_0 - (A@X + X@L)
    
    Del_0 = jnp.linalg.inv(X.T@jnp.linalg.inv(A + D[0,0]*I)@X)@X.T@jnp.linalg.inv(A + D[0,0]*I)@E[:,0]
    Z_0 = jnp.linalg.inv(A + D[0,0]*I)@(-X@Del_0 + E[:,0])
    
    Del_1 = jnp.linalg.inv(X.T@jnp.linalg.inv(A + D[1,1]*I)@X)@X.T@jnp.linalg.inv(A + D[1,1]*I)@E[:,1]
    Z_1 = jnp.linalg.inv(A + D[1,1]*I)@(-X@Del_1 + E[:,1])
    
    Z = jnp.stack([Z_0, Z_1], axis=1)
    Del = jnp.stack([Del_0, Del_1], axis=1)
    
    return Z, Del

"""Perform iterations of PND + backtracking line search."""
def newton(opt_params, A, L, C, X_k, b_x, b_y, convergence_criterion, 
           maxiters=100, alpha=1e-2, beta=0.9):
    opt_state, opt_update, get_params = opt_params
    loss = [np.array(f(X_k, A, A, b_x, b_y))]
    param_hist  = []
    E_0 = np.stack([b_x, b_y], axis=1)
    X_k, _ = get_params(opt_state)
    
    grad_hist= []
    hess_hist = []
    
    data = {'L':[], 'gradcorr':[], 'stp':[]}
    for k in tqdm(range(maxiters)):   
        
        #L = (jnp.linalg.inv(C)@(X_k.T@(A@X_k+E_0)) + 
        #      (jnp.linalg.inv(C)@(X_k.T@(A@X_k+E_0))).T)/2
        L = jnp.linalg.inv(C)@X_k.T@(A@X_k+E_0)
        #L_t = L_t + stp*Del
                
        Z, Del = sqp(A, L, E_0, X_k)
        
        # backtracking line search
        f_x = f(X_k, A, A, b_x, b_y)
        #f_x = f_l(X_k, L, C, A, A, b_x, b_y)
        f_xp = 1e8
        stp = 1
        
        gr = grad(f)(X_k, A, A, b_x, b_y)
        derphi = np.trace(gr.T@Z)
        #derphi = np.trace(np.dot(grad(f_l)(X_k, L, C, A, A, b_x, b_y).T, Z))

        len_p = np.linalg.norm(Z)
        X_k_t = X_k
        
        opt_state_t = opt_state
        
        while f_xp >= f_x:#- alpha * stp * derphi:
            stp *= beta
            opt_state_t = _step(stp, opt_state, (-Z, -Del))
            X_k_t, _ = get_params(opt_state_t)
            
            f_xp = np.array(f(X_k_t, A, A, b_x, b_y))
        
            if stp * len_p < 1e-8:
                break       
                
        #L = L + stp*Del
        
        opt_state = opt_state_t
        X_k, _ = get_params(opt_state_t)
     
        param_hist.append(X_k)
        #gr = grad(f_l)(X_k, L, C, A, A, b_x, b_y)
        #grad_hist.append(np.linalg.norm(gr[:,0]) + np.linalg.norm(gr[:,1]))
        grad_hist.append(np.linalg.norm(f_l_sqp(X_k, -L, C, A, E_0)))
        
        
        if len(loss) > 1 and np.abs(f_xp - loss[-1]) <= convergence_criterion:
            break
        loss.append(np.array(f(X_k, A, A, b_x, b_y)))
        data['gradcorr'].append(derphi)
        data['L'].append(L)
        data['stp'].append(stp)
        
    return {'x':X_k, 'lossh':loss, 'sln_path':param_hist, 'ext_data':data, 'foc':grad_hist}
    
def ssm():
    """
    1. compute newton direction z = sqp(X, Z, v, Ax + E0) & subspace S
    2. approximate locally optimal X, L on S; X = min F(\hat{X}, B, V.T@E0)
    """
    pass