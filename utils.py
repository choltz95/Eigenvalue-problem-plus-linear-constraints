from typing import Any, Callable, NamedTuple, Tuple, Union
Step = int
Schedule = Callable[[Step], float]

import scipy.io as io
import scipy.sparse.csgraph as csgraph
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
import scipy as sp
from scipy.linalg import null_space
from scipy.linalg import sqrtm

import jax
from jax import jit, vmap, random, grad
from jax.experimental import optimizers
from jax.experimental.optimizers import optimizer
from jax import numpy as jnp

from functools import partial
import itertools

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from matplotlib import collections as mc
import seaborn as sns

import datetime
from tqdm.notebook import tqdm

import networkx as nx


def _sqrtm(C):
    # Computing diagonalization
    evalues, evectors = jnp.linalg.eig(C)
    # Ensuring square root matrix exists
    sqrt_matrix = evectors @ jnp.diag(jnp.sqrt(evalues)) @ jnp.linalg.inv(evectors)
    return sqrt_matrix.real

def qr_null(A, tol=None):
    Q, R, P = sp.linalg.qr(A.T, mode='full', pivoting=True)
    tol = np.finfo(R.dtype).eps if tol is None else tol
    rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
    return Q,Q[:, rnk:].conj()

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

def voxel_cluster(pos, size):
    pos = pos.reshape(pos.shape[0],-1)
    start = pos.min(0)
    end = pos.max(0)
    pos = pos - jnp.expand_dims(start,0);

    num_voxels = jnp.divide(end - start,size).astype(int) + 1
    num_voxels = jnp.cumprod(num_voxels,0)
    num_voxels = jnp.concatenate([jnp.ones(1), num_voxels], 0)
    num_voxels = num_voxels[0:size.shape[0]]
    out = jnp.divide(pos, size.reshape(1,-1)).astype(int)
    voxels = out
    out *= num_voxels.reshape(1, -1)
    out = out.sum(1);

    return out, voxels

def quad_voxel_cluster(pos,size):
    pos = pos.reshape(pos.shape[0],-1)
    start = pos.min(0)
    end = pos.max(0)
    pos = pos - jnp.expand_dims(start,0);

    num_voxels = jnp.divide(end - start,size).astype(int) + 1
    
    return out, voxels    

def voxel2centroid(voxelcoord, size, offset):
    return voxelcoord*size - offset

def clust_to_mask(cluster, cid):
    mask = np.zeros_like(cluster)
    mask[cluster==cid]+=1
    return mask

def plot_graph(positions, graph, c=None, title="", fixed_indices=[], filename=None):
    plt.figure(figsize=(20,10))
    ax = plt.axes()
    ax.set_xlim(min(positions[:,0]), max(positions[:,0]))
    ax.set_ylim(min(positions[:,1]), max(positions[:,1]))

    lines = []
    for i,j in zip(*graph.nonzero()):
        if i > j:
            lines.append([positions[i], positions[j]])

    lc = mc.LineCollection(lines, linewidths=1, colors='k', alpha=.25)
    ax.add_collection(lc)
    ax.scatter(positions[:,0], positions[:,1], s=5, c=c)

    for c in fixed_indices:
        ax.annotate('({},{})'.format(str(np.round(positions[c, 0],2)), str(np.round(positions[c, 1],2))), (positions[c, 0], positions[c, 1]))

    plt.title(title)    
    if filename is not None:
        plt.savefig(filename + '.svg', format='svg', dpi=1000)
    return ax