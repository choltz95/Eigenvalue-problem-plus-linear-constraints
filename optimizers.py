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
