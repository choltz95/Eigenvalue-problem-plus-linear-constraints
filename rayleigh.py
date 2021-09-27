@jit
def f(X, A_x, A_y, b_x, b_y):
    obj= X[:,0].T@A_x@X[:,0] + X[:,1].T@A_y@X[:,1] + 2*b_x.T@X[:,0] + 2*b_y.T@X[:,1]
    return obj.real

@jit
def f_l(X, L, C, A_x, A_y, b_x, b_y):
    obj = jnp.trace(jnp.inner(X, A_x@X + 2*jnp.stack([b_x,b_y],axis=1))) + jnp.trace(jnp.inner(L, X.T@X - C))
    return obj.real

@jit
def f_l_sqp(X, L, C, A, E_0):
    obj = A@X + E_0 + X@L
    return obj.real

def g(X, v, c):
    return np.array([v.T@X[:,0], v.T@X[:,1]]) - c

def h(X, D, c1, c2, c3, c=jnp.array([0,0])):
    return np.array([(X[:,0]-c[0]).T@D@(X[:,0]-c[0]) - c1, 
                     (X[:,1]-c[1]).T@D@(X[:,1]-c[1]) - c2, 
                     2*((X[:,0]-c[0]).T@D@(X[:,1]-c[1]) - c3)])

def wl(X, A_x, A_y):
    return X[:,0].T@A_x@X[:,0] + X[:,1].T@A_y@X[:,1]

