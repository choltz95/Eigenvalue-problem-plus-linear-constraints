# Eigenvalue-problem-plus-linear-constraints
Robust Rayleigh quotient minimization + linear constraints applied to graph embedding in Jax.

A bunch of ml problems on undirected graphs (e.g. graph clustering or embedding) can be formulated as finding the eigenvalues of different matrices that characterize graph connectivity (e.g. Adjacency & Laplacian). Under certain assumptions on the connectivity matrix, a variational perspective of eigenvalues implies minimizing a sequence, or sum, of quadratic forms subject to orthogonality (quadratic) constraints.

In many situations, it might also be nice to integrate linear constraints and linear terms in the objective. For example, in graph embedding we may have some prior information about the coordinates of anchor nodes in the embedding space. In graph clustering, we may want to leverage label information by constraining nodes of the same class to be assigned to the same cluster.

In rayleigh.ipynb are two algorithms: a locally convergent projected gradient method + adaptive momentum and the globally convergent sequential quadratic programming algorithm + sequential subspace method (SSM) method originally proposed in Hager, Minimizing a Quadratic Over a Sphere, 01 (currently just the newton direction is implemented).

Both methods are applied to a toy graph (qh882) visualization problem. In this context, the objective + constraints have a nice intuitive meaning. Minimizing the quadratic objective = minimizing the squared distance between connected nodes in the embedding space. Quadratic constraints = spreading nodes out, and linear constraints = the position of fixed nodes, etc.

utils.py includes matrix utilities, optimization functionals (adaptive momentum & one-step updates), a simple voxel clustering algorithm, and a graph plotter.

check ev_init_pgd_top_level.gif for an animation of the pgd procedure.