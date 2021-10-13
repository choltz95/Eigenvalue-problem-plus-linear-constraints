from typing import Any, Callable, NamedTuple, Tuple, Union
Step = int
Schedule = Callable[[Step], float]

import scipy.io as io

import sys, os, shutil, glob
from PIL import Image

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

import utils
from objectives import *

from functools import partial
import itertools

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib import collections as mc
import seaborn as sns

import datetime
from tqdm.notebook import tqdm

import networkx as nx


"""====Matrix utilities==== """

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

"""====Graph utilities==== """

def load_graph(graphpath, A=None, plot_adjacency=False, verbose=True):
    if A is None:
        mat_data = io.loadmat(graphpath + '.mat')
        graph = mat_data['Problem']['A'][0][0]
        G = nx.from_numpy_matrix(graph.toarray().astype(int)!= 0, create_using=None)
        A = nx.adjacency_matrix(G).toarray().astype(np.int16)
    else:
        G = nx.from_numpy_matrix(A.astype(int)!= 0, create_using=None)
        graph = sp.sparse.csc_matrix(A)
    L = csgraph_laplacian(A, normed=False).astype(np.float32)
    D = np.diag(np.sum(A, axis=1)).astype(np.int16)
    n = A.shape[0]
    
    if verbose:
        print(nx.info(G))
    if plot_adjacency:
        plot_adjacency(A)
        
    return graph, G, A, L, D, n

"""====Voxel clustering==== """

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

"""====Visualization utilities==== """

def plot_results(result,sigfig=2):
    """result keys: x, lossh, sln_path, g, h, step_sizes"""
    fig, axes = plt.subplots(
    3, 2, figsize=(25, 8), sharex=True)
    
    foc = result['foc']
    log_foc = np.log(foc)
    step_sizes = result['step_sizes']
    loss_history = result['lossh']
    log_loss_history = np.log(loss_history)
    min_loss = np.min(loss_history)
    min_logloss = np.min(log_loss_history)
    min_loss_idx = np.argmin(loss_history)
    
    gc = np.round(result['g'], sigfig)
    hc = np.round(result['h'], sigfig)
    
    axes[0,0].plot(loss_history)
    axes[0,0].set_title('loss: {:.3f} h: {} g: {}'.format(min_loss, np.round(hc,sigfig), np.round(gc,sigfig)))
     
        
    axes[0,1].plot(log_loss_history)
    axes[0,1].set_title('log-loss: {:.3f}'.format(min_logloss))
     
    axes[1,0].plot(step_sizes)
    axes[1,0].set_title('step sizes')
    axes[1,1].plot(np.log(step_sizes))
    axes[1,1].set_title('log-step sizes')
    
    axes[2,0].plot(foc)
    axes[2,0].set_title('first order condtion: initial foc: {:.3f}, final foc: {:.3f}, min-loss foc: {:.3f}'.format(foc[0], foc[-1], foc[min_loss_idx]))
    axes[2,1].plot(log_foc)
    axes[2,1].set_title('log-first order condtion: initial foc: {:.3f}, final foc: {:.3f}, min-loss foc: {:.3f}'.format(log_foc[0], log_foc[-1],log_foc[min_loss_idx]))    
    
    for ax in axes:
        ax[0].axvline(x=min_loss_idx, c='gray')
        ax[1].axvline(x=min_loss_idx, c='gray')
     
    #pap = result['P']@L@result['P'].T
    #print(result['L'][-1].real,
    #np.linalg.eig(['L'][-1])[0].real, 
    #np.sort(np.linalg.eig(pap)[0])[:10].real,
    #1.0 - ( np.count_nonzero(pap) / float(pap.size) ), 1.0 - ( np.count_nonzero(L) / float(L.size) ))
    
    trans = mtrans.blended_transform_factory(fig.transFigure,
                                         mtrans.IdentityTransform())

    txt = fig.text(.5, 15, "second order condition: {}", ha='center')
    txt.set_transform(trans)
    return fig

def plot_adjacency(A):
    fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(10,20))
    ax1.imshow(graph.todense()!=0, cmap='gray')
    ax1.set_title("adjacency")
    graphdist = csgraph.shortest_path(graph, directed=False, unweighted=True)
    ax2.set_title("graph distances")
    ax2.imshow(graphdist, cmap='gray')
    
    return fig
    
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

def plot_animation(results, graph, fixed_coordinates=None, directory_name='./frames/', numframes=100):
    """Generate animation frames """

    if os.path.exists(directory_name):
        shutil.rmtree(directory_name)
    os.makedirs(directory_name)

    plt.figure(figsize=(20,10))
    ax = plt.axes()

    for l, result in enumerate(results):
        param_hist = result['sln_path']
        idx = np.linspace(0, len(param_hist)-1, num=min(numframes,len(param_hist)),dtype=int)
        P_tmp = result['P']
        P_x_tmp = P_tmp
        P_y_tmp = P_tmp
        n0_x_tmp, n0_y_tmp = result['n']
        for k in idx:
            X_k_tmp = param_hist[k]       
            X_k_n_tmp = np.zeros((n0_x_tmp.shape[0],2))
            X_k_n_tmp[:,0] = np.array(P_x_tmp.T@X_k_tmp[:,0]) + n0_x_tmp.T
            X_k_n_tmp[:,1] = np.array(P_y_tmp.T@X_k_tmp[:,1]) + n0_y_tmp.T
            positions_tmp = X_k_n_tmp
            
            #### TEMPORARY
            X_k_n_tmp = np.concatenate([fixed_coordinates, positions_tmp])
            ####
            
            
            
            voxel_id, voxel_bound = voxel_cluster(X_k_n_tmp, np.array([5, 5]))

            ax.clear()

            ax = utils.plot_graph(X_k_n_tmp, graph, c=voxel_id)

            plt.savefig(directory_name+'{}_{}.png'.format(l, k))
            # only needed for inline animation
            #display.clear_output(wait=True)
            #display.display(plt.gcf())
        
    # save animation as gif
    # filepaths
    fp_in = directory_name+"*.png"
    fp_out = directory_name+"animation.gif"

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in),key=os.path.getmtime)]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=500, loop=0)