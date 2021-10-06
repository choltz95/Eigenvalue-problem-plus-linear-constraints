X_k = X_k_n
width = 10
size = np.array([width/2, width/2])

# levels
for _ in tqdm(range(2)): 
    size = size/2
    voxel_id, voxel2bound = voxel_cluster(X_k, size)
    unique_voxels, unique_vidx = np.unique(voxel_id, return_index=True)
    
    num_voxels_per_row = width/size[0]
    num_voxels=num_voxels_per_row**2
    
    for _ in range(5):
        np.random.shuffle(unique_vidx)
        
        for vidx in unique_vidx:
            vid = voxel_id[vidx]
            fixed_indices = (~clust_to_mask(voxel_id, vid).astype(bool)).astype(int)
            #fixed_indices = (~np.sum([clust_to_mask(voxel_id, vid), 
            #                          clust_to_mask(voxel_id, min(vid+1, num_voxels_per_row//(vid+1)*num_voxels_per_row))], axis=0).astype(bool)).astype(int) 
            #fixed_indices = (~np.sum([clust_to_mask(voxel_id, vid), 
            #                          clust_to_mask(voxel_id, min(vid+1, num_voxels_per_row//(vid+1)*num_voxels_per_row)),
            #                          clust_to_mask(voxel_id, min(vid+num_voxels_per_row, num_voxels)), 
            #                          clust_to_mask(voxel_id, min(vid+num_voxels_per_row+1, num_voxels))], axis=0).astype(bool)).astype(int) 

            if (1-fixed_indices).sum() < 4:
                continue

            num_move = (1-fixed_indices).sum()
            cc1 = num_move*size[0]**2*1/12
            cc2 = num_move*size[1]**2*1/12
            #cc1 = num_move*(2*size[0])**2*1/12
            #cc2 = num_move*(2*size[1])**2*1/12
            cc3 = 0

            C = jnp.block([[cc1, cc3],[cc3, cc2]])

            offset = (np.max(X_k,axis=0) - np.min(X_k,axis=0))/2
            centroid = voxel2bound[vidx]*size + size/2 - offset
            #centroid = jax.ops.index_add(centroid, 0, size[0]/2)
            #centroid = voxel2bound[vidx]*size - offset
            #centroid = np.mean(X_k[(1-fixed_indices)],axis=0)

            method = "pgd"
            if method == "pgd":
                opt_init, opt_update, get_params = padam(1e-2, partial(lambda x, y: project(y,x),C), 
                                                         b1=0.9, b2=0.999, eps=1e-08)
            elif method == "pnd":
                opt_init, opt_update, get_params = psgd(partial(lambda x, y: project(y,x),C))


            result = cluster(rng, (opt_init, opt_update, get_params), 
                             X_k, L, np.where(fixed_indices)[0], maxiters=1000, c1=cc1, c2=cc2, c3=0, centroid=centroid, 
                             centercons=num_move*centroid, v=None, D=None, eps=1e-8, 
                             convergence_criterion=1e-3, method=method)

            results.append(result)
            X_k = result['x']
            gc = result['g']
            hc = result['h']
            loss = result['lossh']
            #param_hist.extend(result['sln_path'])
            print('loss: {} h: {} g: {}'.format(str(np.round(loss[-1],2)), np.round(hc,2), np.round(gc,2)))   

            
utils.plot_graph(X_k, graph, c=voxel_id,title=str(np.round(wl(X_k, A, A),3)))


############################################


_, unique_vidx = np.unique(voxel_id, return_index=True)
X_k = X_k_n
#opt_init, opt_update, get_params = padam(1e-1,partial(lambda x, y: project(y,x),C), b1=0.9, b2=0.999, eps=1e-08)
#opt_init, opt_update, get_params = padam(1e-1,lambda x: x, b1=0.9, b2=0.999, eps=1e-08)

for vidx in unique_vidx:
    vid = voxel_id[vidx]
    fixed_indices = (~clust_to_mask(voxel_id, vid).astype(bool)).astype(int)
    #if (1-fixed_indices).sum() < 100:
    #    continue
    
    num_move = (1-fixed_indices).sum()
    cc1 = num_move*size[0]**2*1/12
    cc2 = num_move*size[1]**2*1/12
    cc3 = 0
    
    C = jnp.block([[cc1, cc3],[cc3, cc2]])
        
    offset = (np.max(positions,axis=0) - np.min(positions,axis=0))/2
    centroid = voxel2bound[vidx]*size + size/2 - offset
    
    #opt_init, opt_update, get_params = padam(1e-4,partial(lambda x, y, z: project(z, x, y), C, centroid), 
    #                                         b1=0.9, b2=0.999, eps=1e-08)
    #opt_init, opt_update, get_params = padam(1e-2,lambda x : x, 
    #                                         b1=0.9, b2=0.999, eps=1e-08)
    opt_init, opt_update, get_params = psgd(partial(lambda x, y: project(y,x),C))


    result = cluster(rng, (opt_init, opt_update, get_params), 
                     X_k, L, np.where(fixed_indices)[0], maxiters=5000, c1=cc1, c2=cc2, c3=0, centroid=centroid, 
                     centercons=num_move*centroid, v=None, D=None, eps=1e-8)
    results.append(result)
    X_k = result['x']
    gc = result['g']
    hc = result['h']
    loss = result['lossh']
    #param_hist.extend(result['sln_path'])
    print('loss: {} h: {} g: {}'.format(str(np.round(loss[-1],2)), np.round(hc,2), np.round(gc,2)))    
    
    
"""BFS"""
"""
def BFS(i, maxdepth, X_k, results):
    if i > maxdepth:
        return results
    size = np.array([10, 10])
    voxel_id, voxel2bound = voxel_cluster(X_k_n, size)
    _, unique_vidx = np.unique(voxel_id, return_index=True)
    
    for vidx in unique_vidx:
        vid = voxel_id[vidx]
        fixed_indices = (~clust_to_mask(voxel_id, vid).astype(bool)).astype(int)

        num_move = (1-fixed_indices).sum()
        cc1 = num_move*size[0]**2*1/12
        cc2 = num_move*size[1]**2*1/12
        cc3 = 0

        C = jnp.block([[cc1, cc3],[cc3, cc2]])

        offset = (np.max(positions,axis=0) - np.min(positions,axis=0))/2
        centroid = voxel2bound[vidx]*size + size/2 - offset

        result = cluster(rng, (opt_init, opt_update, get_params), 
                     X_k, L, np.where(fixed_indices)[0], maxiters=5000, c1=cc1, c2=cc2, c3=0, centroid=centroid, 
                     centercons=num_move*centroid, v=None, D=None, eps=1e-8)

        results.append(result)
        X_k = result['x']
        gc = result['g']
        hc = result['h']
        loss = result['lossh']

        print('loss: {} h: {} g: {}'.format(str(np.round(loss[-1],2)), np.round(hc,2), np.round(gc,2)))

        DFS(i+1, maxdepth, X_k, results)
"""
