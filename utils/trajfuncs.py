import pandas as pd
import numpy as np

import elpigraph

from tqdm import trange



def point_to_segment_dist(point, seg_start, seg_end, bound_start=True, bound_end=True):
	point = np.asarray(point)
	seg_start = np.asarray(seg_start)
	seg_end = np.asarray(seg_end)

	delta_seg = seg_end - seg_start
	epsilon = ((point - seg_start) * delta_seg).sum() / (np.linalg.norm(delta_seg) ** 2)
	if epsilon > 1 and bound_end:
		epsilon = 1
	elif epsilon < 0 and bound_start:
		epsilon = 0
	d = np.linalg.norm(point - seg_start - delta_seg * epsilon)
	proj_point = seg_start + epsilon * delta_seg
	return d, epsilon, proj_point


def point_to_path_dist(point, path, bound_start=True, bound_end=True):
	epsilons = np.zeros(path.shape[0] - 1)
	dists = np.zeros(path.shape[0] - 1)
	point_projs = np.zeros((path.shape[0] - 1, path.shape[1]))
	for i in range(path.shape[0] - 1):
		d, eps, proj = point_to_segment_dist(point, path[i, :], path[i + 1, :], bound_start=bound_start, bound_end=bound_end)
		dists[i] = d
		epsilons[i] = eps
		point_projs[i, :] = proj
	lengths = np.sqrt(((path[1:, :] - path[:-1, :]) ** 2).sum(axis=1))
	tot_length = lengths.sum()
	i_min = np.argmin(dists)
	traj_pos = (lengths[:i_min].sum() + lengths[i_min] * epsilons[i_min]) / tot_length
	return traj_pos, dists.min(), point_projs[np.argmin(dists), :]


def proj_point_on_trajectory(point, node_pos, edges, traj):
	traj = np.array(traj)
	traj_edges = np.zeros(traj.shape[0] - 1)
	for i in range(traj.shape[0] - 1):
		traj_edges[i] = np.where(((edges == (traj[i], traj[i + 1])) | (edges == (traj[i+1], traj[i]))).all(axis=1))[0][0]

	epsilons = np.zeros(edges.shape[0])
	dists = np.zeros(edges.shape[0])
	point_projs = np.zeros((edges.shape[0], node_pos.shape[1]))
	for i in range(edges.shape[0]):
		d, eps, proj = point_to_segment_dist(point, node_pos[edges[i, 0], :], node_pos[edges[i, 1]])
		dists[i] = d
		epsilons[i] = eps
		point_projs[i, :] = proj
	i_min = (dists == dists.min()).nonzero()[0].max()
	if i_min not in traj_edges:
		return None
	else:
		path = node_pos[traj, :]
		return point_to_path_dist(point, path)




def get_projs(embeds,traj, bound_start=True, bound_end=True):
	projs = pd.DataFrame(np.asarray([point_to_path_dist(embeds.iloc[i].values, traj, bound_start=bound_start, bound_end=bound_end)[0:2] for i in trange(len(embeds))]),columns=['traj_pos','traj_dist'],index=embeds.index)
	return projs




def get_tree(embeds,n_nodes=30,collapse=True,MaxNumberOfIterations=30,nReps=1):
	if nReps > 1:
		obj = elpigraph.computeElasticPrincipalTree(embeds.values,n_nodes,Lambda=0.05,verbose=True,Do_PCA=True,nReps=nReps,ProbPoint=0.8,MaxNumberOfIterations=MaxNumberOfIterations,AvoidResampling=False)
	else:
		obj = elpigraph.computeElasticPrincipalTree(embeds.values, n_nodes, Lambda=0.05, verbose=True, Do_PCA=True,nReps=nReps, ProbPoint=1, MaxNumberOfIterations=MaxNumberOfIterations,AvoidResampling=False)
	graph = obj[-1]
	if collapse:
		collapsed = elpigraph.CollapseBranches(embeds.values,graph,Mode='EdgesNumber',ControlPar=2)
		graph['NodePositions'] = collapsed['Nodes']
		graph['Edges'] = [collapsed['Edges'],np.zeros(len(collapsed['Edges'])),np.zeros(len(collapsed['Edges']))]
	return graph