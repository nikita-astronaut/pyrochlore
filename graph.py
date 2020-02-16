import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import sys

h = np.sqrt(2. / 3.)
r = np.sqrt(1. / 3.)

E = np.zeros((3, 3))
E[:, 0] = np.array([r * np.cos(0), r * np.sin(0), h]) * 2
E[:, 1] = np.array([r * np.cos(2 * np.pi / 3), r * np.sin(2 * np.pi / 3), h]) * 2
E[:, 2] = np.array([r * np.cos(4 * np.pi / 3), r * np.sin(4 * np.pi / 3), h]) * 2

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def add_unique(sites, new_site):
    if new_site not in sites:
        sites.append(new_site)
    return sites

def build_pyrochlore_graph(n_x, n_y, n_z):
    sites_A = []
    sites_B = []
    sites_C = []
    sites_D = []

    for x_super in range(n_x):
        for y_super in range(n_y):
            for z_super in range(n_z):
                vec = np.array([x_super, y_super, z_super])  # actually these are not x-y-z but rather 1-2-3
                A_site = np.einsum('ij,j->i', E, vec); sites_A.append(A_site)
                B_site = np.einsum('ij,j->i', E, vec) + E[:, 0] / 2; sites_B.append(B_site)
                C_site = np.einsum('ij,j->i', E, vec) + E[:, 1] / 2; sites_C.append(C_site)
                D_site = np.einsum('ij,j->i', E, vec) + E[:, 2] / 2; sites_D.append(D_site)
    return np.array(sites_A), np.array(sites_B), np.array(sites_C), np.array(sites_D)

def get_bc_copies(site, n_x, n_y, n_z):
    copies = []
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                shift = dx * E[:, 0] * n_x + dy * E[:, 1] * n_y + dz * E[:, 2] * n_z

                copies.append(site + shift)
    return copies

def get_J1J2_edges(sites, n_x, n_y, n_z):
    edges = []
    physical_edges = []
    for idx1, first in enumerate(sites):
        for idx2, second in enumerate(sites):
            copies = get_bc_copies(second, n_x, n_y, n_z)
            distances = [np.sum((np.array(first) - np.array(copy)) ** 2) for copy in copies]
            idx = np.argmin(distances)
            distance = distances[idx]
            copy = copies[idx]  # the copy that actually gave the minimum value
            add = ''
            if np.sum(np.abs(second - copy)) > 0.:
                add = 'b'
            
            if (np.abs(distance - 1.) < 1e-5) and ((idx1, idx2, 'J1' + add) not in edges) and ((idx2, idx1, 'J1' + add) not in edges):
                edges.append((idx1, idx2, 'J1' + add))
                physical_edges.append((first, copy))
            if (np.abs(distance - 3.) < 1e-5) and ((idx1, idx2, 'J2' + add) not in edges) and ((idx2, idx1, 'J2' + add) not in edges):
                edges.append((idx1, idx2, 'J2' + add))
                physical_edges.append((first, copy))
    return edges, physical_edges

n_x, n_y, n_z = 2, 2, 2
sites_A, sites_B, sites_C, sites_D = build_pyrochlore_graph(n_x, n_y, n_z)
sites = np.concatenate([sites_A, sites_B, sites_C, sites_D])
edges, physical_edges = get_J1J2_edges(sites, n_x, n_y, n_z)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

ax.plot(sites_A[:, 0], sites_A[:, 1], sites_A[:, 2], '*', markersize=15, color='g', alpha=0.4)
ax.plot(sites_B[:, 0], sites_B[:, 1], sites_B[:, 2], '*', markersize=15, color='r', alpha=0.4)
ax.plot(sites_C[:, 0], sites_C[:, 1], sites_C[:, 2], '*', markersize=15, color='b', alpha=0.4)
ax.plot(sites_D[:, 0], sites_D[:, 1], sites_D[:, 2], '*', markersize=15, color='orange', alpha=0.4)

print(len(edges))
for edge, pedge in zip(edges, physical_edges):
    fi, si, j = edge
    f, s = pedge
    ax.text(*f, fi, color='black')
    ax.text(*s, si, color='black')
    a = Arrow3D([f[0], s[0]], [f[1], s[1]], [f[2], s[2]], mutation_scale=20, ls = '-' if j[-1] != 'b' else '--',
                lw=1 if j[1] == '1' else 0.5, arrowstyle="-", color = 'r' if j[1] == '1' else 'b')
    ax.add_artist(a)

plt.show()

edges_j1 = [(e[0], e[1]) for e in edges if e[2][1] == '1']
edges_j2 = [(e[0], e[1]) for e in edges if e[2][1] == '2']
print(edges_j1, edges_j2)
f = open('hamiltonian.dat', 'w')
f.write(str(edges_j1)); f.write('\n\n'); f.write(str(edges_j2))
f.close()
