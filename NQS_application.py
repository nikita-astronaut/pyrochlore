import netket as nk
import torch
import numpy as np
import json
import ast

def test_oddness(model, dims, odds, dim):
    tensor = torch.arange(4 * dims[0] * dims[1] * dims[2])
    f_ini = model(tensor.view(1, -1).type(torch.FloatTensor))

    tensor = tensor.view(1, 4, *dims)
    tensor = torch.roll(torch, 1, dim + 2)
    f_fin = model(tensor.view(1, -1).type(torch.FloatTensor))
    signs = ((f_fin / f_ini + 1) / 2.).detach().numpy().astype(np.int64).astype(np.bool_)
    if signs[0] == odds[dim] and signs[1] == odds[dim]
        print('Oddness in {:d} direction confirmed: {:b}'.format(dim, odds[dim]))
        return True
    return False


class Net_1body_simple(torch.nn.Module):
    def __init__(self, nx:int, ny:int, nz:int):
        super().__init__()
        self._conv1 = torch.nn.Conv3d(4, 16, (nx, ny, nz), stride=1, padding = 0, dilation=1, groups=1, bias=True)
        self._conv2 = torch.nn.Conv3d(16, 32, (nx, ny, nz), stride=1, padding = 0, dilation=1, groups=1, bias=True)
        self._dense = torch.nn.Linear(32, 2, bias=False)

        self.nx = nx
        self.ny = ny
        self.nz = nz

    def pad_circular(self, x): # x[Nbatch, 1, W, H] -> x[Nbatch, 1, W + 2 pad, H + 2 pad] (pariodic padding)
        if self.nx > 1:
            x = torch.cat([x, x[:, :, 0:self.nx - 1, :, :]], dim=2)
        if self.ny > 1:
            x = torch.cat([x, x[:, :, :, 0:self.ny - 1, :]], dim=3)
        if self.nz > 1:
            x = torch.cat([x, x[:, :, :, :, 0:self.nz - 1]], dim=4)

        return x

    def forward(self, x):
        x = x.view((x.shape[0], 4, self.nx, self.ny, self.nz))
        x = self.pad_circular(x)
        x = self._conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pad_circular(x)
        x = self._conv2(x)
        x = torch.nn.functional.relu(x)
        x = x.view(x.shape[0], 4 * self.nx * self.ny * self.nz, -1).mean(dim = 2)

        x = self._dense(x)

        return x



class Net_1body_custom(torch.nn.Module):
    def __init__(self, nx:int, ny:int, nz:int, x_odd: bool, y_odd: bool, z_odd: bool):
        super().__init__()
        self._conv1 = torch.nn.Conv3d(4, 16, (nx, ny, nz), stride=1, padding = 0, dilation=1, groups=1, bias=True)
        self._conv2 = torch.nn.Conv3d(16, 32, (nx, ny, nz), stride=1, padding = 0, dilation=1, groups=1, bias=True)
        self._dense = torch.nn.Linear(32, 1, bias=False)

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.mask = torch.ones(nx, ny, nz)
        if x_odd:
            self.mask[torch.arange(0, nx, 2), ...] *= -1
        if y_odd:
            self.mask[:, torch.arange(0, ny, 2), :] *= -1
        if z_odd:
            self.mask[:, :, torch.arange(0, nz, 2)] *= -1
        self.mask = self.mask.type(torch.FloatTensor)

    def pad_circular(self, x): # x[Nbatch, 1, W, H] -> x[Nbatch, 1, W + 2 pad, H + 2 pad] (pariodic padding)
        if self.nx > 1:
            x = torch.cat([x, x[:, :, 0:self.nx - 1, :, :]], dim=2)
        if self.ny > 1:
            x = torch.cat([x, x[:, :, :, 0:self.ny - 1, :]], dim=3)
        if self.nz > 1:
            x = torch.cat([x, x[:, :, :, :, 0:self.nz - 1]], dim=4)

        return x

    def forward(self, x):
        x = x.view((x.shape[0], 4, self.nx, self.ny, self.nz))
        x = self.pad_circular(x)
        x = self._conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pad_circular(x)
        x = self._conv2(x)
        x = torch.nn.functional.relu(x)
        #x = pad_circular(x, 4, 2)
        #x = self._conv3(x)
        #x = torch.nn.functional.relu(x)
        ampl = x.view(x.shape[0], 4 * self.nx * self.ny * self.nz, -1).mean(dim = 2)
        phase = (x * self.mask).view(x.shape[0], 4 * self.nx * self.ny * self.nz, -1).mean(dim = 2)
        ampl = self._dense(ampl)
        phase = self._dense(phase)

        phase = (1. + torch.sign(phase)) / 2. * np.pi + torch.abs(phase)

        return torch.cat([ampl, phase], dim = 1)


varphi = float(sys.argv[1])
nx, ny, nz = [int(x) for x in sys.argv[2:5]]
odd_x, odd_y, odd_z = [bool(x) for x in sys.argv[5:]]

j1_edges = ast.literal_eval(open('./hamiltonians_j1_{:d}x{:d}x{:d}.dat'.format(nx, ny, nz), 'r').read())
j2_edges = ast.literal_eval(open('./hamiltonians_j2_{:d}x{:d}x{:d}.dat'.format(nx, ny, nz), 'r').read())

'''
sites order: first 8 are the A--sublattice, then B-siblattice at cetera
within the same sublattice, the order is as follows: (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), ...
'''

J = [np.cos(varphi), np.sin(varphi)]

all_edges = [[e[0], e[1], 1] for e in j1_edges] + [[e[0], e[1], 2] for e in j2_edges]
g = nk.graph.CustomGraph(all_edges)
hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, graph=g)

# Pauli Matrices
sigmaz = [[1, 0], [0, -1]]
sigmax = [[0, 1], [1, 0]]
sigmay = [[0, -1j], [1j, 0]]

# Bond Operator
interaction = np.kron(sigmaz, sigmaz) + np.kron(sigmax, sigmax) + np.kron(sigmay, sigmay)  

bond_operator = [
    (J[0] * interaction).tolist(),
    (J[1] * interaction).tolist(),
]
bond_color = [1, 2]
go = nk.operator.GraphOperator(hi, bondops=bond_operator, bondops_colors=bond_color)
len(g.edges)


if not odd_x and not odd_y and not odd_z:
    model = Net_1body_simple(nx, ny, nz)
else:
    model = Net_1body_custom(nx, ny, nz, odd_x, odd_y, odd_z)

success_test = True
for dim in range(3):
    success_test = success_test and test_oddness(model, [nx, ny, nz], [odd_x, odd_y, odd_z], dim)
if success_test:
    print('The network has the right momentum properties')
else:
    exit(-1)


ma = nk.machine.Torch(model, hilbert=hi)
ma.parameters = 0.1 * (np.random.randn(ma.n_par))
sa = nk.sampler.MetropolisHamiltonian(machine=ma, n_chains=48, hamiltonian = go)
op = nk.optimizer.Sgd(learning_rate=0.01)

gs = nk.variational.Vmc(
    hamiltonian=go, sampler=sa, optimizer=op, n_samples=600, method="Sr"
)


for i in range(1000):
    gs.run(output_prefix='./logs/test_{:d}x{:d}x{:d}_'.format(nx, ny, nz) + sys.argv[1], n_iter=10)
    ma.save('/home/cluster/niastr/data/pyrochlore/models/{:d}x{:d}x{:d}_'.format(nx, ny, nz) + sys.argv[1] + '.data')
    print(json.load(open('/home/cluster/niastr/data/pyrochlore/logs/test_{:d}x{:d}x{:d}_'.format(nx, ny, nz) + sys.argv[1] + '.log', 'r'))['Output'], flush = True)

