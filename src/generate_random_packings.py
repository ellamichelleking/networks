import hoomd
import gsd.hoomd
import numpy as np
import math
import itertools
import os

#Specify parameters
N_particles=10000
particle_radius = 1.4
spacing = 1.6 #initial interparticle separation
final_rho = 1.0 #final density after compression

# Initialize the system
K = math.ceil(N_particles ** (1 / 2))
L = K * spacing
x = np.linspace(-L / 2, L / 2, K, endpoint=False)
position = list(itertools.product(x, repeat=2))
position = np.concatenate((position, np.zeros((N_particles,1))), axis=1)
position = position[0:N_particles]

# Save initial configuration
frame = gsd.hoomd.Frame()
frame.particles.N = N_particles
frame.particles.position = position
frame.particles.typeid = [0] * N_particles
frame.particles.types = ['A', 'B']
IDs = np.array([0] * int(N_particles / 2) + [1] * int(N_particles / 2)).flatten()
np.random.shuffle(IDs)
frame.particles.typeid = IDs
frame.configuration.box = [L, L, 0, 0, 0, 0]
if not os.path.isfile('lattice.gsd'):
  with gsd.hoomd.open(name='lattice.gsd', mode='x') as f:
      f.append(frame)


# Preparation: Create a MD simulation.
#simulation = hoomd.util.make_example_simulation()
simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
simulation.create_state_from_gsd(filename='lattice.gsd')


# Step 1: Use hoomd.md.minize.FIRE as the integrator.
'''
constant_volume = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
fire = hoomd.md.minimize.FIRE(dt=0.001,
                              force_tol=1e-3,
                              angmom_tol=1e-3,
                              energy_tol=1e-6,
                              methods=[constant_volume])
simulation.operations.integrator = fire
'''
integrator = hoomd.md.Integrator(dt=0.005)
nvt = hoomd.md.methods.ConstantVolume(
    filter=hoomd.filter.All(), thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5)
)
integrator.methods.append(nvt)
simulation.operations.integrator = integrator


# Step 2: Apply forces to the particles.
lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4))
lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
lj.r_cut[('A', 'A')] = 2.5
lj.params[('A', 'B')] = dict(epsilon=1.0, sigma=(particle_radius + 1.0)/2.)
lj.r_cut[('A', 'B')] = 2.5
lj.params[('B', 'B')] = dict(epsilon=1.0, sigma=particle_radius)
lj.r_cut[('B', 'B')] = 2.5
simulation.operations.integrator.forces = [lj]

# Randomize the positions
simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)
simulation.run(1e4)

# Compress the system
ramp = hoomd.variant.Ramp(A=0, B=1, t_start=simulation.timestep, t_ramp=10000)
rho = simulation.state.N_particles / simulation.state.box.volume
initial_box = simulation.state.box
final_box = hoomd.Box.from_box(initial_box)
final_box.volume = simulation.state.N_particles / final_rho
box_resize_trigger = hoomd.trigger.Periodic(10)
box_resize = hoomd.update.BoxResize(
    box1=initial_box, box2=final_box, variant=ramp, trigger=box_resize_trigger
)
simulation.operations.updaters.append(box_resize)


# Step 3: Run simulation steps until the minimization converges.
'''
while not fire.converged:
    simulation.run(100)
'''
simulation.run(1e4)

final_snapshot = simulation.state.get_snapshot()
final_pos = final_snapshot.particles.position
np.save('hoomdrcp_finalpos.npy', final_pos, allow_pickle=False)

