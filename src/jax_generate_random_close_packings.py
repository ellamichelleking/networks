import numpy as np
import jax.numpy as jnp
from jax import grad, vmap, jit, random
from jax_md import space, smap, energy, minimize, quantity, simulate

N=100
N_steps=200
box_size = quantity.box_size_at_number_density(N, 1.1, 2)
displacement, shift = space.periodic(box_size)

key = random.PRNGKey(0)
R = box_size * random.uniform(key, (N, 2), dtype=jnp.float32)

sigmas = jnp.array([np.random.normal(loc=0.08, scale=0.01, size=N)])
sigma_mat = (sigmas + sigmas.T)/2

energy_fn = energy.soft_sphere_pair(displacement, species=np.arange(N), sigma=sigma_mat)
fire_init, fire_apply = minimize.fire_descent(energy_fn, shift)
fire_apply = jit(fire_apply)
fire_state = fire_init(R)

E = []
#trajectory = []

for i in range(N_steps):
  fire_state = fire_apply(fire_state)
  
  E += [energy_fn(fire_state.position)]
  #trajectory += [fire_state.position]
  
R = fire_state.position
np.save(f'../lattices/fire_descent_N{N}.npy', R, allow_pickle=False)
#trajectory = np.stack(trajectory)
