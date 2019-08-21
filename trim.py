# pylint: disable=invalid-name, too-many-locals, missing-docstring, too-many-arguments, redefined-outer-name
import time
import matplotlib.pyplot as plt
import casadi as ca
import numpy as np
import f16


# %%
start = time.time()
p = f16.Parameters()
x0, u0 = f16.trim(
    s0=[0, 0, 0, 0, 0, 0],
    x=f16.State(VT=500, alt=5000),
    p=p,
    phi_dot=0, theta_dot=0, psi_dot=0.2, gam=0)
print('trim computation time', time.time() - start)
# %%
start = time.time()
data = f16.simulate(x0, u0, p, 0, 10, 0.01)
print('sim computation time', time.time() - start)

state_index = f16.State().name_to_index

plt.figure()
plt.plot(data['x'][:, state_index('p_E')], data['x'][:, state_index('p_N')])
plt.xlabel('E, ft')
plt.ylabel('N, ft')
plt.show()

plt.figure()
plt.plot(data['t'], data['x'][:, state_index('alpha')], label='alpha')
plt.plot(data['t'], data['x'][:, state_index('beta')], label='beta')
plt.plot(data['t'], data['x'][:, state_index('theta')], label='theta')
plt.legend()
plt.show()

plt.figure()
plt.plot(data['t'], data['x'][:, state_index('VT')], label='VT')
plt.legend()
plt.show()


plt.figure()
plt.plot(data['t'], data['x'][:, state_index('phi')], label='phi')
plt.plot(data['t'], data['x'][:, state_index('theta')], label='theta')
plt.plot(data['t'], data['x'][:, state_index('psi')], label='psi')
plt.legend()
plt.show()


# plt.figure()
#plt.plot(data['t'], data['x'][:, state_index('p_E')])

p = f16.Parameters()
u_list = []
VT_list = np.arange(100, 800, 50)
for VT in VT_list:
    x0, u0 = trim(np.zeros(6), f16.State(VT=VT), p, 0, 0, 0, 0)
    u_list.append(np.array(u0.to_casadi()))
u_list = np.hstack(u_list)
plt.plot(VT_list, 100*u_list[0, :])
plt.xlabel('VT, ft/s')
plt.ylabel('power, %')
# %%
