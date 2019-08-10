#%% [markdown]

# # F16 Rollrate Compensator

# This notebooks gives an example of F16 rollrate compensator design.


#%%
import f16
import casadi as ca
import numpy as np
import control
import matplotlib.pyplot as plt


def linearize(x0, u0, p0):
    x0 = x0.to_casadi()
    u0 = u0.to_casadi()# Plot the compensated openloop bode plot

    x_sym = ca.MX.sym('x', x0.shape[0])
    u_sym = ca.MX.sym('u', u0.shape[0])
    x = f16.State.from_casadi(x_sym)
    u = f16.Control.from_casadi(u_sym)
    tables = f16.build_tables()
    dx = f16.dynamics(x, u, p0, tables)
    A = ca.jacobian(dx.to_casadi(), x_sym)
    B = ca.jacobian(dx.to_casadi(), u_sym)
    f_A = ca.Function('A', [x_sym, u_sym], [A])
    f_B = ca.Function('B', [x_sym, u_sym], [B])
    A = f_A(x0, u0)
    B = f_B(x0, u0)
    n = A.shape[0]
    p = B.shape[1]
    C = np.eye(n)
    D = np.zeros((n, p))
    return control.ss(A, B, C, D)

#%%
# trim the f16 in steady level flight
p0 = f16.Parameters(xcg=0.38)
x0 = f16.State(VT=502, alpha=0.03544, theta=0.03544)
u0 = f16.Control(thtl=0.1325, elv_deg=-0.0559)
tables = f16.build_tables()
x0.power = tables['tgear'](u0.thtl)
dx = f16.dynamics(x0, u0, p0, tables)

#%%
# create a state mapping to make getting transfer functions easier

y_id = {
    'VT': 0,
    'alpha': 1,
    'beta': 2,
    'phi': 3,
    'theta': 4,
    'psi': 5,
    'P': 6,
    'Q': 7,
    'R': 8,
    'p_N': 9,
    'p_E': 10,
    'alt': 11,
    'poewr': 12
}

u_id = {
    'thtl': 0,
    'elv_deg': 1,
    'ail_deg': 2,
    'rdr_deg': 3
}

ss = linearize(x0, u0, p0)
G = ss[y_id['P'], u_id['ail_deg']];

#%%
# open loop root locus
control.rlocus(G, kvect=np.linspace(0, 1e5, 1000));

#%%
# open loop bode plot
control.bode(G, omega=np.logspace(-2, 2), dB=True, Hz=True);


#%%
# Design a simple P controller
H = control.ss(0, 0, 0, -1e5)

#%%
# Plot the compensated openloop root locus
plt.figure()
control.rlocus(G*H, kvect=np.linspace(0, 1));

#%%
# Plot the compensated openloop bode plot
plt.figure()
control.bode(G*H, omega=np.logspace(-2, 2), dB=True, Hz=True);

#%%
# Plot the closed loop bode
plt.figure()
Gc = control.feedback(G*H, 1)
control.bode(Gc, omega=np.logspace(-2, 2), dB=True, Hz=True);

#%%
# Plot the closed loop step response
plt.figure()
t, y = control.step_response(Gc, T=np.linspace(0, 30, 1000))
plt.plot(t, y)
plt.hlines(1, 0, 30)

#%%
