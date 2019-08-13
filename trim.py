# pylint: disable=invalid-name, too-many-locals, missing-docstring, too-many-arguments, redefined-outer-name
import matplotlib.pyplot as plt
import casadi as ca
import numpy as np
import f16


def trim(s0, x: f16.State, p: f16.Parameters, tables, phi_dot, theta_dot, psi_dot, gam):

    def constrain(x, s):
        u = f16.Control(thtl=s[0], elv_deg=s[1], ail_deg=s[2], rdr_deg=s[3])
        alpha = s[4]
        beta = s[5]

        x.alpha = alpha
        x.beta = beta

        cos = ca.cos
        sin = ca.sin
        tan = ca.tan
        atan = ca.arctan
        sqrt = ca.sqrt

        VT = x.VT
        g = p.g
        G = psi_dot*VT/g

        a = 1 - G*tan(alpha)*sin(beta)
        b = sin(gam)/cos(beta)
        c = 1 + G**2*cos(beta)**2

        # coordinated turn constraint pg. 188
        phi = atan(G*cos(beta)/cos(alpha) *
                   ((a - b**2) + b*tan(alpha)*sqrt(c*(1 - b**2) + G**2*sin(beta)**2))
                   / (a**2 - b**2*(1 + c*tan(alpha)**2)))
        x.phi = phi

        # rate of climb constraint pg. 187
        a = cos(alpha)*cos(beta)
        b = sin(phi)*sin(beta) + cos(phi)*sin(alpha)*cos(beta)
        theta = (a*b + sin(gam)*sqrt(a**2 - sin(gam)**2 + b**2)) \
            / (a**2 - sin(gam)**2)
        x.theta = theta

        # kinematics pg. 20
        x.P = phi_dot - sin(theta)*psi_dot
        x.Q = cos(phi)*phi_dot + sin(phi)*cos(theta)*psi_dot
        x.R = -sin(phi)*theta_dot + cos(phi)*cos(theta)*psi_dot

        # engine power constraint
        x.power = tables['tgear'](u.thtl)
        return x, u

    s = ca.MX.sym('s', 6)
    x, u = constrain(x, s)
    f = f16.trim_cost(f16.dynamics(x, u, p, tables))
    nlp = {'x': s, 'f': f}
    S = ca.nlpsol('S', 'ipopt', nlp, {
        'ipopt': {
            'print_level': 0,
        }
    })
    r = S(x0=s0, lbg=0, ubg=0)
    s_opt = r['x']
    x, u = constrain(x, s_opt)
    return x, u


def simulate(x0, u0, t0, tf, dt):
    xs = ca.MX.sym('x', 13)
    x = f16.State.from_casadi(xs)
    us = ca.MX.sym('u', 4)
    u = f16.Control.from_casadi(us)
    dae = {'x': xs, 'p': us, 'ode': f16.dynamics(x, u, p, tables).to_casadi()}
    F = ca.integrator('F', 'idas', dae, {'t0': 0, 'tf': dt})
    x = np.array(x0.to_casadi()).reshape(-1)
    u = np.array(u0.to_casadi()).reshape(-1)
    data = {
        't': [0],
        'x': [x]
    }
    t_vect = np.arange(t0, tf, dt)
    dt = 0.01
    for t in t_vect:
        x = np.array(F(x0=x, p=u)['xf']).reshape(-1)
        data['t'].append(t)
        data['x'].append(x)
    for k in data.keys():
        data[k] = np.array(data[k])
    return data


# %%
p = f16.Parameters()
tables = f16.build_tables()
x0, u0 = trim(
    s0=[0, 0, 0, 0, 0, 0],
    x=f16.State(VT=500, alt=5000),
    p=p,
    tables=tables,
    phi_dot=0, theta_dot=0, psi_dot=0.3, gam=0)


data = simulate(x0, u0, 0, 50, 0.1)
state_index = f16.State().name_to_index

plt.figure()
plt.plot(data['x'][:, state_index('p_E')], data['x'][:, state_index('p_N')])
plt.xlabel('E, ft')
plt.ylabel('N, ft')
plt.show()

# plt.figure()
#plt.plot(data['t'], data['x'][:, state_index('p_N')])

# plt.figure()
#plt.plot(data['t'], data['x'][:, state_index('p_E')])


# %%
