import matplotlib.pyplot as plt
import numpy as np
import pathlib
import f16
import casadi as ca
import pytest
from casadi.tools.graph import graph
import os


TRIM_TOL = 1e-5


def create_graph(expr, path):
    pdot = graph.dotgraph(expr)
    os.remove('source.dot')
    pdot.write_png(path)


def plot_table2D(title, path, x_grid, y_grid, x_label, y_label, f_table):
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros((len(x_grid), len(y_grid)))
    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            Z[i, j] = f_table(x, y)
    plt.figure()
    plt.contourf(X, Y, Z.T, levels=20)
    plt.colorbar()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(path.joinpath('{:s}.png'.format(title)))
    plt.close()


def test_tables():
    alpha_deg_grid = np.linspace(-15, 50, 20)
    beta_deg_grid = np.linspace(-35, 35, 20)
    elev_deg_grid = np.linspace(-30, 30, 20)
    ail_deg_grid = np.linspace(-30, 30, 20)
    mach_grid = np.linspace(0, 1.1, 20)
    alt_grid = np.linspace(-1e4, 6e4, 20)

    path = pathlib.Path('results')
    path.mkdir(parents=True, exist_ok=True)

    tables = f16.tables
    plot_table2D('Cl', path, alpha_deg_grid, beta_deg_grid, 'alpha_deg', 'beta_deg', tables['Cl'])
    plot_table2D('Cm', path, alpha_deg_grid, elev_deg_grid, 'alpha_deg', 'elev_deg', tables['Cm'])
    plot_table2D('Cn', path, alpha_deg_grid, beta_deg_grid, 'alpha_deg', 'beta_deg', tables['Cn'])
    plot_table2D('Cx', path, alpha_deg_grid, elev_deg_grid, 'alpha_deg', 'elev_deg', tables['Cx'])
    plot_table2D('Cy', path, beta_deg_grid, ail_deg_grid, 'beta_deg', 'ail_deg',
                 lambda x, y: tables['Cy'](x, y, 0))
    plot_table2D('Cz', path, alpha_deg_grid, beta_deg_grid, 'alpha_deg', 'beta_deg',
                 lambda x, y: tables['Cz'](x, y, 0))
    plot_table2D('thrust_idle', path, alt_grid, mach_grid, 'alt, ft', 'mach', tables['thrust_idle'])
    plot_table2D('thrust_mil', path, alt_grid, mach_grid, 'alt, ft', 'mach', tables['thrust_mil'])
    plot_table2D('thrust_max', path, alt_grid, mach_grid, 'alt, ft', 'mach', tables['thrust_max'])

    plt.figure()
    lift = []
    for alpha in alpha_deg_grid:
        lift.append(-tables['Cz'](alpha, 0, 0))
    plt.plot(alpha_deg_grid, lift)
    plt.xlabel('alpha, deg')
    plt.ylabel('CL')
    plt.savefig(path.joinpath('CL.png'))
    plt.close()

    plt.figure()
    plot_table2D('amach', path, np.linspace(0, 1000), np.linspace(0, 60000), 'VT, ft/s', 'alt, ft', tables['amach'])
    plt.close()

    names = ['CXq', 'CYr', 'CYp', 'CZq', 'Clr', 'Clp', 'Cmq', 'Cnr', 'Cnp']
    for name in names:
        plt.figure()
        data = [tables[name](alpha) for alpha in alpha_deg_grid]
        plt.plot(alpha_deg_grid, data)
        plt.xlabel('alpha, deg')
        plt.ylabel(name)
        plt.savefig(path.joinpath('damp_{:s}.png'.format(name)))
        plt.close()


def test_jacobian():
    x_sym = ca.MX.sym('x', 16)
    u_sym = ca.MX.sym('u', 4)
    x = f16.State.from_casadi(x_sym)
    u = f16.Control.from_casadi(u_sym)
    p = f16.Parameters()
    dx = f16.dynamics(x, u, p)
    A = ca.jacobian(dx.to_casadi(), x_sym)
    B = ca.jacobian(dx.to_casadi(), u_sym)
    f_A = ca.Function('A', [x_sym, u_sym], [A])
    f_B = ca.Function('B', [x_sym, u_sym], [B])
    print('A', f_A(np.ones(16), np.ones(4)))
    print('B', f_B(np.ones(16), np.ones(4)))


def test_trim1():
    # pg 197
    p = f16.Parameters()
    x = f16.State(VT=502, alpha=0.03691, theta=0.03691)
    u = f16.Control(thtl=0.1385, elv_cmd_deg=-0.7588)
    x = f16.trim_actuators(x, u)
    dx = f16.dynamics(x, u, p)
    print(dx)
    assert f16.trim_cost(dx) < TRIM_TOL


def test_trim2():
    # pg 197
    p = f16.Parameters(xcg=0.3)
    x = f16.State(VT=502, alpha=0.03936, theta=0.03936)
    u = f16.Control(thtl=0.1485, elv_cmd_deg=-1.931)
    x = f16.trim_actuators(x, u)
    x.power = f16.tables['tgear'](u.thtl)
    dx = f16.dynamics(x, u, p)
    print(dx)
    assert f16.trim_cost(dx) < TRIM_TOL


def test_trim3():
    # pg 197
    p = f16.Parameters(xcg=0.38)
    x = f16.State(VT=502, alpha=0.03544, theta=0.03544)
    u = f16.Control(thtl=0.1325, elv_cmd_deg=-0.0559)
    x = f16.trim_actuators(x, u)
    dx = f16.dynamics(x, u, p)
    assert f16.trim_cost(dx) < TRIM_TOL


def test_trim4():
    # pg 197
    p = f16.Parameters(xcg=0.3)
    # psi_dot = 0.3
    x = f16.State(VT=502, alpha=0.2485, beta=4.8e-4, phi=1.367, theta=0.05185,
                  P=-0.0155, Q=0.2934, R=0.06071)
    u = f16.Control(
        thtl=0.8499, elv_cmd_deg=-6.256,
        ail_cmd_deg=0.09891, rdr_cmd_deg=-0.4218)
    x = f16.trim_actuators(x, u)
    dx = f16.dynamics(x, u, p)
    print(dx)
    assert f16.trim_cost(dx) < TRIM_TOL


def test_trim5():
    # pg 197
    p = f16.Parameters(xcg=0.3)  # listed as -0.3, must be typo
    # theta_dot = 0.3
    x = f16.State(VT=502, alpha=0.3006, beta=4.1e-5, theta=0.3006, Q=0.3)
    u = f16.Control(
        thtl=1.023, elv_cmd_deg=-7.082,
        ail_cmd_deg=-6.2e-4, rdr_cmd_deg=0.01655)
    x = f16.trim_actuators(x, u)
    dx = f16.dynamics(x, u, p)
    print(dx)
    assert f16.trim_cost(dx) < 2e-2  # doesn't converge as close


def test_trim6():
    # pg 195
    p = f16.Parameters()
    x = f16.State(VT=502, alpha=2.392628e-1, beta=5.061803e-4,
                  phi=1.366289, theta=5.000808e-2, psi=2.340769e-1,
                  P=-1.499617e-2, Q=2.933811e-1, R=6.084932e-2,
                  p_N=0, p_E=0, alt=0, power=6.412363e1)
    u = f16.Control(thtl=8.349601e-1, elv_cmd_deg=-1.481766,
                    ail_cmd_deg=9.553108e-2, rdr_cmd_deg=-4.118124e-1)
    x = f16.trim_actuators(x, u)
    dx = f16.dynamics(x, u, p)
    print(dx)
    assert f16.trim_cost(dx) < TRIM_TOL

def test_trim_computation():
    # pg 195
    p = f16.Parameters()
    x = f16.State(VT=502)
    x0, u0 = f16.trim(x=x, p=p, phi_dot=0, theta_dot=0, psi_dot=0, gam=0)
    dx = f16.dynamics(x0, u0, p)
    print(dx)
    assert f16.trim_cost(dx) < TRIM_TOL


def test_table_3_5_2():
    # pg 187
    p = f16.Parameters(xcg=0.4)
    x = f16.State(
        VT=500, alpha=0.5, beta=-0.2,
        phi=-1, theta=1, psi=-1,
        P=0.7, Q=-0.8, R=0.9,
        p_N=1000, p_E=900, alt=10000)
    u = f16.Control(
        thtl=0.9, elv_cmd_deg=20,
        ail_cmd_deg=-15, rdr_cmd_deg=-20)
    x = f16.trim_actuators(x, u)
    x.power = 90
    dx = f16.dynamics(x, u, p)
    dx_compute = np.array(dx.to_casadi())[:, 0]
    dx_check = np.array([
        -75.23724, -0.8813491, -0.4759990,
        2.505734, 0.3250820, 2.145926,
        12.62679, 0.9649671, 0.5809759,
        342.4439, -266.7707, 248.1241, -58.68999, 0, 0, 0
    ])
    print('\nexpected:\n\t', dx_check)
    print('\nactual:\n\t', dx_compute)
    print('\nerror:\n\t', dx_check - dx_compute)
    assert np.allclose(dx_compute, dx_check, 1e-3)
