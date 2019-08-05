import matplotlib.pyplot as plt
import numpy as np
import pathlib
import f16
import casadi as ca
import pytest


def plot_table2D(title, path, x_grid, y_grid, x_label, y_label, f_table):
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros((len(x_grid), len(y_grid)))
    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            Z[i, j] = f_table(x, y)
    plt.figure()
    plt.contourf(X, Y, Z.T)
    plt.colorbar()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(path.joinpath('{:s}.png'.format(title)))
    plt.close()


def test_tables():
    alpha_deg_grid = np.linspace(-10, 45, 20)
    beta_deg_grid = np.linspace(-30, 30, 20)
    elev_deg_grid = np.linspace(-24, 24, 20)
    ail_deg_grid = np.linspace(-24, 24, 20)

    path = pathlib.Path('results')
    path.mkdir(parents=True, exist_ok=True)

    tables = f16.build_tables()
    plot_table2D('Cl', path, alpha_deg_grid, beta_deg_grid, 'alpha_deg', 'beta_deg', tables['Cl'])
    plot_table2D('Cm', path, alpha_deg_grid, elev_deg_grid, 'alpha_deg', 'elev_deg', tables['Cm'])
    plot_table2D('Cn', path, alpha_deg_grid, beta_deg_grid, 'alpha_deg', 'beta_deg', tables['Cn'])
    plot_table2D('Cx', path, alpha_deg_grid, elev_deg_grid, 'alpha_deg', 'elev_deg', tables['Cx'])
    plot_table2D('Cy', path, beta_deg_grid, ail_deg_grid, 'beta_deg', 'ail_deg',
                 lambda x, y: tables['Cy'](x, y, 0))
    plot_table2D('Cz', path, alpha_deg_grid, beta_deg_grid, 'alpha_deg', 'beta_deg',
                 lambda x, y: tables['Cz'](x, y, 0))

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


def trim_cost(dx: f16.StateDot):
    return dx.VT_dot**2 + \
        100*(dx.alpha_dot**2 + dx.beta_dot**2) + \
        10*(dx.P_dot**2 + dx.Q_dot**2 + dx.R_dot**2)


def test_jacobian():
    x_sym = ca.MX.sym('x', 13)
    u_sym = ca.MX.sym('u', 4)
    x = f16.State.from_casadi(x_sym)
    u = f16.Control.from_casadi(u_sym)
    p = f16.Parameters()
    tables = f16.build_tables()
    dx = f16.dynamics(x, u, p, tables)
    A = ca.jacobian(dx.to_casadi(), x_sym)
    B = ca.jacobian(dx.to_casadi(), u_sym)
    f_A = ca.Function('A', [x_sym, u_sym], [A])
    f_B = ca.Function('B', [x_sym, u_sym], [B])
    print('A', f_A(np.ones(13), np.ones(4)))
    print('B', f_B(np.ones(13), np.ones(4)))


def test_trim1():
    p = f16.Parameters()
    tables = f16.build_tables()
    x = f16.State(VT=502, alpha=0.03691, beta=-4e-9, theta=0.03691)
    u = f16.Control(thtl=0.1385, elv_deg=-0.7588, ail_deg=-1.2e-7, rdr_deg=6.2e-7)
    x.power = tables['tgear'](u.thtl)
    dx = f16.dynamics(x, u, p, tables)
    assert trim_cost(dx) < 1e-2


def test_trim2():
    p = f16.Parameters(xcg=0.3)
    x = f16.State(VT=502, alpha=0.03936, beta=4.1e-9, theta=0.03936)
    u = f16.Control(thtl=0.1485, elv_deg=-1.931, ail_deg=-7.0e-8, rdr_deg=8.3e-7)
    tables = f16.build_tables()
    x.power = tables['tgear'](u.thtl)
    dx = f16.dynamics(x, u, p, tables)
    assert trim_cost(dx) < 1e-1


def test_trim3():
    p = f16.Parameters(xcg=0.38)
    x = f16.State(VT=502, alpha=0.03544, beta=3.1e-8, theta=0.03544)
    u = f16.Control(thtl=0.1325, elv_deg=-0.0559, ail_deg=-5.1e-7, rdr_deg=4.3e-6)
    tables = f16.build_tables()
    x.power = tables['tgear'](u.thtl)
    dx = f16.dynamics(x, u, p, tables)
    assert trim_cost(dx) < 1e-4


def test_trim4():
    p = f16.Parameters(xcg=0.3)
    psi_dot = 0.3
    x = f16.State(VT=502, alpha=0.2485, beta=4.8e-4, phi=1.367, theta=0.05185,
                  P=-0.0155, Q=0.2934, R=0.06071)
    u = f16.Control(thtl=0.8499, elv_deg=-6.256, ail_deg=0.09891, rdr_deg=-0.4218)
    tables = f16.build_tables()
    x.power = tables['tgear'](u.thtl)
    dx = f16.dynamics(x, u, p, tables)
    assert trim_cost(dx) < 3000


def test_trim5():
    p = f16.Parameters(xcg=-0.3)
    theta_dot = 0.3
    x = f16.State(VT=502, alpha=0.3006, beta=4.1e-5, theta=0.3006, Q=0.3)
    u = f16.Control(thtl=1.023, elv_deg=-7.082, ail_deg=-6.2e-4, rdr_deg=0.01655)
    tables = f16.build_tables()
    x.power = tables['tgear'](u.thtl)
    dx = f16.dynamics(x, u, p, tables)
    assert trim_cost(dx) < 2000
