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
    plt.contourf(X, Y, Z.T, levels=20)
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

    assert tables['Cx'](-10, -24) == -0.099
    assert tables['Cx'](45, 24) == 0.040
    assert tables['Cm'](-10, -24) == 0.205
    assert tables['Cl'](-10, 5) == -0.001
    assert tables['Cn'](-10, 5) == 0.018
    assert tables['DlDa'](-10, -30) == -0.041
    assert tables['DlDr'](-10, -30) == 0.005
    assert tables['DnDa'](-10, -30) == 0.001
    assert tables['DnDr'](-10, -30) == -0.018
    assert tables['DnDr'](-10, -30) == -0.018
    assert tables['CXq'](-10) == -0.267
    assert tables['CYr'](-10) == 0.882
    assert tables['CYp'](-10) == -0.108
    assert tables['CZq'](-10) == -8.80
    assert tables['Clr'](-10) == -0.126
    assert tables['Clp'](-10) == -0.360
    assert tables['Cmq'](-10) == -7.21
    assert tables['Cnr'](-10) == -0.380
    assert tables['Cnp'](45) == 0.150

    names = ['CXq', 'CYr', 'CYp', 'CZq', 'Clr', 'Clp', 'Cmq', 'Cnr', 'Cnp']
    for name in names:
        plt.figure()
        data = [tables[name](alpha) for alpha in alpha_deg_grid]
        plt.plot(alpha_deg_grid, data)
        plt.xlabel('alpha, deg')
        plt.ylabel(name)
        plt.savefig(path.joinpath('damp_{:s}.png'.format(name)))
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
    # pg 197
    p = f16.Parameters()
    tables = f16.build_tables()
    x = f16.State(VT=502, alpha=0.03691, theta=0.03691)
    u = f16.Control(thtl=0.1385, elv_deg=-0.7588)
    x.power = tables['tgear'](u.thtl)
    dx = f16.dynamics(x, u, p, tables)
    print(dx)
    assert trim_cost(dx) < 1e-3


def test_trim2():
    # pg 197
    p = f16.Parameters(xcg=0.3)
    x = f16.State(VT=502, alpha=0.03936, theta=0.03936)
    u = f16.Control(thtl=0.1485, elv_deg=-1.931)
    tables = f16.build_tables()
    x.power = tables['tgear'](u.thtl)
    dx = f16.dynamics(x, u, p, tables)
    print(dx)
    assert trim_cost(dx) < 1e-3


def test_trim3():
    # pg 197
    p = f16.Parameters(xcg=0.38)
    x = f16.State(VT=502, alpha=0.03544, theta=0.03544)
    u = f16.Control(thtl=0.1325, elv_deg=-0.0559)
    tables = f16.build_tables()
    x.power = tables['tgear'](u.thtl)
    dx = f16.dynamics(x, u, p, tables)
    print(dx)
    assert trim_cost(dx) < 1e-3


def test_trim4():
    # pg 197
    p = f16.Parameters(xcg=0.3)
    psi_dot = 0.3
    x = f16.State(VT=502, alpha=0.2485, phi=1.367, theta=0.05185,
                  P=-0.0155, Q=0.2934, R=0.06071)
    u = f16.Control(thtl=0.8499, elv_deg=-6.256, ail_deg=0.09891, rdr_deg=-0.4218)
    tables = f16.build_tables()
    x.power = tables['tgear'](u.thtl)
    dx = f16.dynamics(x, u, p, tables)
    print(dx)
    assert trim_cost(dx) < 1e-3


def test_trim5():
    # pg 197
    p = f16.Parameters(xcg=-0.3)
    theta_dot = 0.3
    x = f16.State(VT=502, alpha=0.3006, theta=0.3006, Q=0.3)
    u = f16.Control(thtl=1.023, elv_deg=-7.082, rdr_deg=0.01655)
    tables = f16.build_tables()
    x.power = tables['tgear'](u.thtl)
    dx = f16.dynamics(x, u, p, tables)
    print(dx)
    assert trim_cost(dx) < 1e-3


def test_trim6():
    # pg 195
    p = f16.Parameters()
    x = f16.State(VT=502, alpha=2.392628e-1, beta=5.061803e-4,
                  phi=1.366289, theta=5.000808e-2, psi=2.340769e-1,
                  P=-1.499617e-2, Q=2.933811e-1, R=6.084932e-2,
                  p_N=0, p_E=0, alt=0, power=6.412363e1)
    u = f16.Control(thtl=8.349601e-1, ail_deg=-1.481766,
                    elv_deg=9.553108e-2, rdr_deg=-4.118124e-1)
    tables = f16.build_tables()
    dx = f16.dynamics(x, u, p, tables)
    print(dx)
    assert trim_cost(dx) < 1e-3


def test_table_3_5_2():
    #pg 187
    p = f16.Parameters(xcg=0.4)
    x = f16.State(
        VT=500, alpha=0.5, beta=-0.2,
        phi=-1, theta=1, psi=-1,
        P=0.7, Q=-0.8, R=0.9,
        p_N=1000, p_E=900, alt=10000, power=90)
    u = f16.Control(thtl=0.9, ail_deg=20, elv_deg=-15, rdr_deg=-20)
    tables = f16.build_tables()
    dx = f16.dynamics(x, u, p, tables)
    dx_compute = np.array(dx.to_casadi())[:, 0]
    dx_check = np.array([
        -75.23724, -0.8813491, -0.4759990,
        2.505734, 0.3250820, 2.145926,
        12.62679, 0.9649671, 0.5809759,
        342.4439, -266.7707, 248.1241, -58.68999
    ])
    print('\nexpected:\n\t', np.round(dx_check))
    print('\nactual:\n\t', np.round(dx_compute))
    print('\nerror:\n\t', np.round(dx_check - dx_compute))
    assert np.allclose(dx_compute, dx_check, 1e-3)
