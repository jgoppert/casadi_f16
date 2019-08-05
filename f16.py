import casadi as ca
import numpy as np
from casadi.tools.graph import graph
from dataclasses import dataclass, asdict, astuple


INTERP_DEFAULT = 'linear'


def build_tables():
    tables = {}

    def create_table2D(name, row_label, col_label, data, abs_row=False, abs_col=False, interp_method=INTERP_DEFAULT):
        """
        Creates a table interpolation function with x as rows and y as columns
        """
        assert data[0, 0] == 0
        row_grid = data[1:, 0]
        col_grid = data[0, 1:]

        interp = ca.interpolant(name + '_interp', interp_method, [row_grid, col_grid],
                                data[1:, 1:].ravel(order='F'))
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        if abs_row:
            xs = ca.fabs(x)
        else:
            xs = x
        if abs_col:
            ys = ca.fabs(y)
        else:
            ys = y
        return ca.Function('Cx', [x, y], [interp(ca.vertcat(xs, ys))], [row_label, col_label], [name])

    def create_damping():
        alpha_deg = ca.MX.sym('alpha_deg')
        data = np.array([
            [-10,    -5,      0,      5,      10,     15,     20,     25,     30,     35,     40,     45],  # alpha, deg
            [-0.267, -0.110,  0.308,  1.34,   2.08,   2.91,   2.76,   2.05,   1.50,   1.49,   1.83,   1.21],  # CXq
            [0.882,   0.852,  0.876,  0.958,  0.962,  0.974,  0.819,  0.483,  0.590,  1.21,  -0.493, -1.04],  # CYr
            [-0.108, -0.108, -0.188,  0.110,  0.258,  0.226,  0.344,  0.362,  0.611,  0.529,  0.298, -2.27],  # CYp
            [-8.80,  -25.8,  -28.9,  -31.4,  -31.2,  -30.7,  -27.7,  -28.2,  -29.0,  -29.8,  -38.3,  -35.3],  # CZq
            [-0.126, -0.26,   0.063,  0.113,  0.208,  0.230,  0.319,  0.437,  0.680,  0.100,  0.447, -0.330],  # Clr
            [-0.360, -0.359, -0.443, -0.420, -0.383, -0.375, -0.329, -0.294, -0.230, -0.210, -0.120, -0.100],  # Clp
            [-7.21,  -0.540, -5.23,  -5.26,  -6.11,  -6.64,  -5.69,  -6.00,  -6.20,  -6.40,  -6.60,  -6.00],  # Cmq
            [-0.380, -0.363, -0.378, -0.386, -0.370, -0.453, -0.550, -0.582, -0.595, -0.637, -1.02,  -0.840],  # Cnr
            [0.061,   0.052,  0.052, -0.012, -0.013, -0.024,  0.050,  0.150,  0.130,  0.158,  0.240,  0.150]   # Cnp
        ])
        names = ['CXq', 'CYr', 'CYp', 'CZq', 'Clr', 'Clp', 'Cmq', 'Cnr', 'Cnp']
        for i, name in enumerate(names):
            tables[name] = ca.interpolant('{:s}_interp'.format(name), INTERP_DEFAULT, [data[0, :]], data[i, :].ravel(order='F'))
    create_damping()

    tables['Cx'] = create_table2D(
        name='Cx', row_label='alpha_deg', col_label='elev_deg',
        data=np.array([                           # alpha, deg
            [0,   -10,    -5,      0,      5,      10,     15,     20,     25,     30,     35,      40,     45],
            [-24, -0.099, -0.081, -0.081, -0.063, -0.025,  0.044,  0.097,  0.113,  0.145,  0.167,   0.174,  0.166],
            [-12, -0.048, -0.038, -0.040, -0.021,  0.016,  0.083,  0.127,  0.137,  0.162,  0.177,   0.179,  0.167],  # elev, deg
            [0,   -0.022, -0.020, -0.021, -0.004,  0.032,  0.094,  0.128,  0.130,  0.154,  0.161,   0.155,  0.138],
            [12,  -0.040, -0.038, -0.039, -0.025,  0.006,  0.062,  0.087,  0.085,  0.100,  0.110,   0.104,  0.091],
            [24,  -0.083, -0.073, -0.076, -0.072, -0.046,  0.012,  0.024,  0.025,  0.043,  0.053,   0.047,  0.040]
        ]).T)

    def create_Cy():
        beta_deg = ca.MX.sym('beta_deg')
        ail_deg = ca.MX.sym('ail_deg')
        rdr_deg = ca.MX.sym('rdr_deg')
        tables['Cy'] = ca.Function('Cy', [beta_deg, ail_deg, rdr_deg], [-0.02*beta_deg + 0.021*ail_deg/20 + 0.086*rdr_deg/30],
                                   ['beta_deg', 'ail_deg', 'rdr_deg'], ['Cy'])
    create_Cy()

    def create_Cz():
        alpha_deg = ca.MX.sym('alpha_deg')
        beta_deg = ca.MX.sym('beta_deg')
        elev_deg = ca.MX.sym('elev_deg')
        data = np.array([
            [-10,   -5,      0,      5,     10,      15,     20,     25,     30,     35,     40,     45],
            [0.770,  0.241, -0.100, -0.416, -0.731, -1.053, -1.366, -1.646, -1.917, -2.120, -2.248, -2.229]
        ])
        interp = ca.interpolant('Cz_interp', INTERP_DEFAULT, [data[0, :]],
                                data[1, :].ravel(order='F'))
        return ca.Function('Cz',
                           [alpha_deg, beta_deg, elev_deg],
                           [interp(alpha_deg)*(1 - (beta_deg/57.3)**2 - 0.19*elev_deg/25.0)],
                           ['alpha_deg', 'beta_deg', 'elev_deg'], ['Cz'])

    tables['Cz'] = create_Cz()

    tables['Cl'] = create_table2D(
        name='Cl', row_label='alpha_deg', col_label='beta_deg',
        data=np.array([                           # alpha, deg
            [0,  -10,    -5,      0,      5,      10,     15,     20,     25,     30,     35,      40,     45],
            [0,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0,       0,      0],
            [5,  -0.001, -0.004, -0.008, -0.012, -0.016, -0.019, -0.020, -0.020, -0.015, -0.008,  -0.013, -0.015],
            [10, -0.003, -0.009, -0.017, -0.024, -0.030, -0.034, -0.040, -0.037, -0.016, -0.002,  -0.010, -0.019],  # beta, deg
            [15, -0.001, -0.010, -0.020, -0.030, -0.039, -0.044, -0.050, -0.049, -0.023, -0.006,  -0.014, -0.027],
            [20,  0.000, -0.010, -0.022, -0.034, -0.047, -0.046, -0.059, -0.061, -0.033, -0.036,  -0.035, -0.035],
            [25,  0.007, -0.010, -0.023, -0.034, -0.049, -0.046, -0.068, -0.071, -0.060, -0.058,  -0.062, -0.059],
            [30,  0.009, -0.011, -0.023, -0.037, -0.050, -0.047, -0.074, -0.079, -0.091, -0.076,  -0.077, -0.076]
        ]).T, abs_col=True)

    tables['Cm'] = create_table2D(
        name='Cm', row_label='alpha_deg', col_label='elev_deg',
        data=np.array([                           # alpha, deg
            [0,   -10,    -5,      0,      5,      10,     15,     20,     25,     30,     35,      40,     45],
            [-24,  0.205,  0.168,  0.186,  0.196,  0.213,  0.251,  0.245,  0.238,  0.252,  0.231,   0.198,  0.192],
            [-12,  0.081,  0.077,  0.107,  0.110,  0.110,  0.141,  0.127,  0.119,  0.133,  0.108,   0.081,  0.093],  # elev, deg
            [0,   -0.046, -0.020, -0.009, -0.005, -0.006,  0.010,  0.006, -0.001,  0.014,  0.000,  -0.013,  0.032],
            [12,  -0.174, -0.145, -0.121, -0.127, -0.129, -0.102, -0.097, -0.113, -0.087, -0.084,  -0.069, -0.006],
            [24,  -0.259, -0.202, -0.184, -0.193, -0.199, -0.150, -0.160, -0.167, -0.104, -0.076,  -0.041, -0.005]
        ]).T)

    tables['Cn'] = create_table2D(
        name='Cn', row_label='alpha_deg', col_label='beta_deg',
        data=np.array([                           # alpha, deg
            [0,  -10,    -5,      0,      5,      10,     15,     20,     25,     30,     35,      40,     45],
            [0,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0,       0,      0],
            [5,   0.018,  0.019,  0.018,  0.019,  0.019,  0.018,  0.013,  0.007,  0.004, -0.014,  -0.017, -0.033],
            [10,  0.038,  0.042,  0.042,  0.042,  0.043,  0.039,  0.030,  0.017,  0.004, -0.035,  -0.047, -0.057],  # beta, deg
            [15,  0.056,  0.057,  0.059,  0.058,  0.058,  0.053,  0.032,  0.012,  0.002, -0.046,  -0.071, -0.073],
            [20,  0.064,  0.077,  0.076,  0.074,  0.073,  0.057,  0.029,  0.007,  0.012, -0.034,  -0.065, -0.041],
            [25,  0.074,  0.086,  0.093,  0.089,  0.080,  0.062,  0.049,  0.022,  0.028, -0.012,  -0.002, -0.013],
            [30,  0.079,  0.090,  0.106,  0.106,  0.096,  0.080,  0.068,  0.030,  0.064,  0.015,   0.011, -0.001]
        ]).T, abs_col=True)

    tables['DlDa'] = create_table2D(
        name='DlDa', row_label='alpha_deg', col_label='beta_deg',
        data=np.array([                           # alpha, deg
            [0,   -10,    -5,      0,      5,      10,     15,     20,     25,     30,     35,     40,     45],
            [-30, -0.041, -0.052, -0.053, -0.056, -0.050, -0.056, -0.082, -0.059, -0.042, -0.038, -0.027, -0.017],
            [-20, -0.041, -0.053, -0.053, -0.053, -0.050, -0.051, -0.066, -0.043, -0.038, -0.027, -0.023, -0.016],
            [-10, -0.042, -0.053, -0.052, -0.051, -0.049, -0.049, -0.043, -0.035, -0.026, -0.016, -0.018, -0.014],  # beta, deg
            [0,   -0.040, -0.052, -0.051, -0.052, -0.048, -0.048, -0.042, -0.037, -0.031, -0.026, -0.017, -0.012],
            [10,  -0.043, -0.049, -0.048, -0.049, -0.043, -0.042, -0.042, -0.036, -0.025, -0.021, -0.016, -0.011],
            [20,  -0.044, -0.048, -0.048, -0.047, -0.042, -0.041, -0.020, -0.028, -0.013, -0.014, -0.011, -0.010],
            [30,  -0.043, -0.049, -0.047, -0.045, -0.042, -0.037, -0.003, -0.013, -0.010, -0.003, -0.007, -0.008]
        ]).T)

    tables['DlDr'] = create_table2D(
        name='DlDr', row_label='alpha_deg', col_label='beta_deg',
        data=np.array([                           # alpha, deg
            [0,   -10,    -5,      0,      5,      10,     15,     20,     25,     30,     35,     40,     45],
            [-30,  0.005,  0.017,  0.014,  0.010, -0.005,  0.009,  0.019,  0.005, -0.000, -0.005, -0.011,  0.008],
            [-20,  0.007,  0.016,  0.014,  0.014,  0.013,  0.009,  0.012,  0.005,  0.000,  0.004,  0.009,  0.007],
            [-10,  0.013,  0.013,  0.011,  0.012,  0.011,  0.009,  0.008,  0.005, -0.002,  0.005,  0.003,  0.005],  # beta, deg
            [0,    0.018,  0.015,  0.015,  0.014,  0.014,  0.014,  0.014,  0.015,  0.013,  0.011,  0.006,  0.001],
            [10,   0.015,  0.014,  0.013,  0.013,  0.012,  0.011,  0.011,  0.010,  0.008,  0.008,  0.007,  0.003],
            [20,   0.021,  0.011,  0.010,  0.011,  0.010,  0.009,  0.008,  0.010,  0.006,  0.005,  0.000,  0.001],
            [30,   0.023,  0.010,  0.011,  0.011,  0.011,  0.010,  0.008,  0.010,  0.006,  0.014,  0.020,  0.000]
        ]).T)

    tables['DnDa'] = create_table2D(
        name='DnDa', row_label='alpha_deg', col_label='beta_deg',
        data=np.array([                           # alpha, deg
            [0,   -10,    -5,      0,      5,      10,     15,     20,     25,     30,     35,    40,      45],
            [-30,  0.001, -0.027, -0.017, -0.013, -0.012, -0.016,  0.001,  0.017,  0.011,  0.017,  0.008,  0.016],
            [-20,  0.002, -0.014, -0.016, -0.016, -0.014, -0.019, -0.021,  0.002,  0.012,  0.015,  0.015,  0.011],
            [-10, -0.006, -0.008, -0.006, -0.006, -0.005, -0.008, -0.005,  0.007,  0.004,  0.007,  0.006,  0.006],  # beta, deg
            [0,   -0.011, -0.011, -0.010, -0.009, -0.008, -0.006,  0.000,  0.004,  0.007,  0.010,  0.004,  0.010],
            [10,  -0.015, -0.015, -0.014, -0.012, -0.011, -0.008, -0.002,  0.002,  0.006,  0.012,  0.011,  0.011],
            [20,  -0.024, -0.010, -0.004, -0.002, -0.001,  0.003,  0.014,  0.006, -0.001,  0.004,  0.004,  0.006],
            [30,  -0.022,  0.002, -0.003, -0.005, -0.003, -0.001, -0.009, -0.009, -0.001,  0.003, -0.002,  0.001]
        ]).T)

    tables['DnDr'] = create_table2D(
        name='DnDr', row_label='alpha_deg', col_label='beta_deg',
        data=np.array([                           # alpha, deg
            [0,    -10,    -5,      0,      5,      10,     15,     20,     25,     30,     35,     40,     45],
            [-30,  -0.018, -0.052, -0.052, -0.052, -0.054, -0.049, -0.059, -0.051, -0.030, -0.037, -0.026, -0.013],
            [-20,  -0.028, -0.051, -0.043, -0.046, -0.045, -0.049, -0.057, -0.052, -0.030, -0.033, -0.030, -0.008],
            [-10,  -0.037, -0.041, -0.038, -0.040, -0.040, -0.038, -0.037, -0.030, -0.027, -0.024, -0.019, -0.013],  # beta, deg
            [0,    -0.048, -0.045, -0.045, -0.045, -0.044, -0.045, -0.047, -0.048, -0.049, -0.045, -0.033, -0.016],
            [10,   -0.043, -0.044, -0.041, -0.041, -0.040, -0.038, -0.034, -0.035, -0.035, -0.029, -0.022, -0.009],
            [20,   -0.052, -0.034, -0.036, -0.036, -0.035, -0.028, -0.024, -0.023, -0.020, -0.016, -0.010, -0.014],
            [30,   -0.062, -0.034, -0.027, -0.028, -0.027, -0.027, -0.023, -0.023, -0.019, -0.009, -0.025, -0.010]
        ]).T)

    tables['thrust_idle'] = create_table2D(
        name='thrust_idle', row_label='alt_ft', col_label='mach',
        data=np.array([   # alt, ft
            [0,    0,     1.0e4, 2.0e4, 3.0e4, 4.0e4, 5.0e4],
            [0,    1060,  670,   890,   1140,  1500,  1860],
            [0.2,  635,   425,   690,   1010,  1330,  1700],
            [0.4,  60,    25,    345,   755,   1130,  1525],
            [0.6, -1020, -710,  -300,   350,   910,   1360],  # mach
            [0.8, -2700, -1900, -1300, -247,   600,   1100],
            [1.0, -3600, -1400, -595,  -342,  -200,   700],
        ]).T)

    tables['thrust_mil'] = create_table2D(
        name='thrust_mil', row_label='alt_ft', col_label='mach',
        data=np.array([   # alt, ft
            [0,     0, 1.0e4, 2.0e4, 3.0e4, 4.0e4, 5.0e4],
            [0, 12680,  9150,  6200,  3950,  2450,  1400],
            [0.2, 12680,  9150,  6313,  4040,  2470,  1400],
            [0.4, 12610,  9312,  6610,  4290,  2600,  1560],  # mach
            [0.6, 12640,  9839,  7090,  4660,  2840,  1660],
            [0.8, 12390, 10176,  7750,  5320,  3250,  1930],
            [1.0, 11680,  9848,  8050,  6100,  3800,  2310]
        ]).T)

    tables['thrust_max'] = create_table2D(
        name='thrust_max', row_label='alt_ft', col_label='mach',
        data=np.array([   # alt, ft
            [0,   0,     1.0e4, 2.0e4,  3.0e4, 4.0e4, 5.0e4],
            [0,   20000, 15000, 10800,  7000,  4000,  2500],
            [0.2, 21420, 15700, 11225,  7323,  4435,  2600],
            [0.4, 22700, 16860, 12250,  8154,  5000,  2835],  # mach
            [0.6, 24240, 18910, 13760,  9285,  5700,  3215],
            [0.8, 26070, 21075, 15975,  11115, 6860,  3950],
            [1.0, 28886, 23319, 18300,  13484, 8642,  5057]
        ]).T)

    def thrust():
        power = ca.MX.sym('power')
        alt = ca.MX.sym('alt')
        rmach = ca.MX.sym('rmach')
        tidl = tables['thrust_idle'](alt, rmach)
        tmil = tables['thrust_mil'](alt, rmach)
        tmax = tables['thrust_max'](alt, rmach)
        thrust = ca.if_else(power < 50,
                            tidl + (tmil - tidl)*power*0.02,
                            tmil + (tmax - tmil)*(power - 50)*0.02)
        return ca.Function('thrust',
                           [power, alt, rmach],
                           [thrust],
                           ['power', 'alt', 'mach'],
                           ['thrust'])
    tables['thrust'] = thrust()

    def propulsion():
        dp = ca.MX.sym('dp')
        thtl = ca.MX.sym('thtl')
        power = ca.MX.sym('power')
        power_cmd = ca.MX.sym('power_cmd')

        # reciprocal of time constant
        rtau = ca.Function('rtau', [dp], [ca.if_else(dp < 25, 1, ca.if_else(dp > 50, 0.1, 1.9 - 0.036*dp))])

        # power command vs. throttle relationship
        tgear = ca.Function('tgear', [thtl],
                            [ca.if_else(thtl < 0.77, 64.94*thtl, 217.38*thtl - 117.38)],
                            ['thtl'], ['pow'])

        # rate of change of power
        pdot = ca.Function('pdot', [power, power_cmd], [
            ca.if_else(power_cmd > 50,
                       ca.if_else(power > 50, 5*(power_cmd - power), rtau(60 - power)*(60 - power)),
                       ca.if_else(power > 50, 5*(40 - power), rtau(power_cmd - power)*(power_cmd - power))
                       )
        ], ['power', 'power_cmd'], ['pdot'])
        tables['tgear'] = tgear
        tables['pdot'] = pdot
    propulsion()

    def atmosphere():
        vt = ca.MX.sym('vt')
        alt = ca.MX.sym('alt')
        R0 = 2.377e-3
        Tfac = 1 - 0.703e-5*alt
        T = ca.if_else(alt > 35000, 390, 519*Tfac)
        rho = R0*(Tfac**4.14)
        tables['amach'] = ca.Function('amach', [vt, alt], [vt/(ca.sqrt(1.4*1716.3*T))], ['vt', 'alt'], ['amach'])
        tables['qbar'] = ca.Function('qbar', [vt, alt], [0.5*rho*vt**2], ['vt', 'alt'], ['qbar'])
        tables['ps'] = ca.Function('qbar', [alt], [1715*rho*T], ['alt'], ['amach'])
    atmosphere()

    return tables


class CasadiDataClass:

    def to_casadi(self):
        return ca.vertcat(*astuple(self))

    @classmethod
    def from_casadi(cls, v):
        return cls(*[v[i] for i in range(v.shape[0])])


@dataclass
class State(CasadiDataClass):
    VT: float = 0  # true velocity, (ft/s)
    alpha: float = 0  # angle of attack, (rad)
    beta: float = 0  # sideslip angle, (rad)
    phi: float = 0  # B321 roll angle, (rad)
    theta: float = 0  # B321 pitch angle, (rad)
    psi: float = 0  # B321 yaw angle, (rad)
    P: float = 0  # body roll rate, (rad/s)
    Q: float = 0  # body pitch rate, (rad/s)
    R: float = 0  # body yaw rate, (rad/s)
    p_N: float = 0  # north position, (m)
    p_E: float = 0  # east position, (m)
    alt: float = 0  # altitude, (m)
    power: float = 0  # power, (0-1)


@dataclass
class StateDot(CasadiDataClass):
    VT_dot: float = 0  # true velocity derivative, (ft/s^2)
    alpha_dot: float = 0  # angle of attack rate, (rad/s)
    beta_dot: float = 0  # sideslip rate, (rad/s)
    phi_dot: float = 0  # B321 roll rate, (rad/s)
    theta_dot: float = 0  # B321 pitch rate, (rad/s)
    psi_dot: float = 0  # B321 yaw rate, (rad/s)
    P_dot: float = 0  # body roll accel, (rad/s^2)
    Q_dot: float = 0  # body pitch accel, (rad/s^2)
    R_dot: float = 0  # body yaw accel, (rad/s^2)
    V_N: float = 0  # north velocity, (m/s)
    V_E: float = 0  # east velocity, (m/s)
    alt_dot: float = 0  # climb rate, (m/s)
    power_dot: float = 0  # power rate, (NA)


@dataclass
class Control(CasadiDataClass):
    thtl: float = 0  # throttle (0-1)
    ail_deg: float = 0  # aileron, (deg)
    elv_deg: float = 0  # elevator, (deg)
    rdr_deg: float = 0  # rudder, (deg)


@dataclass
class Parameters(CasadiDataClass):
    s: float = 300  # reference area, ft^2
    b: float = 30  # wing span, ft
    cbar: float = 11.32  # mean chord, ft
    xcgr: float = 0.35  # reference cg, %chord
    xcg: float = 0.35  # actual cg, %chord
    hx: float = 160
    g: float = 32.17  # acceleration of gravity, ft/s^2
    weight: float = 20490.446  # weight, slugs
    axx: float = 9496.0  # moment of inertia about x
    ayy: float = 55814.0  # moment of inertia about y
    azz: float = 63100.0  # moment of inertia about z
    axz: float = 982.0  # xz moment of inertia


def dynamics(x: State, u: Control, p: Parameters, tables):
    dx = StateDot()

    # functions
    cos = np.cos
    sin = np.sin

    # mass properties
    mass = p.weight/p.g
    axzs = p.axz*p.axz
    xqr = p.azz*(p.azz - p.ayy) + axzs
    xpq = p.axz*(p.axx - p.ayy + p.azz)
    zpq = (p.axx - p.ayy)*p.axx + axzs
    gam = p.axx*p.azz - axzs
    ypr = p.azz - p.axx

    # air data computer and engine model
    amach = tables['amach'](x.VT, x.alt)
    qbar = tables['qbar'](x.VT, x.alt)
    power_cmd = tables['tgear'](u.thtl)
    dx.power_dot = tables['pdot'](x.power, power_cmd)
    thrust = tables['thrust'](x.power, x.alt, amach)

    # look-up tables and component buildup
    rad2deg = 180/np.pi
    alpha_deg = rad2deg*x.alpha
    beta_deg = rad2deg*x.beta

    cxt = tables['Cx'](alpha_deg, u.elv_deg)
    cyt = tables['Cy'](beta_deg, u.ail_deg, u.rdr_deg)
    czt = tables['Cz'](alpha_deg, beta_deg, u.elv_deg)
    dail = u.ail_deg/20.0
    drdr = u.rdr_deg/30.0

    clt = tables['Cl'](alpha_deg, beta_deg) \
        + tables['DlDa'](alpha_deg, beta_deg)*dail \
        + tables['DlDr'](alpha_deg, beta_deg)*drdr
    cmt = tables['Cm'](alpha_deg, u.elv_deg)
    cnt = tables['Cn'](alpha_deg, beta_deg) \
        + tables['DnDa'](alpha_deg, beta_deg)*dail \
        + tables['DnDr'](alpha_deg, beta_deg)*drdr

    # add damping derivatives
    tvt = 0.5/x.VT
    b2v = p.b*tvt
    cq = p.cbar*x.Q*tvt
    cxt += cq*tables['CXq'](alpha_deg)
    cyt += b2v*(tables['CYr'](alpha_deg)*x.R + tables['CYp'](alpha_deg)*x.P)
    czt += cq*tables['CZq'](alpha_deg)
    clt += b2v*(tables['Clr'](alpha_deg)*x.R + tables['Clp'](alpha_deg)*x.P)
    cmt += cq*tables['Cmq'](alpha_deg) + czt*(p.xcgr - p.xcg)
    cnt += b2v*(tables['Cnr'](alpha_deg)*x.R + tables['Cnp'](alpha_deg)*x.P) - cyt*(p.xcgr - p.xcg)*p.cbar/p.b

    # get ready for state equations
    cbta = cos(x.beta)
    U = x.VT*cos(x.alpha)*cbta
    V = x.VT*sin(x.beta)
    W = x.VT*sin(x.alpha)*cbta
    sth = sin(x.theta)
    cth = cos(x.theta)
    sph = sin(x.phi)
    cph = cos(x.phi)
    spsi = sin(x.psi)
    cpsi = cos(x.psi)
    qs = qbar*p.s
    qsb = qs*p.b
    rmqs = qs/mass
    gcth = p.g*cth
    qsph = x.Q*sph
    ay = rmqs*cyt
    az = rmqs*czt

    # force equations
    U_dot = x.R*V - x.Q*W - p.g*sth + (qs*cxt + thrust)/mass
    V_dot = x.P*W - x.R*U + gcth*sph + ay
    W_dot = x.Q*U - x.P*V + gcth*cph + az
    tmp = U**2 + W**2

    dx.VT_dot = (U*U_dot + V*V_dot + W*W_dot)/x.VT
    dx.alpha_dot = (U*W_dot - W*U_dot) / tmp
    dx.beta_dot = (x.VT*V_dot - V*dx.VT_dot)*cbta/tmp

    # kinematics
    dx.phi_dot = x.P + (sth/cth)*(qsph + x.R*cph)
    dx.theta_dot = x.Q*cph - x.R*sph
    dx.psi_dot = (qsph + x.R*cph)/cth

    # moments
    roll = qsb*clt
    pitch = qs*p.cbar*cmt
    yaw = qsb*cnt
    pq = x.P*x.Q
    qr = x.Q*x.R
    qhx = x.Q*p.hx

    dx.P_dot = (xpq*pq - xqr*qr + p.azz*roll + p.axz*(yaw + qhx)) / gam
    dx.Q_dot = (ypr*x.P*x.R - p.axz*(x.P**2 - x.R**2) +
                pitch - x.R*p.hx) / p.ayy
    dx.R_dot = (zpq*pq - xpq*qr + p.axz*roll + p.axx*(yaw + qhx)) / gam

    # navigation
    t1 = sph*cpsi
    t2 = cph*sth
    t3 = sph*spsi
    s1 = cth*cpsi
    s2 = cth*spsi
    s3 = t1*sth - cph*spsi
    s4 = t3*sth + cph*cpsi
    s5 = sph*cth
    s6 = t2*cpsi + t3
    s7 = t2*spsi - t1
    s8 = cph*cth

    dx.V_N = U*s1 + V*s3 + W*s6
    dx.V_E = U*s2 + V*s4 + W*s7
    dx.alt_dot = U*sth - V*s5 - W*s8
    return dx
