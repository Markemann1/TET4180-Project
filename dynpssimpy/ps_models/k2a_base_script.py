from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import dynpssimpy.modal_analysis as dps_mdl
import dynpssimpy.plotting as dps_plt
import importlib
importlib.reload(dps)


if __name__ == '__main__':

    # Load model
    import dynpssimpy.ps_models.k2a_no_controls as model_data
    model = model_data.load()

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    # ps.use_numba = True
    # Power flow calculation
    ps.power_flow()
    # Initialization
    ps.init_dyn_sim()
    #
    np.max(ps.ode_fun(0.0, ps.x0))
    # Specify simulation time
    #
    t_end = 20
    x0 = ps.x0.copy()
    # Add small perturbation to initial angle of first generator
    # x0[ps.gen_mdls['GEN'].state_idx['angle'][0]] += 1
    #
    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x0, t_end, max_step=5e-3)

    # Define other variables to plot
    P_e_stored = []
    E_f_stored = []
    Igen= 0,0
    I_stored = []
    v_bus = []
    # Initialize simulation
    t = 0
    result_dict = defaultdict(list)
    t_0 = time.time()
    # ps.build_y_bus_red(ps.buses['name'])
    ps.build_y_bus(['B8'])
    print('Ybus_full = ', ps.y_bus_red_full)
    print('Ybus_red = ', ps.y_bus_red)

    v_bus_mag = np.abs(ps.v_0)
    v_bus_angle = ps.v_0.imag / v_bus_mag
    #
    print(' ')
    print('Voltage magnitudes (p.u) = ', v_bus_mag)
    print(' ')
    print('Voltage angles     (rad) = ', v_bus_angle)
    print(' ')
    # ** adding angles in degrees
    print('Voltage angles     (deg) = ', v_bus_angle*180/np.pi)
    print(' ')
    # ****
    print('Voltage magnitudes  (kV) = ', v_bus_mag*[20, 20, 20, 20, 230, 230, 230, 230, 230, 230, 230])
    print(' ')
    # print(ps.v_n)
    print('v_vector = ', ps.v_0)
    print(' ')
    # print('Forskjell på red og full Ybus = ',ps.y_bus_red_full - ps.y_bus_red)
    #
    print('state description: ', ps.state_desc)
    print('Initial values on all state variables (G1 and IB) :')
    print(x0)
    print(' ')
    # Adding event flags for c)
    event_flag = True
    event_flag2 = True
    # Run simulation
    while t < t_end:
        # print(t)
        #v_bus_full = ps.red_to_full.dot(ps.v_red)
        # Simulate short circuit
        """if 1 < t < 1.1:
          ps.y_bus_red_mod[7, 7] = 10000
        else:
           ps.y_bus_red_mod[7, 7] = 0"""
        """# simulate a short circuit at bus 5 - preliminaries
        if 1 < t < 1.05:
            ps.y_bus_red_mod[4, 4] = 10000
        else:
            ps.y_bus_red_mod[4, 4] = 0"""
        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        # C) Compare Modal Analysis to time-domain response
        if t >= 1 and event_flag:
            event_flag = False
            ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][5], 'disconnect')
        if t >= 1.05 and event_flag2:
            event_flag2 = False
            ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][5], 'connect')
            ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][4], 'disconnect')


        # Store result
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]
        Igen = ps.y_bus_red_full[7,8]*(v[8] -v[7])
        # Legger til nye outputs
        P_e_stored.append(ps.gen['GEN'].P_e(x, v).copy())
        E_f_stored.append(ps.gen['GEN'].E_f(x, v).copy())
        I_stored.append(np.abs(Igen))

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    # ** Obtaining power flows
    # ps.power_flow(True)
    print(ps.s_0)

    # P15 = (v_bus_mag[0]-v_bus_mag[4]) * ps.y_bus_red_full[0][4]
    # print(P15)

    # ****

    # Convert dict to pandas dataframe
    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    # Plot angle and speed
    fig, ax = plt.subplots(3)
    fig.suptitle('Generator speed, angle and electric power')
    ax[0].plot(result[('Global', 't')], result.xs(key='speed', axis='columns', level=1))
    ax[0].set_ylabel('Speed (p.u.)')
    ax[0].legend(['G1', 'G2', 'G3', 'G4'],loc='upper center', bbox_to_anchor=(1,1))
    ax[1].plot(result[('Global', 't')], result.xs(key='angle', axis='columns', level=1))
    ax[1].set_ylabel('Angle (rad.)')
    ax[1].legend(['G1', 'G2', 'G3', 'G4'],loc='upper center', bbox_to_anchor=(1,1))
    ax[2].plot(result[('Global', 't')], np.array(P_e_stored)/[900, 900, 900, 900])
    ax[2].set_ylabel('Power (p.u.)')
    ax[2].set_xlabel('time (s)')
    ax[2].legend(['G1', 'G2', 'G3', 'G4'],loc='upper center', bbox_to_anchor=(1,1))

    plt.figure()
    plt.plot(result[('Global', 't')], np.array(E_f_stored))
    plt.xlabel('time (s)')
    plt.ylabel('E_q (p.u.)')

    plt.figure()
    plt.plot(result[('Global', 't')], np.array(I_stored))
    plt.xlabel('time (s)')
    plt.ylabel('I_8-9 (magnitude p.u.)')

    # ** Plot for 1a)
    """plt.figure()
    plt.plot(result[('Global', 't')], result.xs(key='V_t_abs', axis='columns', level=1))"""

    # ****

    # Plot eigenvalues in c)

    # Perform system linearization
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    # Plot eigenvalues
    dps_plt.plot_eigs(ps_lin.eigs)
    plt.axis([-4, 4, -4, 4])
    print(ps_lin.eigs)

    # Get mode shape for electromechanical modes
    mode_idx = ps_lin.get_mode_idx(['em'], damp_threshold=0.3)
    rev = ps_lin.rev
    mode_shape = rev[np.ix_(ps.gen['GEN'].state_idx_global['speed'], mode_idx)]

    # Plot mode shape
    fig, ax = plt.subplots(1, mode_shape.shape[1], subplot_kw={'projection': 'polar'})
    for ax_, ms in zip(ax, mode_shape.T):
        dps_plt.plot_mode_shape(ms, ax=ax_, normalize=True)
    # plt.show()

    plt.show()
