import dynamic as dps
import modal_analysis as ma
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import RK23, RK45, solve_ivp
import importlib
import time
import sys


if __name__ == '__main__':

    # Load model
    import k2a_hygov_tgov as model_data
    # import ps_models.ieee39 as model_data
    # import ps_models.n44 as model_data

    [importlib.reload(mdl) for mdl in [dps, model_data]]
    model = model_data.load()
    t_0 = time.time()

    if not model['gov_on']: model.pop('gov', None)
    if not model['avr_on']: model.pop('avr', None)
    if not model['pss_on']: model.pop('pss', None)

    ps = dps.PowerSystemModel(model=model)
    ps.pf_max_it = 10 # Max iterations for power flow
    ps.power_flow()

    # Print power flow results
    float_formatter = "{:.3f}".format
    np.set_printoptions(formatter={'float_kind': float_formatter})
    print('   Gen  |  Load |  Grid')
    print('P: {}'.format(np.real(ps.s_0)))
    print('Q: {}'.format(np.imag(ps.s_0)))
    print('V: {}'.format(np.abs(ps.v_0)))
    print('d: {}'.format(np.angle(ps.v_0)*180/np.pi))

    ps.init_dyn_sim()
    ps.ode_fun(0.0, ps.x0)
    x0 = ps.x0.copy()
    t = 0
    t_end = 20 # End of simulation
    sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=1e-2)
    result_dict = defaultdict(list)

    # Additional plot variables below. All states are stored by default, but algebraic variables like powers and
    # currents have to be specified in the lists below. Currently supports GEN output and input variables, and
    # powers ['P_l', 'Q_l', 'S_l'] at load buses.
    #avr_outs = []
    #gov_outs = []
    gen_vars = ['P_e', 'I_g','P_m']
    load_vars = ['P_l']  # l subscript means "load"

    gen_var_desc = ps.var_desc('GEN',gen_vars)
    load_var_desc = ps.var_desc('load',load_vars)

    event_flag = True
    event_flag2 = True
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/t_end*100))

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t

        if t >= 1 and event_flag:
            event_flag = False
            ps.network_event('sc','B8', 'activate')
            #ps.network_event('line', 'L7-8-2', 'disconnect')
        if t >= 1.05 and event_flag2:
            event_flag2 = False
            ps.network_event('sc', 'B8', 'deactivate')
            ps.network_event('line', 'L8-9-1', 'disconnect')
            #ps.network_event('line', 'L7-8-2', 'connect')

        # Store result
        result_dict['Global', 't'].append(sol.t)                                                # Time
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]       # States
        ps.store_vars('GEN',gen_vars, gen_var_desc, result_dict)                                # Additional gen vars
        ps.store_vars('load',load_vars, load_var_desc, result_dict)                             # Load vars

    print('   Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))
    
    # Perform system linearization
    ps_lin = ma.PowerSystemModelLinearization(ps)
    ps_lin.linearize()

    # Plot eigenvalues
    ma.dps_plt.plot_eigs(ps_lin.eigs)
    print(ps_lin.eigs)
    
    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)
    t_plot = result[('Global', 't')]
    # Plotting section
    fig, ax = plt.subplots(1, sharex = True)

    var1 = 'speed'                                      # variable to plot
    p1 = result.xs(key=var1, axis='columns', level=1)   # time domain values for var1
    legnd1 = list(np.array(var1 + ': ')+p1.columns)     # legend for var1

    var2 = 'angle'                                      # variable to plot
    p2 = result.xs(key=var2, axis='columns', level=1)   # time domain values for var2                                    # Double brackets to access specific devices (e.g. G1)
    legnd2 = list(np.array(var2 + ': ') + p2.columns)   # legend for var2

    ax.plot(t_plot, p1.values)
    ax.legend(legnd1, bbox_to_anchor=(1,1))
    ax.set_ylabel('Speed')
    ax.grid(True)

    #ax[1].plot(t_plot, p2.values)
    #ax[1].legend(legnd2,bbox_to_anchor=(1,1))
    #ax[1].set_ylabel('Power angle')
    #ax[1].grid(True)
    
    fig, ax1 = plt.subplots(1)
    var3 = 'P_e'  # variable to plot
    p3 = result.xs(key=var3, axis='columns', level=1)
    legnd3 = list(np.array(var3 + ': ') + p3.columns)
    
    var4 = 'P_m'  # variable to plot
    p4 = result.xs(key=var4, axis='columns', level=1)
    legnd4 = list(np.array(var4 + ': ') + p4.columns)
    
    ax1.plot(t_plot, p3.values)
    ax1.plot(t_plot, p4.values)
    ax1.legend(legnd3 + legnd4, bbox_to_anchor=(1,1))
    ax1.set_ylabel('Electrical power and mechanical power')
    
    

    fig.text(0.5, 0.04, 'Time [seconds]', ha='center')
    plt.show()