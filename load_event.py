import dynamic as dps
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import RK23, RK45, solve_ivp
import importlib
import time
import sys
import modal_analysis as dps_mdl
import plotting as dps_plt


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
    sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=2e-2)
    result_dict = defaultdict(list)

    # Additional plot variables below. All states are stored by default, but algebraic variables like powers and
    # currents have to be specified in the lists below. Currently supports GEN output and input variables, and
    # powers ['P_l', 'Q_l', 'S_l'] at load buses.
    #avr_outs = []
    #gov_outs = []
    gen_vars = ['P_e','I_g','P_m']
    load_vars = ['P_l','Q_l']  # l subscript means "load"

    gen_var_desc = ps.var_desc('GEN',gen_vars)
    load_var_desc = ps.var_desc('load',load_vars)
    
    v_bus_full = ps.red_to_full.dot(ps.v_red)
    v_bus_mag = np.abs(v_bus_full)
    
    Volt_stored = []

    #event_flag = True
    event_flag2 = True
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/t_end*100))

        #if t >= 1 and event_flag:
            #event_flag = False

            #ps.network_event('line','L6-7', 'disconnect')
        if 5 < t <10:
            ps.y_bus_red_mod[4,4] = 0.0 + 1j*0.09
        else:
            ps.y_bus_red_mod[4,4] = 0

        #if t >= 5 and event_flag2:
            #event_flag2 = False
            #ps.network_event('load_change','L1', 'activate' ,dS = 90)  # dS in MVA (not pu), can be complex

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t
        
        v_bus_full = ps.red_to_full.dot(ps.v_red)
        v_bus_mag = np.abs(v_bus_full)

        # Store result
        result_dict['Global', 't'].append(sol.t)                                                # Time
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]       # States
        ps.store_vars('GEN',gen_vars, gen_var_desc, result_dict)                                # Additional gen vars
        ps.store_vars('load',load_vars, load_var_desc, result_dict)                             # Load vars
        Volt_stored.append(v_bus_mag[6])

    print('   Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))
    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)
    t_plot = result[('Global', 't')]
    # Plotting section
    fig, ax = plt.subplots(1, sharex = True)

    var1 = 'speed'                                      # variable to plot
    p1 = result.xs(key=var1, axis='columns', level=1)   # time domain values for var1
    #p1 = p1[['G1']]                                     # Double brackets to access specific devices (e.g. G1)
    legnd1 = list(np.array(var1 + ': ')+p1.columns)     # legend for var1

    var2 = 'P_l'                                      # variable to plot
    p2 = result.xs(key=var2, axis='columns', level=1)   # time domain values for var2
    p2 = p2[['L1']]                                     # Double brackets to access specific devices (e.g. G1)
    legnd2 = list(np.array(var2 + ': ') + p2.columns)   # legend for var2

    var3 = 'P_e'  # variable to plot
    p3 = result.xs(key=var3, axis='columns', level=1)
    #p3 = p3[['G1']]
    legnd3 = list(np.array(var3 + ': ') + p3.columns)

    var4 = 'P_m'  # variable to plot
    p4 = result.xs(key=var4, axis='columns', level=1)
    #p4 = p4[['G1']]
    legnd4 = list(np.array(var4 + ': ') + p4.columns)
    
    var5 = 'V'
    p5 = Volt_stored

   # ax[1].plot(t_plot, p2.values)
    ax.plot(t_plot, p3.values)                              # Plotting two variables in same plot
    ax.plot(t_plot, p4.values)
    ax.legend(legnd3 + legnd4,bbox_to_anchor=(1,1))
    ax.set_ylabel('Power gen')
    
    fig, ax2 = plt.subplots(1)
    ax2.plot(t_plot, p2.values)
    ax2.legend(legnd2,bbox_to_anchor=(1,1))
    ax2.set_ylabel('Power load')
    
    fig, ax1 = plt.subplots(1)
    ax1.plot(t_plot, p1.values)
    ax1.legend(legnd1, bbox_to_anchor=(1,1))
    ax1.set_ylabel('Speed')
    
    fig, ax3 = plt.subplots(1)
    ax3.plot(t_plot, p5)
    ax3.set_ylabel('Voltage magnitude')
    

    fig.text(0.5, 0.04, 'Time [seconds]', ha='center')
    plt.show()