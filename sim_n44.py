# For running simulation in N44


import dynamic as dps
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import utility_functions as dps_uf
import importlib
import sys
import time
import numpy as np

# Line outage when HYGOV is used
# Added for not making fatal changes to the original file


if __name__ == '__main__':
    importlib.reload(dps)

    # Load model
    import n44 as model_data
    # import ps_models.ieee39 as model_data
    # import ps_models.sm_ib as model_data
    # import ps_models.sm_load as model_data
    model = model_data.load()


    # Calling model twice, first is to get the desired names and so on.
    # Could probably do it another way, but this works
    ps = dps.PowerSystemModel(model=model)

    # Add controls for all generators (not specified in model)
    """
    model['gov'] = {'TGOV1':
                        [['name', 'gen', 'R', 'D_t', 'V_min', 'V_max', 'T_1', 'T_2', 'T_3']] +
                        [['GOV' + str(i), gen_name, 0.05, 0, 0, 1, 0.2, 1, 2] for i, (gen_name, gen_p) in
                         enumerate(zip(ps.generators['name'], ps.generators['P']))]
                    }
    model['avr'] = {'SEXS':
                        [['name', 'gen', 'K', 'T_a', 'T_b', 'T_e', 'E_min', 'E_max']] +
                        [['AVR' + str(i), gen_name, 100, 2.0, 10.0, 0.5, -3, 3] for i, gen_name in
                         enumerate(ps.generators['name'])]
                    }
    #model.pop('avr')
    model['pss'] = {'STAB1':
                        [['name', 'gen', 'K', 'T', 'T_1', 'T_2', 'T_3', 'T_4', 'H_lim']] +
                       [['PSS' + str(i), gen_name, 50, 10.0, 0.5, 0.5, 0.05, 0.05, 0.03] for i, gen_name in
                         enumerate(ps.generators['name'])]
                    }
    #model.pop('pss')
    """

    # Power system with governos, avr and pss
    ps = dps.PowerSystemModel(model=model)
    #ps.use_numba = True

    ps.pf_max_it = 100
    ps.power_flow()
    ps.init_dyn_sim()
    
    v_bus_full = ps.red_to_full.dot(ps.v_red)
    v_bus_full_conj = np.conj(v_bus_full)
    v_bus_mag = np.abs(v_bus_full)
    #print(v_bus_full)
    
    S12 = ps.y_bus[23,24]*(v_bus_full[24] - v_bus_full[23])*v_bus_full_conj[23]
    P12 = S12.real
    Q12 = -S12.imag
    
    S23 = ps.y_bus[24,13]*(v_bus_full[13] - v_bus_full[24])*v_bus_full_conj[24]
    P23 = S23.real
    Q23 = -S23.imag
    
    S34 = ps.y_bus[13,9]*(v_bus_full[9] - v_bus_full[13])*v_bus_full_conj[13]
    P34 = S34.real
    Q34 = -S34.imag
    
    S45 = ps.y_bus[9,4]*(v_bus_full[4] - v_bus_full[9])*v_bus_full_conj[9]
    P45 = S45.real
    Q45 = -S45.imag
    
    S56 = ps.y_bus[4,8]*(v_bus_full[8] - v_bus_full[4])*v_bus_full_conj[4]
    P56 = S56.real
    Q56 = -S56.imag
    
    S47 = ps.y_bus[9, 10]*(v_bus_full[10]-v_bus_full[9])*v_bus_full_conj[9]
    P47 = S47.real
    Q47 = -S47.imag
    
    S68 = ps.y_bus[8,0]*(v_bus_full[0] - v_bus_full[8])*v_bus_full_conj[8]
    P68 = S68.real
    Q68 = -S68.imag
    
    S89 = ps.y_bus[0,1]*(v_bus_full[1] - v_bus_full[0])*v_bus_full_conj[0]
    P89 = S89.real
    Q89 = -S89.imag
    
    print('----------------------------------')
    print('P from 5500 to 5501 = ', "{:.2f}".format(P12*1000), ' MW')
    print('Q from 5500 to 5501 = ', "{:.2f}".format(Q12*1000), ' MVAr')
    print('----------------------------------')
    print('P from 5501 to 5101 = ', "{:.2f}".format(P23*1000), ' MW')
    print('Q from 5501 to 5101 = ', "{:.2f}".format(Q23*1000), ' MVAr')
    print('----------------------------------')
    print('P from 5101 to 3359 = ', "{:.2f}".format(P34*1000), ' MW')
    print('Q from 5101 to 3359 = ', "{:.2f}".format(Q34*1000), ' MVAr')
    print('----------------------------------')
    print('P from 3359 to 3360 = ', "{:.2f}".format(P47*1000), ' MW')
    print('Q from 3359 to 3360 = ', "{:.2f}".format(Q47*1000), ' MVAr')
    print('----------------------------------')
    print('P from 3359 to 3200 = ', "{:.2f}".format(P45*1000), ' MW')
    print('Q from 3359 to 3200 = ', "{:.2f}".format(Q45*1000), ' MVAr')
    print('----------------------------------')
    print('P from 3200 to 3300 = ', "{:.2f}".format(P56*1000), ' MW')
    print('Q from 3200 to 3300 = ', "{:.2f}".format(Q56*1000), ' MVAr')
    print('----------------------------------')
    
    print(' ')
    
    print('----------------------------------')
    print('Voltage at bus 5500 = ', "{:.2f}".format(v_bus_mag[23]*300), ' kV')
    print('----------------------------------')
    print('Voltage at bus 5501 = ', "{:.2f}".format(v_bus_mag[24]*420), ' kV')
    print('----------------------------------')
    print('Voltage at bus 5101 = ', "{:.2f}".format(v_bus_mag[13]*420), ' kV')
    print('----------------------------------')
    print('Voltage at bus 3359 = ', "{:.2f}".format(v_bus_mag[9]*420), ' kV')
    print('----------------------------------')
    print('Voltage at bus 3360 = ', "{:.2f}".format(v_bus_mag[10]*135), ' kV')
    print('----------------------------------')
    print('Voltage at bus 3200 = ', "{:.2f}".format(v_bus_mag[4]*420), ' kV')
    print('----------------------------------')
    print('Voltage at bus 3300 = ', "{:.2f}".format(v_bus_mag[8]*420), ' kV')
    print('----------------------------------')
    
    # Solver
    t_end = 100
    sol = dps_uf.ModifiedEuler(ps.ode_fun, 0, ps.x0, t_end, max_step=30e-3)

    t = 0
    result_dict = defaultdict(list)
    t_0 = time.time()

    gen_vars = ['P_e', 'I_g','P_m']
    load_vars = ['P_l']  # l subscript means "load"

    gen_var_desc = ps.var_desc('GEN',gen_vars)
    load_var_desc = ps.var_desc('load',load_vars)

    event_flag = True
    event_flag2 = True
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t

        if t > 5 and event_flag:
            event_flag = False
            ps.network_event('line', 'L3359-5101-1', 'disconnect')
            #ps.network_event('line', 'L5101-5501', 'disconnect')
            #ps.network_event('line', 'L3200-3359', 'disconnect')
        if t > 5.05 and event_flag2:
            event_flag2 = False
            #ps.network_event('line', 'L5101-5501', 'connect')
            #ps.network_event('line', 'L3200-3359', 'connect')

        # Store result
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]

        # Store generator values
        ps.store_vars('GEN', gen_vars, gen_var_desc, result_dict)
        ps.store_vars('load',load_vars, load_var_desc, result_dict) 

        # Store ACE signals
        if bool(ps.ace_mdls):
            for key, dm in ps.ace_mdls.items():
                # Store ACE signals
                [result_dict[tuple([n, 'ace'])].append(ace) for n, ace in zip(dm.par['name'], dm.ace)]
                # Store instantaneous line flows
                [result_dict[tuple([n, 'p_tie'])].append(p_tie*ps.s_n) for n, p_tie in zip(dm.par['name'], dm.p_tie)]
                # Store scheduled P_tie values
                [result_dict[tuple([n, 'p_tie0'])].append(p_tie0 * ps.s_n) for n, p_tie0 in zip(dm.par['name'], dm.int_par['Ptie0'])]

    print('\nSimulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    fig, ax = plt.subplots(1, sharex = True)
    
    var1 = 'speed'
    p1 = result.xs(key='speed', axis='columns', level=1)
    p1_1 = p1[['G3359-1']]
    p1_2 = p1[['G3359-2']]
    p1_3 = p1[['G3359-3']]
    p1_4 = p1[['G3359-4']]
    p1_5 = p1[['G3359-5']]
    p1_6 = p1[['G3359-6']]
    p1_7 = p1[['G3300-1']]
    p1_8 = p1[['G3300-2']]
    p1_9 = p1[['G3300-3']]
    p1_10 = p1[['G5500-1']]
    legnd1_1 = list(np.array(var1 + ': ') + p1_1.columns)
    legnd1_2 = list(np.array(var1 + ': ') + p1_2.columns)
    legnd1_3 = list(np.array(var1 + ': ') + p1_3.columns)
    legnd1_4 = list(np.array(var1 + ': ') + p1_4.columns)
    legnd1_5 = list(np.array(var1 + ': ') + p1_5.columns)
    legnd1_6 = list(np.array(var1 + ': ') + p1_6.columns)
    legnd1_7 = list(np.array(var1 + ': ') + p1_7.columns)
    legnd1_8 = list(np.array(var1 + ': ') + p1_8.columns)
    legnd1_9 = list(np.array(var1 + ': ') + p1_9.columns)
    legnd1_10 = list(np.array(var1 + ': ') + p1_10.columns)
    ax.plot(result[('Global', 't')], p1_1.values)
    ax.plot(result[('Global', 't')], p1_2.values)
    ax.plot(result[('Global', 't')], p1_3.values)
    ax.plot(result[('Global', 't')], p1_4.values)
    ax.plot(result[('Global', 't')], p1_5.values)
    ax.plot(result[('Global', 't')], p1_6.values)
    ax.plot(result[('Global', 't')], p1_7.values)
    ax.plot(result[('Global', 't')], p1_8.values)
    ax.plot(result[('Global', 't')], p1_9.values)
    ax.plot(result[('Global', 't')], p1_10.values)
    ax.set_ylabel('Speed')
    ax.legend(legnd1_1 + legnd1_2 + legnd1_3 + legnd1_4 + legnd1_5 + legnd1_6 + legnd1_7 + legnd1_8 + legnd1_9 + legnd1_10, bbox_to_anchor=(1,1))
    ax.grid(True)
    
    fig, ax1 = plt.subplots(1)
    var2 = 'angle'
    p2 = result.xs(key='angle', axis='columns', level=1)
    p2_1 = p2[['G3359-1']]
    p2_2 = p2[['G3359-2']]
    p2_3 = p2[['G3359-3']]
    p2_4 = p2[['G3359-4']]
    p2_5 = p2[['G3359-5']]
    p2_6 = p2[['G3359-6']]
    p2_7 = p2[['G3300-1']]
    p2_8 = p2[['G3300-2']]
    p2_9 = p2[['G3300-3']]
    p2_10 = p2[['G5500-1']]
    legnd2_1 = list(np.array(var2 + ': ') + p2_1.columns)
    legnd2_2 = list(np.array(var2 + ': ') + p2_2.columns)
    legnd2_3 = list(np.array(var2 + ': ') + p2_3.columns)
    legnd2_4 = list(np.array(var2 + ': ') + p2_4.columns)
    legnd2_5 = list(np.array(var2 + ': ') + p2_5.columns)
    legnd2_6 = list(np.array(var2 + ': ') + p2_6.columns)
    legnd2_7 = list(np.array(var2 + ': ') + p2_7.columns)
    legnd2_8 = list(np.array(var2 + ': ') + p2_8.columns)
    legnd2_9 = list(np.array(var2 + ': ') + p2_9.columns)
    legnd2_10 = list(np.array(var2 + ': ') + p2_10.columns)
    ax1.plot(result[('Global', 't')], p2_1.values)
    ax1.plot(result[('Global', 't')], p2_2.values)
    ax1.plot(result[('Global', 't')], p2_3.values)
    ax1.plot(result[('Global', 't')], p2_4.values)
    ax1.plot(result[('Global', 't')], p2_5.values)
    ax1.plot(result[('Global', 't')], p2_6.values)
    ax1.plot(result[('Global', 't')], p2_7.values)
    ax1.plot(result[('Global', 't')], p2_8.values)
    ax1.plot(result[('Global', 't')], p2_9.values)
    ax1.plot(result[('Global', 't')], p2_10.values)
    ax1.set_ylabel('Angle')
    ax1.legend(legnd2_1 + legnd2_2 + legnd2_3 + legnd2_4 + legnd2_5 + legnd2_6 + legnd2_7 + legnd2_8 + legnd2_9 + legnd2_10, bbox_to_anchor=(1,1))
    ax1.grid(True)

    # Plot ACEs if these are included
    if bool(ps.ace_mdls):
        fig2, ax2 = plt.subplots(2)
        ax2[0].plot(result[('Global', 't')], result.xs(key='ace', axis='columns', level=1))
        ax2[0].set_title('ACE (top) and P-tie (bottom)')
        ax2[0].set_ylabel('ACE')
        ax2[1].plot(result[('Global', 't')], result.xs(key='p_tie', axis='columns', level=1))
        ax2[1].set_ylabel('P [MW]')
        # Append plot of scheduled value of active power transfer
        ax2[1].plot(result[('Global', 't')], result.xs(key='p_tie0', axis='columns', level=1))
        ax2[0].grid(True)
        ax2[1].grid(True)

    # Plot Pe to see how much each generator outputs
    var3 = 'P_e'  # variable to plot
    p3 = result.xs(key=var3, axis='columns', level=1)
    p3_1 = p3[['G3359-1']]
    p3_2 = p3[['G3359-2']]
    p3_3 = p3[['G3359-3']]
    p3_4 = p3[['G3359-4']]
    p3_5 = p3[['G3359-5']]
    p3_6 = p3[['G3359-6']]
    p3_7 = p3[['G3300-1']]
    p3_8 = p3[['G3300-2']]
    p3_9 = p3[['G3300-3']]
    p3_10 = p3[['G5500-1']]
    legnd3_1 = list(np.array(var3 + ': ') + p3_1.columns)
    legnd3_2 = list(np.array(var3 + ': ') + p3_2.columns)
    legnd3_3 = list(np.array(var3 + ': ') + p3_3.columns)
    legnd3_4 = list(np.array(var3 + ': ') + p3_4.columns)
    legnd3_5 = list(np.array(var3 + ': ') + p3_5.columns)
    legnd3_6 = list(np.array(var3 + ': ') + p3_6.columns)
    legnd3_7 = list(np.array(var3 + ': ') + p3_7.columns)
    legnd3_8 = list(np.array(var3 + ': ') + p3_8.columns)
    legnd3_9 = list(np.array(var3 + ': ') + p3_9.columns)
    legnd3_10 = list(np.array(var3 + ': ') + p3_10.columns)
    fig3, ax3 = plt.subplots(1)
    ax3.plot(result[('Global', 't')], p3_1.values)
    ax3.plot(result[('Global', 't')], p3_2.values)
    ax3.plot(result[('Global', 't')], p3_3.values)
    ax3.plot(result[('Global', 't')], p3_4.values)
    ax3.plot(result[('Global', 't')], p3_5.values)
    ax3.plot(result[('Global', 't')], p3_6.values)
    ax3.plot(result[('Global', 't')], p3_7.values)
    ax3.plot(result[('Global', 't')], p3_8.values)
    ax3.plot(result[('Global', 't')], p3_9.values)
    ax3.plot(result[('Global', 't')], p3_10.values)
    ax3.set_title('Pe')
    ax3.set_ylabel('Pe [MW]')
    ax3.set_xlabel('Time [s]')
    ax3.legend(legnd3_1 + legnd3_2 + legnd3_3 + legnd3_4 + legnd3_5 + legnd3_6 + legnd3_7 + legnd3_8 + legnd3_9 + legnd3_10, bbox_to_anchor=(1,1))
    ax3.grid(True)
    


    plt.show()
