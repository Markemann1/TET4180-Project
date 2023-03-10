# Synchronous machine connected to infinite bus

def load():
    return {
        'base_mva': 50,
        'f': 50,
        'slack_bus': 'B3',

        'buses': [
            ['name',    'V_n'],
            ['B1',      10],
            ['B2',      245],
            ['B3',      245],
        ],

        'lines': [
            ['name',    'from_bus', 'to_bus',   'length',   'S_n',  'V_n',  'unit', 'R',    'X',      'B'],
            ['L1-2',    'B2',       'B3',       250,         50,    245,     'ohm',   0,   0.4,     0],
        ],

        'transformers': [
            ['name',    'from_bus', 'to_bus',   'S_n',  'V_n_from', 'V_n_to',   'R',    'X'],
            ['T1',      'B1',       'B2',       50,    10,         245,        0,      0.1],
        ],

        'loads': [
            ['name', 'bus', 'P', 'Q', 'model'],
            ['L1', 'B2', 25, 0, 'Z'],
        ],

        'generators': {
            'GEN': [
                ['name',    'bus',  'S_n',  'V_n',  'P',    'V',        'H',        'D',    'X_d',  'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st'],
                ['G1',      'B1',    50,      10,     40,    0.93,       6.5,        0,     1.05,   0.66,   0.328,        0.66,        0.254,        0.273,        8.0,        0.6,        0.05,       0.05],
                ['IB',      'B3',   10000,   245,    -15,    0.898,       10.0,        0,     1.8,    1.8,    0.3,        0.3,        0.2,        0.2,        6.67,       6.67,       0.15,       0.15],
            ],
        },

        'gov_on': False,
        'avr_on': False,
        'pss_on': False,  # not used yet

        'gov': {
            'MYGOV': [
                ['name', 'gen', 'R', 'K','Kw'],
                ['GOV1', 'G1', 0.1, 100, 10],
                ]
        },

        'avr': {
            'SEXS': [
                ['name', 'gen', 'K', 'T_a', 'T_b', 'T_e', 'E_min', 'E_max'],
                ['AVR1', 'G1', 100, 2.0, 10.0, 0.5, -3, 3],
            ]
        },
    }