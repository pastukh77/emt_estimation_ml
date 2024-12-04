data_config = {
    "file_path": "data\measures_v2.csv",
    "input_param_names": ["ambient",
                        "coolant",
                        "u_d",
                        "u_q",
                        "motor_speed",
                        "torque",
                        "i_d",
                        "i_q"
                        ],
    "target_param_names": ["pm",
                        "stator_yoke",
                        "stator_tooth",
                        "stator_winding"
                        ],
    "lags": [840, 960, 1080, 1200],
    "valset": [58],
    "testset": [65, 72],
}