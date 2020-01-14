import numpy as np
# ---------------------attack config------------------------#
attack_params = {
    "FGSM_MNIST": {
    'epsilon': 0.2,
    'order': np.inf,
    'clip_max': None,
    'clip_min': None
    },

    "PGD_CIFAR10": {
    'epsilon': 0.1,
    'clip_max': (1 - 0.4914) / 0.2023,
    'clip_min': (0-0.4914) / 0.2023,
    'print_process': True
    },

    "LBFGS_MNIST": {
    'epsilon': 1e-4,
    'maxiter': 20,
    'clip_max': 1,
    'clip_min': 0,
    'class_num': 10
    },

    "CW_MNIST": {
    'confidence': 1e-4,
    'clip_max': 1,
    'clip_min': 0,
    'max_iterations': 1000,
    'initial_const': 1e-2,
    'binary_search_steps': 5,
    'learning_rate': 5e-3,
    'abort_early': True,
    }

}

#-----------defense(Adversarial training) config------------#

defense_params = {
    "PGDtraining_MNIST":{
        'save_dir': dir,
        'save_model': True
    } 
}