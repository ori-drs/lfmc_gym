# Created by Siddhant Gangapurwala

import torch

import os
import numpy as np

from common.paths import ProjectPaths
from raisim_gym_torch.algo.ppo.networks import MultiLayerPerceptron

INPUT_DIM = 48
OUTPUT_DIM = 12
HIDDEN_LAYERS = [256, 256]

ENV_NAME = 'anymal_velocity_command'
PARAMETERS_DIR = '2022-09-06-11-05-45'


def export_to_actor_pt(state_dict, network, file_path, file_name, suffix, rename_keys=True):
    if rename_keys:
        mod_state_dict = dict()

        for key in state_dict.keys():
            if 'architecture.0' in key:
                mod_state_dict[key.replace('architecture.0', '_fully_connected_layers.0')] = state_dict[key]
            elif 'architecture.2' in key:
                mod_state_dict[key.replace('architecture.2', '_fully_connected_layers.1')] = state_dict[key]
            elif 'architecture.4' in key:
                mod_state_dict[key.replace('architecture.4', '_output_layer')] = state_dict[key]

            if '_network._fully_connected_layers' in key:
                mod_state_dict[key.replace('_network._fully_connected_layers', '_fully_connected_layers')] = \
                    state_dict[key]
    else:
        mod_state_dict = state_dict

    param_save_dir = file_path[:file_path.rfind('/') + 1] + 'exported_parameters/actor_pt/'
    param_save_name = param_save_dir + file_name[:file_name.rfind('.')] + '_' + suffix + '.txt'

    os.makedirs(param_save_dir, exist_ok=True)

    network.load_state_dict(mod_state_dict)
    torch.save(network.state_dict(), param_save_name)

    print('\nSaved model parameters path:', param_save_name)


def main():
    paths = ProjectPaths()
    parameters_path_dir = paths.DATA_PATH + '/' + ENV_NAME + '/' + PARAMETERS_DIR

    network = MultiLayerPerceptron(in_dim=INPUT_DIM, out_dim=OUTPUT_DIM, hidden_layers=HIDDEN_LAYERS)
    network.eval().cpu()

    for path, _, files in os.walk(parameters_path_dir):
        for file_name in files:
            if file_name.endswith('.pt'):
                file_path = os.path.join(parameters_path_dir, file_name)
                state_dict_dict = torch.load(file_path, map_location=torch.device('cpu'))
                state_dict = state_dict_dict['actor_architecture_state_dict']

                export_to_actor_pt(state_dict, network, file_path, file_name, 'policy')


if __name__ == '__main__':
    main()
