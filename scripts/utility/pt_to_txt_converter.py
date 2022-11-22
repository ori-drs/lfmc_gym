# Created by Siddhant Gangapurwala

import torch

import os
import numpy as np

from common.paths import ProjectPaths
from raisim_gym_torch.algo.ppo.networks import MultiLayerPerceptron

INPUT_DIM = 72
OUTPUT_DIM = 16
HIDDEN_LAYERS = [512, 256, 128]

ENV_NAME = 'anymal_pmtg_velocity_command'
PARAMETERS_DIR = '2022-11-16-15-26-23'

OUTPUT_BIAS = True


def export_to_txt(state_dict, network, file_path, file_name, suffix, rename_keys=True):
    if rename_keys:
        mod_state_dict = dict()

        for key in state_dict.keys():
            if 'architecture.0' in key:
                mod_state_dict[key.replace('architecture.0', '_fully_connected_layers.0')] = state_dict[key]
            elif 'architecture.2' in key:
                mod_state_dict[key.replace('architecture.2', '_fully_connected_layers.1')] = state_dict[key]
            elif 'architecture.4' in key:
                mod_state_dict[key.replace('architecture.4', '_fully_connected_layers.2')] = state_dict[key]
            elif 'architecture.6' in key:
                mod_state_dict[key.replace('architecture.6', '_output_layer')] = state_dict[key]

            if '_network._fully_connected_layers' in key:
                mod_state_dict[key.replace('_network._fully_connected_layers', '_fully_connected_layers')] = \
                    state_dict[key]
    else:
        mod_state_dict = state_dict

    network.load_state_dict(mod_state_dict)

    model_parameters = list(network.state_dict().keys())
    model_parameters = np.concatenate(
        [network.state_dict()[key].cpu().numpy().transpose().reshape(-1) for key in model_parameters])

    if not OUTPUT_BIAS:
        model_parameters = np.concatenate((model_parameters, np.zeros(OUTPUT_DIM)))

    param_save_dir = file_path[:file_path.rfind('/') + 1] + 'exported_parameters/actor/'
    param_save_name = param_save_dir + file_name[:file_name.rfind('.')] + '_' + suffix + '.txt'

    os.makedirs(param_save_dir, exist_ok=True)

    np.savetxt(param_save_name, model_parameters.reshape((1, -1)), delimiter=', ',
               newline='\n', fmt='%1.10f')

    print('\nSaved model parameters in the following order:')
    for parameter_key in list(network.state_dict().keys()):
        print('   ', parameter_key, '| Dimension:', network.state_dict()[parameter_key].shape)


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

                export_to_txt(state_dict, network, file_path, file_name, 'policy')


if __name__ == '__main__':
    main()
