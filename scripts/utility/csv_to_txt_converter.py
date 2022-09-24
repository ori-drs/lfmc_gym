import os
import numpy as np

from common.paths import ProjectPaths


ENV_NAME = 'anymal_velocity_command'
PARAMETERS_DIR = '2022-09-06-11-05-45'


def csv_to_csv_c(filename_in, filename_out):
    parameters = np.loadtxt(filename_in).reshape((-1, 1)).transpose()
    np.savetxt(filename_out, parameters, delimiter=',')


def main():
    paths = ProjectPaths()
    parameters_path_dir = paths.DATA_PATH + '/' + ENV_NAME + '/' + PARAMETERS_DIR

    state_mean_paths = [f.path for f in os.scandir(parameters_path_dir) if
                        f.is_file() and f.path.endswith('.csv') and 'mean' in f.path]

    if len(state_mean_paths) > 0:
        parameters_export_dir = parameters_path_dir + '/exported_parameters/state_mean'
        os.makedirs(parameters_export_dir, exist_ok=True)

        for mean_parameter_path in state_mean_paths:
            save_file_name = mean_parameter_path[mean_parameter_path.rfind('/') + 1:].replace('csv', 'txt')
            save_file_path = parameters_export_dir + '/' + save_file_name

            csv_to_csv_c(mean_parameter_path, save_file_path)

    state_var_paths = [f.path for f in os.scandir(parameters_path_dir) if
                       f.is_file() and f.path.endswith('.csv') and 'var' in f.path]

    if len(state_var_paths) > 0:
        parameters_export_dir = parameters_path_dir + '/exported_parameters/state_var'
        os.makedirs(parameters_export_dir, exist_ok=True)

        for var_parameter_path in state_var_paths:
            save_file_name = var_parameter_path[var_parameter_path.rfind('/') + 1:].replace('csv', 'txt')
            save_file_path = parameters_export_dir + '/' + save_file_name

            csv_to_csv_c(var_parameter_path, save_file_path)

    print('Converted all files')


if __name__ == '__main__':
    main()
