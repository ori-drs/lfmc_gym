import torch
import modules

import copy


def get_actor_critic_module_from_config(cfg, env, action_sampler, device):
    activation = {'tanh': torch.nn.Tanh, 'leaky_relu': torch.nn.LeakyReLU, 'softsign': torch.nn.Softsign}

    if cfg['module']['type'] == 'dense':
        observation_indices = copy.deepcopy(cfg['environment']['observation_indices'])

        if 'observation_history' in cfg['environment'].keys():
            if cfg['environment']['observation_history'] > 0:
                observation_indices['actor_input'][1] += cfg['environment']['observation_history'] * 24
                observation_indices['critic_input'][1] += cfg['environment']['observation_history'] * 24

        return modules.DenseActorCriticModule(
            obs_shape=env.num_obs,
            action_shape=env.num_acts,
            actor_hidden_layers=cfg['module']['actor']['hidden'],
            actor_activation=activation[cfg['module']['actor']['activation']],
            critic_hidden_layers=cfg['module']['critic']['hidden'],
            critic_activation=activation[cfg['module']['critic']['activation']],
            action_sampler=action_sampler,
            seed=cfg['seed'],
            num_envs=cfg['environment']['num_envs'],
            device=device,
            shuffle_batch=cfg['module']['properties']['dense']['shuffle_batch'],
            predict_values_during_act=cfg['module']['properties']['dense']['predict_values_during_act'],
            initial_action_std=cfg['module']['properties']['dense']['initial_action_std'],
            compute_jacobian=cfg['module']['properties']['dense']['compute_jacobian'],
            observation_indices=observation_indices,
            network_weights_gain=cfg['module']['properties']['dense']['network_weights_gain']
        )

    else:
        raise NotImplemented('Support for Recurrent and Dynamics Predictive Modules will be added in the near future.')
