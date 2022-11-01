import os

import numpy as np
import torch
from cgtasknet.instruments.instrument_accuracy_network import CorrectAnswerNetwork
from norse.torch import LIFAdExState
from tqdm import tqdm

from create_dataset import create_dataset_default_values
from load_network import load_network

criterion = torch.nn.MSELoss()


def run_one_trial(task, model, init_state, device: torch.device):
    inputs, t_outputs = task.dataset(1)
    inputs = torch.from_numpy(inputs).type(torch.float).to(device)
    return model(inputs, init_state)[0], t_outputs, inputs


def full_tasks_list(tasks_list: list, model, batch_size: int, hidden_size: int, device: torch.device, can=None,
                    n_trials=1):
    mse_loses = []
    accuracy_list = []
    for task in tasks_list:
        answers = 0
        for _ in range(n_trials):
            init_state = LIFAdExState(
                torch.zeros(batch_size, hidden_size).to(device),
                torch.rand(batch_size, hidden_size).to(device),
                torch.zeros(batch_size, hidden_size).to(device),
                torch.zeros(batch_size, hidden_size).to(device),
            )
            outputs, t_outputs, inputs = run_one_trial(task, model, init_state, device)
            t_outputs = torch.from_numpy(t_outputs).type(torch.float).to(device)
            mse_loses.append(criterion(t_outputs, outputs).item())
            type_tasks = list(np.where(inputs[-1, :, 3:].detach().cpu().numpy() == 1)[1])
            answers += can.run(t_outputs[50:, :, 0].cpu(), outputs[50:, :, 0].cpu(), t_outputs[50:, :, 1:].cpu(),
                               outputs[50:, :, 1:].cpu(), type_tasks)

        answers /= batch_size * n_trials
        accuracy_list.append(answers)
    return mse_loses, accuracy_list


def main():
    N = 2999
    n_clusters = 14
    f = os.path.join('..', '..', '..',
                     r'models\low_freq\mean_fr_filter_less_v_th_0_45\weights\weights_100_N_256_without_square_2999_')
    cluster_path = os.path.join('..', '..', '..',
                                fr'models\low_freq\mean_fr_filter_less_v_th_0_45\{N}clusters{n_clusters}.npy')
    v_th = 0.45
    hidden_size = 256
    filter_parameter = 20
    tau_a_inv = 1 / 2
    device = torch.device('cuda')
    batch_size = 50
    n_trials = 10
    tasks_list, sorted_task_labels = create_dataset_default_values(batch_size)
    re_word = 'Go'
    choices_tasks = []
    values_tasks = []
    for i in range(len(sorted_task_labels)):
        if re_word in sorted_task_labels[i]:
            values_tasks.append(i)
        else:
            choices_tasks.append(i)
    can = CorrectAnswerNetwork(choices_tasks, values_tasks, 0.15)
    feature_size, output_size = tasks_list[0].feature_and_act_size
    clusters = np.load(cluster_path, allow_pickle=True)
    mse_loses_list = []
    accuracy_list = []
    print('Run...')
    model = load_network(f, v_th, tau_a_inv, filter_parameter, hidden_size, feature_size, output_size, device=device)
    mse_loses, accuracy = full_tasks_list(tasks_list, model, batch_size, hidden_size, device, can, n_trials)
    accuracy_list.append(accuracy)
    mse_loses_list.append(mse_loses)
    for cluster in tqdm(clusters):
        print(accuracy_list)
        model = load_network(f, v_th, tau_a_inv, filter_parameter, hidden_size, feature_size, output_size,
                             device=device)

        for name, param in model.named_parameters():
            if name == 'alif.recurrent_weights':
                rec_w = param
            if name == 'exp_f.linear.weight':
                out_w = param
        with torch.no_grad():
            rec_w[:, np.array(list(set(range(hidden_size)) - set(cluster)))] *= 0
            out_w[:, np.array(list(set(range(hidden_size)) - set(cluster)))] *= 0
        mse_loses, accuracy = full_tasks_list(tasks_list, model, batch_size, hidden_size, device, can, n_trials)

        mse_loses_list.append(mse_loses)
        accuracy_list.append(accuracy)
    for cluster in tqdm(clusters):
        model = load_network(f, v_th, tau_a_inv, filter_parameter, hidden_size, feature_size, output_size,
                             device=device)

        for name, param in model.named_parameters():
            if name == 'alif.recurrent_weights':
                rec_w = param
            if name == 'exp_f.linear.weight':
                out_w = param
        with torch.no_grad():
            rec_w[:, cluster] *= 0
            out_w[:, cluster] *= 0
        mse_loses, accuracy = full_tasks_list(tasks_list, model, batch_size, hidden_size, device, can, n_trials)
        mse_loses_list.append(mse_loses)
        accuracy_list.append(accuracy)
    print(mse_loses_list)
    np.save(f'{N}mse_loses{n_clusters}', mse_loses_list)
    np.save(f'{N}accuracy{n_clusters}', accuracy_list)
    # print(f'{mse_loses}')


if __name__ == '__main__':
    main()
