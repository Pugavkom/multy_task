import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from norse.torch import LIFAdExState
import seaborn as sns
from create_dataset_one_value import create_dataset_default_values
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
    batch_size = 1
    tasks_list, sorted_task_labels = create_dataset_default_values(batch_size)

    feature_size, output_size = tasks_list[0].feature_and_act_size
    clusters = np.load(cluster_path, allow_pickle=True)
    model = load_network(f, v_th, tau_a_inv, filter_parameter, hidden_size, feature_size, output_size, device=device,
                         save_states=True)
    init_state = LIFAdExState(
        torch.zeros(batch_size, hidden_size).to(device),
        torch.rand(batch_size, hidden_size).to(device),
        torch.zeros(batch_size, hidden_size).to(device),
        torch.zeros(batch_size, hidden_size).to(device),
    )
    inputs, t_outputs = tasks_list[0].dataset(1)
    inputs = torch.from_numpy(inputs).type(torch.float).to(device)
    outputs, states = model(inputs, init_state)
    outputs_dm = outputs.detach().cpu()
    s_dm = []
    for state in states:
        s_dm.append(state.z)
    s_dm = torch.stack(s_dm).detach().cpu()
    inputs, t_outputs = tasks_list[sorted_task_labels.index('RomoTask1')].dataset(1)
    inputs = torch.from_numpy(inputs).type(torch.float).to(device)
    outputs, states = model(inputs, init_state)
    outputs_romo = outputs.detach().cpu()
    s_romo = []
    for state in states:
        s_romo.append(state.z)
    s_romo = torch.stack(s_romo).detach().cpu()
    cluster_for_dm = 5  # 6 cluster
    cluster_for_romo = 11 # 11 cluster
    sns.heatmap(s_romo[:, 0, :].T)
    plt.plot([400] * 2, [0, hidden_size])
    plt.plot([1400] * 2, [0, hidden_size])
    plt.plot([2000] * 2, [0, hidden_size])
    np.save('romo_full', s_romo.numpy())
    #np.save('s_dm', s_dm.numpy())
    #np.save('s_romo', s_romo.numpy())
    # DM and GoRt


if __name__ == '__main__':
    main()
