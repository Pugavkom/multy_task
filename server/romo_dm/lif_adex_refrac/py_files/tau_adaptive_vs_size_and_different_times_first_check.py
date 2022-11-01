import os
import sys

import numpy as np
import torch
from cgtasknet.instruments.instrument_accuracy_network import correct_answer
from cgtasknet.net import LIFAdExRefracInitState
from cgtasknet.net.lifadexrefrac import SNNlifadexrefrac
from cgtasknet.tasks.reduce import DMTaskParameters, DMTaskRandomModParameters, MultyReduceTasks, RomoTaskParameters, \
    RomoTaskRandomModParameters
from norse.torch.functional.lif_adex import LIFAdExParameters
from norse.torch.functional.lif_adex_refrac import LIFAdExRefracParameters
from torch import nn
from tqdm import tqdm

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f'Device: {("gpu (cuda)" if device.type == "cuda" else "cpu")}')


def name_file_for_save_model(N: int, tau_ada_inv: float, lr: float, time: int):
    return f"data/N_{N}_taua_{tau_ada_inv}_lr_{lr}_epochs_{time}"


def create_task(
        batch_size: int = 50,
        delay_romo: float = 0.1,
        trial_time_romo=0.1,
        positive_shift_delay_time_romo: float = 0.5,
        positive_shift_trial_time_romo: float = 0.2,
        trial_time_dm: float = 0.1,
        positive_shift_trial_time_dm=0.6,
):
    romo_parameters = RomoTaskRandomModParameters(
        romo=RomoTaskParameters(
            delay=delay_romo,
            positive_shift_delay_time=positive_shift_delay_time_romo,
            trial_time=trial_time_romo,
            positive_shift_trial_time=positive_shift_trial_time_romo,
        ),
        n_mods=1,
    )
    dm_parameters = DMTaskRandomModParameters(
        dm=DMTaskParameters(
            trial_time=trial_time_dm,
            positive_shift_trial_time=positive_shift_trial_time_dm,
        ),
        n_mods=1,
    )

    task_names = ["RomoTask1", "DMTask1"]
    tasks = dict()
    tasks[task_names[0]] = romo_parameters
    tasks[task_names[1]] = dm_parameters
    return MultyReduceTasks(
        tasks=tasks,
        batch_size=batch_size,
        enable_fixation_delay=True,
        sequence_bathces=True,
        number_of_inputs=1,
    )




test_sequence = 100
def test_loop(
        task: MultyReduceTasks,
        hidden_size: int,
        tau_ada_inv: float,
        lr: float,
        epoch: int,
        all_epochs: int,
        save_every_epoch: int,
        number_of_tasks: int = 1,
):
    batch_size = task.batch_size
    init_state = LIFAdExRefracInitState(batch_size, hidden_size, device=device)
    feature_size, output_size = task.feature_and_act_size
    neuron_parameters = LIFAdExRefracParameters(
        LIFAdExParameters(
            v_th=torch.as_tensor(0.65),
            tau_ada_inv=tau_ada_inv,
            alpha=100,
            method="super",
        ),
        rho_reset=1,
    )
    model = SNNlifadexrefrac(
        feature_size,
        hidden_size,
        output_size,
        neuron_parameters=neuron_parameters,
        tau_filter_inv=500,
    ).to(device)
    # model = torch.nn.DataParallel(model, device_ids=[1, 2], dim=1).to(device)
    running_loss = 0
    model.load_state_dict(torch.load(name_file_for_save_model(hidden_size, tau_ada_inv, lr, epoch)))
    result = 0
    for _ in tqdm(range(test_sequence)):
        inputs, target_outputs = task.dataset(1, delay_between=0)
        inputs += np.random.normal(0, 0.01, size=inputs.shape)
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        target_outputs = torch.from_numpy(target_outputs).type(torch.float)
        outputs = model(inputs)[0]
        answers = correct_answer(
            outputs[:, :, 1:], target_outputs[:, :, 1:], target_outputs[:, :, 0]
        )
        result += torch.sum(answers).item()
    return round(result / batch_size / test_sequence * 100, 5)




def main(N_start, N_stop):
    if N_start == 2:
        N_array = [*range(0, N_stop, 80)]
        N_array[0] = 2
    else: 
        N_array = [*range(N_start, N_stop, 80)]
    lr_array = [1e-3]
    tau_ada_inv_array = [*np.arange(0.5, 7, 1)]
    t_array = [*range(39, 2500, 200)]
    print(f"N: {N_array}")
    print(f"learning rates: {lr_array}")
    print(f"tau_ada_inv: {tau_ada_inv_array}")
    print(f"all iterations: {len(N_array) * len(lr_array) * len(tau_ada_inv_array) * len(t_array)}")
    data_dir = "data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    task = create_task()
    for N in N_array:
        for lr in lr_array:
            for ts in t_array:
                for tau_ada_inv in tau_ada_inv_array:
                    result = test_loop(task, N, tau_ada_inv, lr, epoch=ts, all_epochs=2500, save_every_epoch=40)
                    with open(f'check_params_lr_{lr}', 'a') as f:
                        f.write(f'N_{N}:epoch_{ts}:tau_ada_inv_{tau_ada_inv}:result_{result}')
                        f.write('\n')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        N_start = int(sys.argv[1])
        N_stop = int(sys.argv[2])
        print(f'{N_start} : {N_stop}')
        main(N_start, N_stop)
    else:
        print('You need set up N_start and N_stop')
