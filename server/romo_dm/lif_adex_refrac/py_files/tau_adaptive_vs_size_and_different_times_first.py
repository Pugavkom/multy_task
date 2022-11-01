import os

import numpy as np
import torch
from cgtasknet.net import LIFAdExRefracInitState
from cgtasknet.net.lifadexrefrac import SNNlifadexrefrac
from cgtasknet.tasks.reduce import DMTaskParameters
from cgtasknet.tasks.reduce import DMTaskRandomModParameters
from cgtasknet.tasks.reduce import MultyReduceTasks
from cgtasknet.tasks.reduce import RomoTaskParameters
from cgtasknet.tasks.reduce import RomoTaskRandomModParameters
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


def train_loop(
    task: MultyReduceTasks,
    hidden_size: int,
    tau_ada_inv: float,
    lr: float,
    all_epochs: int,
    save_every_epoch: int,
    number_of_tasks: int = 1,
):

    batch_size = task.batch_size
    l_inputs = []
    l_outputs = []
    for _ in tqdm(range(all_epochs)):
        tmp_inputs, tmp_target_outputs = task.dataset(number_of_tasks, delay_between=0)
        tmp_inputs += np.random.normal(0, 0.01, size=tmp_inputs.shape)
        l_inputs.append(tmp_inputs)
        l_outputs.append(tmp_target_outputs)
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
    #model = torch.nn.DataParallel(model, device_ids=[1, 2], dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    running_loss = 0
    for i in tqdm(range(all_epochs)):
        inputs = torch.from_numpy(l_inputs[i]).type(torch.float).to(device)
        target_outputs = torch.from_numpy(l_outputs[i]).type(torch.float).to(device)
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, _ = model(inputs, init_state.random_state())
        #outputs, _ = model(inputs)

        loss = criterion(outputs, target_outputs)
        loss.backward()
        optimizer.step()

        # save model
        running_loss += loss.item()
        if i % save_every_epoch == save_every_epoch - 1:
            with open("log_multy.txt", "a") as f:
                f.write("epoch: {:d} loss: {:0.5f}\n".format(i + 1, running_loss / 10))
            running_loss = 0.0
            with torch.no_grad():
                torch.save(
                    model.state_dict(),
                    name_file_for_save_model(hidden_size, tau_ada_inv, lr, i),
                )


N_array = [*range(0, 680, 80)]
N_array[0] = 2
lr_array = [1e-3]
tau_ada_inv_array = [*np.arange(0.5, 7, 1)]
print(f"N: {N_array}")
print(f"learning rates: {lr_array}")
print(f"tau_ada_inv: {tau_ada_inv_array}")
print(f"all iterations: {len(N_array) * len(lr_array) * len(tau_ada_inv_array)}")
data_dir = "data/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

task = create_task()

for lr in lr_array:
    for N in N_array:
        for tau_ada_inv in tau_ada_inv_array:
            train_loop(task, N, tau_ada_inv, lr, all_epochs=2500, save_every_epoch=40)
            with open(f'params_lr{lr}', 'a') as f:
                f.write(name_file_for_save_model(N, tau_ada_inv, lr, 0))
                f.write('\n')
