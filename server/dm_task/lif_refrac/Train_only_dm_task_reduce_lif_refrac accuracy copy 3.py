import os

import matplotlib.pyplot as plt  # for analys
import numpy as np
import torch
import torch.nn as nn
from cgtasknet.instruments.instrument_accuracy_network import correct_answer
from cgtasknet.net.lifrefrac import SNNLifRefrac
from cgtasknet.tasks.reduce import DMTask
from norse.torch.functional.lif import LIFParameters
from norse.torch.functional.lif_refrac import LIFRefracParameters
from tqdm import tqdm

# from norse.torch import LIF

# %% [markdown]
# ## Step -1: Create dataset

# %%
device = torch.device("cuda:0")
# device = torch.device('cpu')
print(f'Device: {("gpu (cuda)" if device.type=="cuda" else "cpu")}')

# %%
batch_size = 400
number_of_tasks = 1
# tasks = dict(task_list)
Task = DMTask(batch_size=batch_size)
Task_test = DMTask(batch_size=1)

# %% [markdown]
# ## Step 1.1: Create model

# %%
feature_size, output_size = Task.feature_and_act_size
hidden_size = 400

neuron_parameters = LIFRefracParameters(
    LIFParameters(
        tau_mem_inv=torch.as_tensor(1 / 0.01).to(device),
        alpha=torch.as_tensor(100),
        method="super",
        v_th=torch.as_tensor(0.65).to(device),
    ),
    rho_reset=torch.as_tensor(1),
)


# ## Step 2: loss and creterion

# %%
learning_rate = 1e-3


def test_lopp(
    all_number_of_times: int = 1,
    learning_rate: float = 1e-3,
    hidden_size: int = 1,
    learning_time: int = 0,
    path_to_data: str = "./",
):
    model = SNNLifRefrac(
        feature_size,
        hidden_size,
        output_size,
        neuron_parameters=neuron_parameters,
        tau_filter_inv=500,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # %% [markdown]
    # ## Step 3: Train loop
    model.load_state_dict(
        torch.load(
            f"{path_to_data}_lr_{learning_rate}_N_{hidden_size}_number_{learning_time}"
        )
    )
    model.to(device)
    result = 0
    for _ in range(all_number_of_times):
        inputs, target_outputs = Task.dataset(number_of_tasks)
        inputs += np.random.normal(0, 0.01, size=(inputs.shape))
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        target_outputs = torch.from_numpy(target_outputs).type(torch.float).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)[0]
        answers = correct_answer(
            outputs[:, :, 1:], target_outputs[:, :, 1:], target_outputs[:, :, 0]
        )
        result += torch.sum(answers).item()
    return result / batch_size / all_number_of_times


interval_train = 100
size_interval = [0.1]
lr_interval = [10 ** i * 1e-4 for i in range(0, 4)]
time_interval = np.linspace(199, 1990, 10).astype(int)

if os.name == "nt":
    data_directory = "server\\dm_task\\lif_refrac\\data\\"
    interval_directory = data_directory + "intervals\\"
    data = data_directory + "data\\"
else:
    data_directory = "data/"
    interval_directory = data_directory + "intervals/"
    data = data_directory + "data/"


for lr in tqdm(lr_interval):
    with open(f"{data_directory}lr_{lr}", "w") as f:
        pass
    for size in size_interval:
        for time in time_interval:
            fraction_answer = test_lopp(interval_train, lr, size, time, data)
            with open(f"{data_directory}lr_{lr}", "a") as f:
                f.write(f"{fraction_answer} ")
        with open(f"{data_directory}lr_{lr}", "a") as f:
            f.write("\n")


# %%
