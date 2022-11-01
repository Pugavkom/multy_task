import os

import matplotlib.pyplot as plt  # for analys
import numpy as np
import torch
import torch.nn as nn
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
batch_size = 100
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


def train_lopp(
    number_of_times_save: int = 1,
    all_number_of_times: int = 1,
    learning_rate: float = 1e-3,
    hidden_size: int = 1,
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

    for i in range(all_number_of_times):
        inputs, target_outputs = Task.dataset(number_of_tasks)
        inputs += np.random.normal(0, 0.01, size=(inputs.shape))
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        target_outputs = torch.from_numpy(target_outputs).type(torch.float).to(device)
        optimizer.zero_grad()
        outputs, states = model(inputs)
        loss = criterion(outputs, target_outputs)
        loss.backward()
        optimizer.step()
        if i % (number_of_times_save - 1) == 0 and i > 0:
            torch.save(
                model.state_dict(),
                f"{path_to_data}_lr_{learning_rate}_N_{hidden_size}_number_{i}",
            )


interval_train = 2000
number_of_times_save = 200
size_interval = [i for i in range(1, 620, 40)]
lr_interval = [10 ** i * 1e-4 for i in range(0, 4)]


if os.name == "nt":
    data_directory = "data\\"
    interval_directory = data_directory + "intervals\\"
    data = data_directory + "data\\"
else:
    data_directory = "data/"
    interval_directory = data_directory + "intervals/"
    data = data_directory + "data/"


if not os.path.exists(data_directory):
    os.mkdir(data_directory)
if not os.path.exists(interval_directory):
    os.mkdir(interval_directory)
if not os.path.exists(data):
    os.mkdir(data)
np.savetxt(interval_directory + "interval" + "_size", size_interval)
np.savetxt(interval_directory + "interval" + "_lr", lr_interval)


for lr in tqdm(lr_interval):
    for size in size_interval:
        train_lopp(number_of_times_save, interval_train, lr, size, data)


# %%
