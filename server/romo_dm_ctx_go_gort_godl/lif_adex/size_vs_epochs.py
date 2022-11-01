import os

import click
import numpy as np
import torch
from cgtasknet.instruments.instrument_accuracy_network import CorrectAnswerNetwork
from cgtasknet.net import SNNlifadex
from cgtasknet.tasks.reduce import CtxDMTaskParameters, DMTaskParameters, DMTaskRandomModParameters, \
    GoDlTaskParameters, GoDlTaskRandomModParameters, GoRtTaskParameters, GoRtTaskRandomModParameters, GoTaskParameters, \
    GoTaskRandomModParameters, MultyReduceTasks, RomoTaskParameters, \
    RomoTaskRandomModParameters
from dotenv import dotenv_values
from norse.torch import LIFAdExParameters, LIFAdExState
from tqdm import tqdm


class Error(Exception):
    """

    """


class NetworkPathIsNotSetup(Error):
    """

    """


class DeviceError(Error):
    """

    """


def config_loader(config: str) -> dict:
    conf_file = dotenv_values(config)
    conf_file_out = {}
    for key, item in conf_file.items():
        conf_file_out[key.lower()] = int(item) if key == 'N' else float(item)
    return conf_file_out


def generate_dataset(config_dataset: dict):
    for key, item in config_dataset.items():
        print(f'{key} = {item}')
    batch_size = int(config_dataset['batch_size'])
    go_task_list_values = np.linspace(0, 1, 8)
    romo_parameters = RomoTaskRandomModParameters(
        romo=RomoTaskParameters(
            delay=config_dataset['romo_delay'],
            positive_shift_delay_time=config_dataset['romo_positive_shift_delay'],
            trial_time=config_dataset['romo_trial_time'],
            positive_shift_trial_time=config_dataset['romo_positive_shift_trial_time'],
            answer_time=config_dataset['romo_answer_time']
        ),
    )
    dm_parameters = DMTaskRandomModParameters(
        dm=DMTaskParameters(trial_time=config_dataset['dm_trial_time'],
                            positive_shift_trial_time=config_dataset['dm_positive_shift_trial_time'],
                            answer_time=config_dataset['dm_answer_time'])
    )
    ctx_parameters = CtxDMTaskParameters(dm=dm_parameters.dm)
    go_parameters = GoTaskRandomModParameters(
        go=GoTaskParameters(
            trial_time=config_dataset['go_trial_time'],
            positive_shift_trial_time=config_dataset['go_positive_shift_trial_time'],
            value=go_task_list_values,
            answer_time=config_dataset['go_answer_time'],
        )
    )
    gort_parameters = GoRtTaskRandomModParameters(
        go_rt=GoRtTaskParameters(
            trial_time=config_dataset['gort_trial_time'],
            positive_shift_trial_time=config_dataset['gort_positive_shift_trial_time'],
            answer_time=config_dataset['gort_answer_time'],
            value=go_task_list_values,
        )
    )
    godl_parameters = GoDlTaskRandomModParameters(
        go_dl=GoDlTaskParameters(
            go=GoTaskParameters(trial_time=config_dataset['godl_trial_time'],
                                positive_shift_trial_time=config_dataset['godl_positive_shift_trial_time'],
                                answer_time=config_dataset['godl_answer_time'],
                                value=go_task_list_values),
            delay=config_dataset['godl_delay'],
            positive_shift_delay_time=config_dataset['godl_positive_shift_delay'],

        )
    )
    tasks = [
        "RomoTask1",
        "RomoTask2",
        "DMTask1",
        "DMTask2",
        "CtxDMTask1",
        "CtxDMTask2",
        "GoTask1",
        "GoTask2",
        "GoRtTask1",
        "GoRtTask2",
        "GoDlTask1",
        "GoDlTask2",
    ]
    task_dict = {
        tasks[0]: romo_parameters,
        tasks[1]: romo_parameters,
        tasks[2]: dm_parameters,
        tasks[3]: dm_parameters,
        tasks[4]: ctx_parameters,
        tasks[5]: ctx_parameters,
        tasks[6]: go_parameters,
        tasks[7]: go_parameters,
        tasks[8]: gort_parameters,
        tasks[9]: gort_parameters,
        tasks[10]: godl_parameters,
        tasks[11]: godl_parameters,
    }
    task = MultyReduceTasks(
        tasks=task_dict,
        batch_size=batch_size,
        delay_between=0,
        enable_fixation_delay=True,
        mode="random",
    )
    sorted_tasks = sorted(tasks)
    re_word = 'Go'
    choices_tasks = []
    values_tasks = []
    for i in range(len(sorted_tasks)):
        if re_word in sorted_tasks[i]:
            values_tasks.append(i)
        else:
            choices_tasks.append(i)
    can = CorrectAnswerNetwork(choices_tasks, values_tasks, config_dataset['error_go'])
    return task, can


def network_loader_adex(network_params: dict, device, sizes: dict):
    feature_size = sizes['feature_size']
    output_size = sizes['output_size']
    hidden_size = sizes['hidden_size']
    if 'method' not in network_params:
        method = 'super'
    else:
        method = network_params['method']
    if 'alpha' not in network_params:
        alpha = 100
    else:
        alpha = network_params['alpha']
    neuron_parameters = LIFAdExParameters(
        v_th=torch.as_tensor(network_params['v_th']),
        tau_ada_inv=torch.as_tensor(network_params['tau_ada_inv']),
        alpha=alpha,
        method=method,
    )

    model = SNNlifadex(
        feature_size,
        hidden_size,
        output_size,
        neuron_parameters=neuron_parameters,
        tau_filter_inv=network_params['tau_filter_inv'],
    ).to(device)
    return model


class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        result = torch.sum(diff2) / torch.sum(mask)
        return result


def every_bath_generator(
        start_sigma: float,
        stop_sigma: float,
        times: int = 1,
        batches: int = 1,
        actions: int = 1,
):
    data = np.zeros((times, batches, actions))
    for i in range(batches):
        data[:, i, :] = np.random.normal(
            0, np.random.uniform(start_sigma, stop_sigma), size=(times, actions)
        )
    return data


def forward(self, input, target, mask):
    diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
    result = torch.sum(diff2) / torch.sum(mask)
    return result


def state(batch_size: int, hidden_size: int, device: torch.device):
    return LIFAdExState(
        torch.zeros(batch_size, hidden_size).to(device),
        torch.rand(batch_size, hidden_size).to(device),
        torch.zeros(batch_size, hidden_size).to(device),
        torch.rand(batch_size, hidden_size).to(device) * 0,
    )


def run_loop(model, tasks: MultyReduceTasks, config_train: dict, epochs: int, save_model_every: int, save_data: str,
             pbar, sizes: dict, device: torch.device):
    batch_size = int(tasks.batch_size)
    hidden_size = int(sizes['hidden_size'])

    criterion = MaskedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config_train['lr'])
    sigma = config_train['sigma']
    for j in tqdm(range(epochs), colour='blue', ncols=50):
        inputs, target_outputs = tasks.dataset()
        inputs[:, :, 1:3] += every_bath_generator(sigma, sigma, inputs.shape[0], inputs.shape[1], 2)
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        target_outputs = torch.from_numpy(target_outputs).type(torch.float).to(device)
        loss_mask = torch.zeros_like(target_outputs)

        mask_indexes_signes = torch.where(target_outputs[:, :, 0] == 1)
        mask_indexes_zeros = torch.where(target_outputs[:, :, 0] == 0)
        loss_mask[mask_indexes_signes[0], mask_indexes_signes[1], :] = 1
        loss_mask[mask_indexes_zeros[0], mask_indexes_zeros[1], :] = 5
        optimizer.zero_grad()
        init_state = state(batch_size, hidden_size, device)
        outputs, _ = model(inputs, init_state)
        loss = criterion(outputs, target_outputs, loss_mask)
        loss.backward()
        optimizer.step()
        if j % save_model_every == save_model_every - 1:
            with torch.no_grad():
                torch.save(
                    model.state_dict(),
                    os.path.join(save_data, f'epoch_{j}_size_{hidden_size}')
                )
        try:
            del inputs
        except:
            pass

        try:
            del outputs
        except:
            pass

        try:
            del target_outputs
        except:
            pass
        if 'cuda' in device.type:
            torch.cuda.empty_cache()
        pbar.update(1)


def test_network(model, tasks: MultyReduceTasks, device, sizes, config_test: dict,
                 can: CorrectAnswerNetwork, pbar):
    hidden_size = sizes['hidden_size']
    batch_size = tasks.batch_size
    number_of_test_trials = int(config_test['number_of_test_trials'])
    result = 0
    for _ in tqdm(range(number_of_test_trials), ncols=50):
        inputs, target_outputs = tasks.dataset(1, delay_between=0)
        inputs[:, :, 1:3] += every_bath_generator(0.05, 0.05, inputs.shape[0], inputs.shape[1], 2)
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        outputs, _ = model(inputs, state(batch_size, hidden_size, device))
        outputs = outputs.detach().cpu()
        target_outputs = torch.from_numpy(target_outputs).type(torch.float)
        type_tasks = list(np.where(inputs[-1, :, 3:].detach().cpu().numpy() == 1)[1])
        answers = can.run(target_outputs[50:, :, 0], outputs[50:, :, 0],
                          target_outputs[50:, :, 1:], outputs[50:, :, 1:], type_tasks)
        result += answers
        pbar.update(1)
        try:
            del inputs
        except:
            pass

        if 'cuda' in device.type:
            torch.cuda.empty_cache()

    return result / number_of_test_trials / batch_size


@click.command()
@click.option('sizes', '-s', help='List or start, stop, step')
@click.option('epochs', '-e', type=int, default=10, help='Number of epochs')
@click.option('--save_model_every', type=int, default=10, help='Save after every epochs')
@click.option('network_path', '-p', help='Network path (save_model)')
@click.option('--save_data', default='data', help='Save directory')
@click.option('--network_parameters', default='config', help='Network file parameters')
@click.option('--device', default='cpu', help='Your device: cuda | cuda0 | cuda1 | cpu | ...')
@click.option('--dataset_config', default='config_dataset', help='Dataset config path')
@click.option('--train_config', default='config_train', help='Train config path')
@click.option('--type_run', default='train', help='type running: train | test')
@click.option('--test_config', default='config_test', help='Test config parameters')
def main(sizes: str, epochs: int, save_model_every: int, network_path: str, save_data: str, network_parameters: str,
         dataset_config: str, test_config: str, device: str, train_config: str, type_run):
    try:
        device = torch.device(device)
    except DeviceError:
        raise
    if sizes.count('[') == 1 and sizes.count(']') == 1:
        sizes = sizes.replace('[', '').replace(']', '').replace(',', ' ')
        network_sizes = [int(el) for el in sizes.split()]
    else:
        size_start, size_stop, size_step = [int(el) for el in sizes.split()]
        if size_stop - size_start <= 0:
            raise ValueError
        if size_stop - size_start < size_step:
            raise ValueError
        network_sizes = [*range(size_start, size_stop, size_step)]
    if save_model_every > epochs:
        raise ValueError

    if not os.path.exists(save_data):
        if save_data == 'data':
            os.mkdir(save_data)
        else:
            raise ValueError  # TODO: Change to custom error (exception)

    if network_path is None:
        raise NetworkPathIsNotSetup

    tasks, can = generate_dataset(config_loader(dataset_config))
    feature_size, output_size = tasks.feature_and_act_size
    sizes = dict([('feature_size', feature_size), ('output_size', output_size)])
    network_parameters = config_loader(network_parameters)

    with tqdm(total=len(network_sizes) * int(epochs if type_run == 'train' else
                                             (config_loader(test_config)['stop_epoch'] - config_loader(test_config)[
                                                 'start_epoch']) * config_loader(test_config)['number_of_test_trials']/ config_loader(test_config)[
                                                 'step_epoch']), colour='yellow', ncols=50) as pbar:

        for i in tqdm(range(len(network_sizes)), colour='red', ncols=50):
            sizes['hidden_size'] = network_sizes[i]
            model = network_loader_adex(network_params=network_parameters, device=device, sizes=sizes)
            if type_run == 'train':
                run_loop(model, tasks, config_loader(train_config), epochs, save_model_every, save_data, pbar, sizes,
                         device)
            elif type_run == 'test':
                conf = config_loader(test_config)
                start_epoch = int(conf['start_epoch'])
                stop_epoch = int(conf['stop_epoch'])
                step_epoch = int(conf['step_epoch'])
                for epoch in tqdm(range(start_epoch, stop_epoch, step_epoch), colour='yellow', ncols=50):
                    model.load_state_dict(
                        torch.load(
                            os.path.join(save_data, f'epoch_{epoch}_size_{sizes["hidden_size"]}')
                        )
                    )
                    result = test_network(model, tasks, device, sizes, config_loader(test_config), can, pbar)
                    with open(os.path.join(save_data, 'accuracy.txt'), 'a') as f:
                        f.write(f'epoch_{epoch}_size_{sizes["hidden_size"]}:{result}\n')


if __name__ == '__main__':
    main()
