import os
from typing import Iterable, List

import numpy as np
import torch
from cgtasknet.tasks.reduce import (
    CtxDMTaskParameters,
    DMTaskParameters,
    DMTaskRandomModParameters,
    GoDlTaskParameters,
    GoDlTaskRandomModParameters,
    GoRtTaskParameters,
    GoRtTaskRandomModParameters,
    GoTaskParameters,
    GoTaskRandomModParameters,
    MultyReduceTasks,
    RomoTaskParameters,
    RomoTaskRandomModParameters,
)
from dPCA import dPCA
from norse.torch import LIFAdExState
from tqdm import tqdm

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

go_task_list_values = np.linspace(0, 1, 8)


def generate_task_romo_dm_ctx_go_gort_godl(
    values: Iterable[float], romo_number_fix: int = 0, romo_fix: float = 0.5
) -> List[List]:
    """
    Функция, генерирующая набор задач для тестирования.
    :param tasks:
    :param values:
    :param romo_number_fix:
    :param romo_fix:
    :return:
    """

    return_tasks = [[] for _ in range(len(tasks))]

    for value in go_task_list_values:
        batch_size = 1
        romo_parameters = RomoTaskRandomModParameters(
            romo=RomoTaskParameters(
                delay=1.0,
                # positive_shift_delay_time=1.4,
                trial_time=0.5,
                # positive_shift_trial_time=0.2,
                value=(romo_fix, value) if romo_number_fix == 0 else (value, romo_fix),
            ),
        )
        dm_parameters = DMTaskRandomModParameters(
            dm=DMTaskParameters(trial_time=1.0, value=value)
        )
        ctx_parameters = CtxDMTaskParameters(dm=dm_parameters.dm, value=(value, value))
        go_parameters = GoTaskRandomModParameters(
            go=GoTaskParameters(
                trial_time=1.0,
                # positive_shift_trial_time=0.8,
                value=value,
            )
        )
        gort_parameters = GoRtTaskRandomModParameters(
            go_rt=GoRtTaskParameters(
                trial_time=0.6,
                # positive_shift_trial_time=0.8,
                answer_time=1.5,
                value=value,
            )
        )
        godl_parameters = GoDlTaskRandomModParameters(
            go_dl=GoDlTaskParameters(
                go=GoTaskParameters(trial_time=0.3, value=value),
                delay=1.0,
                # positive_shift_delay_time=1.4,
            )
        )

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

        tasks_sequence = [
            MultyReduceTasks(
                tasks=task_dict,
                batch_size=batch_size,
                delay_between=0,
                enable_fixation_delay=True,
                task_number=i,
                mode="value",
            )
            for i in range(len(tasks))
        ]
        for i in range(len(tasks_sequence)):
            return_tasks[i].append(tasks_sequence[i])
    return return_tasks


def dpca_run(
    network,
    tasks,
    device,
    number_of_trials,
    hidden_size,
):
    actR_v_out = []
    actR_a_out = []
    for type_task in tqdm(tasks):
        actR_v = None
        actR_a = None
        value = 0
        for task_number in type_task:
            for trial in range(number_of_trials):
                data, _ = task_number.dataset()
                data = torch.from_numpy(data).type(torch.float).to(device)
                init_state = LIFAdExState(
                    torch.zeros(1, hidden_size).to(device),
                    torch.rand(1, hidden_size).to(device),
                    torch.zeros(1, hidden_size).to(device),
                    torch.zeros(1, hidden_size).to(device),
                )
                out, states, _ = network(data, init_state)
                a, v = [], []
                for j in range(len(states)):
                    v.append(states[j].v)
                    a.append(states[j].a)
                a = torch.stack(a).detach()
                v = torch.stack(v).detach()
                if actR_v is None and actR_a is None:
                    actR_v = np.zeros(
                        (
                            number_of_trials,
                            hidden_size,
                            len(tasks[0]),
                            len(data)
                            # int((dmparams.answer_time + dmparams.trial_time) / dmparams.dt),
                        ),
                        dtype=np.float32,
                    )

                    actR_a = np.zeros_like(actR_v)
                actR_v[trial, :, value, :] = v[:, 0, :].T.cpu().numpy()
                actR_a[trial, :, value, :] = a[:, 0, :].T.cpu().numpy()
            value += 1
        actR_v_out.append(actR_v)
        actR_a_out.append(actR_a)
    return actR_v_out, actR_a_out


def dpca_calculate(actRs):
    result = []
    for actR in actRs:
        R = np.mean(actR, 0)
        R -= np.mean(R.reshape((actR.shape[1], -1)), 1)[:, None, None]
        dpca = dPCA.dPCA(labels="st", regularizer="auto")
        dpca.protect = ["t"]
        dpca.opt_regularizer_flag = True
        Z = dpca.fit_transform(R, actR)
        result.append(Z)
    return result


def main():
    print("*" * 20, "Start", "*" * 20)
    values = np.arange(0, 1.1, 0.1)
    check_tasks = generate_task_romo_dm_ctx_go_gort_godl(values)
    from cgtasknet.net import SNNlifadex
    from norse.torch import LIFAdExParameters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    feature_size, output_size = check_tasks[0][0].feature_and_act_size
    hidden_size = 256
    tau_a = "1_2"
    neuron_parameters = LIFAdExParameters(
        v_th=torch.as_tensor(0.45),
        tau_ada_inv=eval(tau_a.replace("_", "/")),
        alpha=100,
        method="super",
        # rho_reset = torch.as_tensor(5)
    )
    model = SNNlifadex(
        feature_size,
        hidden_size,
        output_size,
        neuron_parameters=neuron_parameters,
        tau_filter_inv=20,
        save_states=True,
        return_spiking=True,
    ).to(device)
    model.load_state_dict(
        torch.load(
            (
                r"A:\src\multy_task\models\low_freq\mean_fr_filter_less_v_th_0_45\weights\weights_100_N_256_without_square_2999_"
            ),
            map_location=device,
        )
    )

    actR_v, acrR_a = dpca_run(model, check_tasks, device, 30, hidden_size)
    z_v_list = dpca_calculate(actR_v)
    z_a_list = dpca_calculate(acrR_a)

    task_names = []
    for key in check_tasks[0][0]._task_list:
        task_names.append(key.name)

    for i in range(len(z_v_list)):
        for key in z_v_list[i]:
            np.save(
                f"data{os.sep}Z_v_{key}_{sorted(tasks)[i]}_{hidden_size}_{tau_a}",
                z_v_list[i][key],
            )
            np.save(
                f"data{os.sep}Z_a_{key}_{sorted(tasks)[i]}_{hidden_size}_{tau_a}",
                z_a_list[i][key],
            )


if __name__ == main():
    main()
