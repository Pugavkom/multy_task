import numpy as np
from cgtasknet.tasks.reduce import CtxDMTaskParameters, DMTaskParameters, DMTaskRandomModParameters, GoDlTaskParameters, \
    GoDlTaskRandomModParameters, \
    GoRtTaskParameters, \
    GoRtTaskRandomModParameters, GoTaskParameters, GoTaskRandomModParameters, MultyReduceTasks, RomoTaskParameters, \
    RomoTaskRandomModParameters


def create_dataset_default_values(batch_size, go_task_list_values=np.linspace(0, 1, 8), ):
    romo_parameters = RomoTaskRandomModParameters(
        romo=RomoTaskParameters(
            delay=0.2,
            positive_shift_delay_time=1.5,
            trial_time=0.2,
            positive_shift_trial_time=0.4,
            answer_time=.25
        ),
    )
    dm_parameters = DMTaskRandomModParameters(
        dm=DMTaskParameters(trial_time=0.3, positive_shift_trial_time=1.5, answer_time=.25)
    )
    ctx_parameters = CtxDMTaskParameters(dm=dm_parameters.dm)
    go_parameters = GoTaskRandomModParameters(
        go=GoTaskParameters(
            trial_time=0.3,
            positive_shift_trial_time=1.5,
            value=go_task_list_values,
            answer_time=.25
        )
    )
    gort_parameters = GoRtTaskRandomModParameters(
        go_rt=GoRtTaskParameters(
            trial_time=0.3,
            positive_shift_trial_time=1.5,
            answer_time=1,
            value=go_task_list_values,
        )
    )
    godl_parameters = GoDlTaskRandomModParameters(
        go_dl=GoDlTaskParameters(
            go=GoTaskParameters(trial_time=0.2, positive_shift_trial_time=0.4, answer_time=.25,
                                value=go_task_list_values),
            delay=0.2,
            positive_shift_delay_time=1.5,

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
    return [MultyReduceTasks(
        tasks=task_dict,
        batch_size=batch_size,
        delay_between=0,
        enable_fixation_delay=True,
        task_number=i,
    ) for i in range(len(tasks))], sorted(tasks)
