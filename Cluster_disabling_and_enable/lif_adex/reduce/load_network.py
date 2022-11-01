import torch
from cgtasknet.net import SNNlifadex
from norse.torch import LIFAdExParameters


def load_network(f: str, v_th: float, tau_a_inv: float, filter_parameter: float, hidden_size: int, feature_size: int,
                 output_size: int, save_states: bool = False, device: torch.device = torch.device('cpu')):
    neuron_parameters = LIFAdExParameters(
        v_th=torch.as_tensor(v_th),
        tau_ada_inv=torch.as_tensor(tau_a_inv),
    )

    model = SNNlifadex(
        feature_size,
        hidden_size,
        output_size,
        neuron_parameters=neuron_parameters,
        tau_filter_inv=filter_parameter,
        save_states=save_states,
    )
    model.load_state_dict(
        torch.load(
            f,
            map_location=device,
        )
    )
    model.to(device)
    return model
