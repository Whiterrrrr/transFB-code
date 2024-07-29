from agents.cexp.module import weight_init
from copy import deepcopy
import torch
from collections import defaultdict
import torch.nn as nn


def perturb(net, optimizer, perturb_factor):
    linear_keys = [
        name for name, mod in net.named_modules()
        if isinstance(mod, torch.nn.Linear)
    ]
    new_net = deepcopy(net)
    new_net.apply(weight_init)

    for name, param in net.named_parameters():
        if any(key in name for key in linear_keys):
            noise = new_net.state_dict()[name] * (1 - perturb_factor)
            param.data = param.data * perturb_factor + noise
        else:
            param.data = net.state_dict()[name]
    optimizer.state = defaultdict(dict)
    return net, optimizer

    
    
def cal_dormant_grad(model, type = 'critic', percentage=0.025):
    metrics = dict()
    total_neurons = 0
    dormant_neurons = 0
    
    count = 0
    for module in (module for module in model.modules() if isinstance(module, nn.Linear) and module.weight.grad is not None):
        grad_norm = module.weight.grad.norm(dim=1)  
        avg_grad_norm = grad_norm.mean()
        dormant_indice = (grad_norm < avg_grad_norm * percentage).nonzero(as_tuple=True)[0]
        total_neurons += module.weight.shape[0]
        dormant_neurons += len(dormant_indice)
        module_dormant_grad = len(dormant_indice) / module.weight.shape[0]
        metrics[
                type + '_' + str(count) +
                '_grad_dormant'] = module_dormant_grad
        count += 1
    metrics[type + "_grad_dormant_ratio"] = dormant_neurons / total_neurons
    return metrics


def perturb_factor(dormant_ratio, max_perturb_factor=0.9, min_perturb_factor=0.2):
    return min(max(min_perturb_factor, 1 - dormant_ratio), max_perturb_factor)


def cal_dormant_ratio(model, *inputs, type='policy', percentage=0.1):
    metrics = dict()
    hooks = []
    hook_handlers = []
    total_neurons = 0
    dormant_neurons = 0
    dormant_indices = dict()
    active_indices = dict()

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = LinearOutputHook()
            hooks.append(hook)
            hook_handlers.append(module.register_forward_hook(hook))

    with torch.no_grad():
        model(*inputs)

    count = 0
    for module, hook in zip((module for module in model.modules() if isinstance(module, nn.Linear)), hooks):
        with torch.no_grad():
            for output_data in hook.outputs:
                mean_output = output_data.abs().mean(0)
                avg_neuron_output = mean_output.mean()
                dormant_indice = (mean_output < avg_neuron_output *
                                   percentage).nonzero(as_tuple=True)[0]
                all_indice = list(range(module.weight.shape[0]))
                active_indice = [index for index in all_indice if index not in dormant_indice]
                total_neurons += module.weight.shape[0]
                dormant_neurons += len(dormant_indice)
                module_dormant_ratio = len(dormant_indices) / module.weight.shape[0]
                if module_dormant_ratio > 0.1:
                    dormant_indices[str(count)] = dormant_indice
                    active_indices[str(count)] = active_indice
                count += 1

    for hook in hooks:
        hook.outputs.clear()

    for hook_handler in hook_handlers:
        hook_handler.remove()

    metrics[type + "_output_dormant_ratio"] = dormant_neurons / total_neurons

    return metrics, dormant_indices, active_indices

class LinearOutputHook:
    def __init__(self):
        self.outputs = []
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
        
def dormant_perturb(model, optimizer, dormant_indices, perturb_factor=0.2):
    random_model = deepcopy(model)
    random_model.apply(weight_init)
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    random_layers = [module for module in random_model.modules() if isinstance(module, nn.Linear)]

    for key in dormant_indices:
        perturb_layer = linear_layers[key]
        random_layer = random_layers[key]
        with torch.no_grad():
            for index in dormant_indices[key]:
                noise = (random_layer.weight[index, :] * (1 - perturb_factor)).clone()
                perturb_layer.weight[index, :] = perturb_layer.weight[index, :] * perturb_factor + noise

    optimizer.state = defaultdict(dict)
    return model, optimizer

def weighted_logsumexp(log_prob1, log_prob2, discount):
    weighted_prob = (1 - discount) * log_prob1.exp() + discount * log_prob2.exp()
    log_prob = weighted_prob.log()
    return log_prob

def stable_weighted_logsumexp(log_prob1, log_prob2, discount):
    a = torch.max(log_prob1, log_prob2)
    log_prob = weighted_logsumexp(log_prob1 - a, log_prob2 - a, discount)
    log_prob = log_prob + a
    return log_prob