import torch
from src.sparse_torch.csx_matrix import CSRMatrix3d, CSCMatrix3d
import torch_geometric as pyg
from copy import deepcopy

def data_to_cuda(inputs):
    """
    Call cuda() on all tensor elements in inputs
    :param inputs: input list/dictionary
    :return: identical to inputs while all its elements are on cuda
    """
    if type(inputs) is list:
        for i, x in enumerate(inputs):
            inputs[i] = data_to_cuda(x)
    elif type(inputs) is tuple:
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            inputs[i] = data_to_cuda(x)
    elif type(inputs) is dict:
        for key in inputs:
            inputs[key] = data_to_cuda(inputs[key])
    elif type(inputs) in [str, int, float]:
        inputs = inputs
    elif type(inputs) in [torch.Tensor, CSRMatrix3d, CSCMatrix3d]:
        inputs = inputs.cuda()
    elif type(inputs) in [pyg.data.Data, pyg.data.Batch]:
        inputs = inputs.to('cuda')
    else:
        raise TypeError('Unknown type of inputs: {}'.format(type(inputs)))
    return inputs

def cuda_copy(inputs, inputs_att):
    """
    Call cuda() on all tensor elements in inputs
    :param inputs: input list/dictionary
    :return: identical to inputs while all its elements are on cuda
    """
    if type(inputs) is list:
        for i, x in enumerate(inputs):
            inputs[i] = data_to_cuda(x)
    elif type(inputs) is tuple:
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            inputs[i] = data_to_cuda(x)
    elif type(inputs) is dict:
        for key in inputs:
            # inputs_att[key] = inputs[key].detach().clone()
            inputs_att[key] = data_to_cuda(inputs[key])
    elif type(inputs) in [str, int, float]:
        inputs = deepcopy(inputs)
    elif type(inputs) in [torch.Tensor, CSRMatrix3d, CSCMatrix3d]:
        inputs = inputs.detach().clone()
    elif type(inputs) in [pyg.data.Data, pyg.data.Batch]:
        inputs = inputs.detach().clone()
    else:
        raise TypeError('Unknown type of inputs: {}'.format(type(inputs)))
    return inputs_att

def data_to_cuda_sample(inputs, idx_to_fool):
    """
    Call cuda() on all tensor elements in inputs
    :param inputs: input list/dictionary
    :return: identical to inputs while all its elements are on cuda
    """
    if type(inputs) is list:
        for i, x in enumerate(inputs):
            inputs[i] = data_to_cuda_sample(x, idx_to_fool)
    elif type(inputs) is tuple:
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            inputs[i] = data_to_cuda_sample(x, idx_to_fool)
    elif type(inputs) is dict:
        for key in inputs:
            # print(key)
            inputs[key] = data_to_cuda_sample(inputs[key], idx_to_fool)
    elif type(inputs) in [str, int, float]:
        inputs = inputs
    elif type(inputs) in [torch.Tensor, CSRMatrix3d, CSCMatrix3d]:
        # import pdb; pdb.set_trace()
        if type(inputs) == torch.Tensor:
            inputs = inputs[idx_to_fool].cuda()
        else:
            inputs = inputs.cuda()
    elif type(inputs) in [pyg.data.Data, pyg.data.Batch]:
        # import pdb; pdb.set_trace()
        inputs = inputs.to('cuda')
    else:
        raise TypeError('Unknown type of inputs: {}'.format(type(inputs)))
    return inputs