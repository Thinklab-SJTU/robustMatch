from numpy.core.numerictypes import english_lower
import torch
import torch.nn.functional as F
import numpy as np
from src.utils.config import cfg
from src.factorize_graph_matching import kronecker_sparse, kronecker_torch
from src.evaluation_metric import *
from src.feature_align import detect_bound
from src.loss_func import GMLoss
from src.utils.data_to_cuda import data_to_cuda
from src.dataset.data_loader import GMDataset, get_dataloader

import time
from collections import OrderedDict

#from models.NGM.model_v2 import Net
from src.utils.model_sl import load_model

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

class AttackedObject:
    def __init__(self, att_key=None, eps=None, bs=None, num=2, dtype=torch.float32, device=None):
        self.att_key = att_key
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.bs = bs # batchsize 
        self.delta = None
        self.max_delta = None
        self.max_loss = torch.zeros(bs, device=device)
        self.delta_shape = None
        self.delta_grads = None
        self.num = num # N-graph Matching, here we only consider 2-graph matching

    def init_max_delta(self, inputs):
        self.delta_shape = list(inputs[0].shape)
        self.delta_shape.insert(0, self.num)
        self.max_delta = torch.zeros(self.delta_shape, device=self.device)

    def init_delta(self):
        self.delta = torch.zeros(self.delta_shape, device=self.device)
        self.delta.uniform_(-self.eps, self.eps)
        self.delta.requires_grad = True  
        self.momentum = torch.zeros(self.delta_shape, device=self.device)

class AttackBase:
    def __init__(self, att_obj:str, criterion: GMLoss, eps:tuple, iter_num, alpha,  device, inv=False):
        self.device = device

        self.criterion = criterion
        self.criterion_name = criterion.name.lower()
        self.att_obj = att_obj
        
        self.epsilon_feat = torch.tensor(int(eps[0])/255., dtype=torch.float32, device=device)
        self.epsilon_loc  = torch.tensor(eps[1], dtype=torch.float32, device=device)
        self.eps_dict = {'Ps': self.epsilon_loc, 'images': self.epsilon_feat}

        self.att_key_list = []
        if 'pos' in att_obj or 'struc' in att_obj:
            self.att_key_list.append('Ps')
        if 'pixel' in att_obj:
            self.att_key_list.append('images')
        if self.att_key_list == []:
            assert False, 'Invalid attacked object' + att_obj

        self.attack_iters = iter_num
        self.alpha        = alpha

        self.restarts = cfg.ATTACK.RESTARTS
        self.early_stop_ratio = cfg.ATTACK.EARLY_STOP_RATIO
        self.grad_inv = -1 if inv else 1
        self.mu = cfg.ATTACK.MU
        self.bn_eval = cfg.TRAIN.BN_EVAL

    def __call__(self, model, inputs, criterion=None):
        if criterion == None:
            return self._main(model, inputs, self.criterion)
        else:
            return self._main(model, inputs, criterion)

    def _main(self, model, inputs, criterion):
        if self.bn_eval:
            model.eval()
        att_obj_dict = OrderedDict()
        batch_size = inputs['batch_size']

        for att_key in self.att_key_list:
            att_obj_dict[att_key] = AttackedObject(att_key, self.eps_dict[att_key], batch_size, device=self.device)
            att_obj_dict[att_key].init_max_delta(inputs[att_key])

        for zz in range(self.restarts):

            for att_obj in att_obj_dict.values():
                att_obj.init_delta()

            for num_iter in range(self.attack_iters):
                with torch.set_grad_enabled(True):
                    for att_key, att_obj in att_obj_dict.items():
                        for i in range(att_obj.num):
                            inputs[att_key][i] += att_obj.delta[i]

                    pred = model(inputs)

                    for att_key, att_obj in att_obj_dict.items():
                        for i in range(att_obj.num):
                            inputs[att_key][i] -= att_obj.delta[i]

                    s_pred, img_pred = pred['ds_mat'], pred['images']

                    # Early Stop when the matching accuracy w.r.t attacked graphs is below the threshold. 
                    precision = matching_precision(pred['perm_mat'], pred['gt_perm_mat'], pred['ns'][0])
                    index = torch.where(precision >= self.early_stop_ratio)[0]
                    if len(index) == 0:
                        break

                    if type(s_pred) is list:
                        s_pred = s_pred[-1]

                    if self.criterion_name == 'ourloss':
                        loss = criterion(pred, pred)
                    else:
                        loss = criterion(pred)

                    delta_grads = torch.autograd.grad(loss, [att_obj.delta for att_obj in att_obj_dict.values()], create_graph=False)
                    for i, att_obj in enumerate(att_obj_dict.values()):
                        att_obj.delta_grads = delta_grads[i]

                    for att_key, att_obj in att_obj_dict.items():
                        for i in range(att_obj.num):
                            if att_key == 'Ps':
                                att_obj.momentum[i]= self._MaskUpdateClip_loc(model, criterion, i, inputs, inputs[att_key][i], inputs['ns'][i], att_obj.delta[i], att_obj.delta_grads[i], index, att_obj.momentum[i])
                            else:
                                att_obj.momentum[i] = self._MaskUpdateClip_feat(att_obj.momentum[i], att_obj.delta[i], att_obj.delta_grads[i], index)

                    if 'pos+struc' in self.att_obj:
                        inputs = GMDataset.collate_batch(inputs, loc_att_mode='both')
                        inputs = data_to_cuda(inputs)
                    elif 'struc' in self.att_obj:
                        inputs = GMDataset.collate_batch(inputs, 'only_struc')
                        inputs = data_to_cuda(inputs)

            if self.restarts > 1:
                all_loss = criterion(model(inputs), reduction='none').detach()
                for att_key, att_obj in att_obj_dict.items():
                    for i in range(att_obj.num):
                        att_obj.max_delta[i][all_loss >= att_obj.max_loss] = att_obj.delta.detach()[i][all_loss >= att_obj.max_loss]
                    att_obj.max_loss = torch.max(att_obj.max_loss, all_loss)
            else:
                for att_key, att_obj in att_obj_dict.items():
                    att_obj.max_delta = att_obj.delta
                    att_obj.max_loss = 0.

        for att_key, att_obj in att_obj_dict.items():
            for i in range(att_obj.num):
                inputs[att_key][i] += att_obj.max_delta[i]

        if self.bn_eval:
            model.train()
        return inputs, None

    def _MaskUpdateClip_feat(self, momentum, delta, grad, index):
            grad = grad.detach()
            delta = delta.detach()
            d = delta[index, :, :]
            g = grad[index, :, :] * self.grad_inv

            # momentum
            if self.mu != 0:
                m = momentum[index, :, :]
                m = self.mu * m + g / torch.mean(torch.abs(g), dim=(1,2,3), keepdim=True)
                g = m
                momentum[index, :, :] = m

            d = clamp(d + self.alpha * self.epsilon_feat * torch.sign(g), -self.epsilon_feat, self.epsilon_feat)
            delta.data[index, :, :] = d
            grad.zero_()   
            return momentum

    def _MaskUpdateClip_loc(self, model, criterion, i, inputs, X, ns, delta, grad, index, momentum):
        """
        index: 1 denotes current delta needs to be updated while 0 means no changes needs to be done;
        inputs: inputs['Ps'] = X;
        X: the last perturbed location; 
        delta: X_ori + delta = X;
        delta_single: delta_new = delta + delta_single;
        """
        delta = delta.detach()
        grad =  grad.detach()
        d = delta[index, :, :]
        g = grad[index, :, :] * self.grad_inv

        # momentum
        if self.mu != 0:
            m = momentum[index, :, :]
            m = self.mu * m + g / torch.mean(torch.abs(g), dim=(1,2), keepdim=True)
            g = m
            momentum[index, :, :] = m

        # Stage one: clip delta in the epsilon-ball;
        # d = clamp(d + alpha * epsilon * torch.sign(g), -epsilon, epsilon)
        delta_single = clamp(self.alpha * self.epsilon_loc * torch.sign(g), -self.epsilon_loc-d, self.epsilon_loc-d)
        # Stage two: clip delta when X+delta crosses the current feature cell; 
        # detect which cell X locates in feature space; 
        bound_X = detect_bound(X[index, :, :], ns, 256, perturb_type='node')
        # detect which cell X+delta locates in feature space; 
        bound_X_plus = detect_bound((X+delta_single)[index, :, :], ns, 256, perturb_type='node')

        clip_ratio = torch.ones(d.shape[:2], device=self.device)
        clip_ratio_ori = torch.ones(d.shape[:2], device=self.device)
        for idx in range(bound_X.shape[0]):
            for p_idx, bounds in enumerate(bound_X[idx]):                
                clip_ratio[idx, p_idx] = self._intersect_bbox(bounds, bound_X_plus[idx, p_idx, :])

        bool_idcs_left = torch.zeros((delta_single.shape[0]), device=self.device, dtype=torch.bool)
        bool_idcs_right = torch.ones((delta_single.shape[0]), device=self.device, dtype=torch.bool)
        delta_single[bool_idcs_left, :, :] *= clip_ratio.unsqueeze(-1)[bool_idcs_left, :, :]
        delta_single[bool_idcs_right, :, :] *= clip_ratio_ori.unsqueeze(-1)[bool_idcs_right, :, :]

        delta.data[index, :, :] += delta_single
        grad.zero_()
        return momentum

    def _decide_loss(self, model, criterion, i, inputs, delta):
        """
        compute loss w.r.t input with added perturbation 
        Note that delta is for the 'i' the graph
        return the loss
        """ 
        with torch.no_grad():
            inputs['Ps'][i] += delta
            pred = model(inputs)
            loss = criterion(pred, reduction='none')
            inputs['Ps'][i] -= delta

        return loss

    def _intersect_bbox(self, bounds, bounds_X_v, out=None, device=None):
        if device is None:
            device = bounds.device
        if out is None:
            out = torch.tensor(1., dtype=torch.float32, device=device)
        eps = 1e-6
        x0, x1, y0, y1, X0, Y0 = bounds
        _, _, _, _, X_v0, Y_v0 = bounds_X_v
        X = (X0, Y0)
        v = (X_v0 - X0, Y_v0 - Y0)
        k_inter = lambda x, v_x, loc_V: (loc_V - x) / (v_x + eps)
        x_or_y_inter = lambda y, k, v_y: y + k * v_y
        if_inter = lambda k, x_or_y, bound_0, bound_1: (k >= -eps) and (k <= 1.+eps) and (x_or_y >= bound_0-eps) and (x_or_y <= bound_1+eps)
        # check if X + out * v intersects with the bounding box vertically 
        # vector: (x0, 0)
        k_inter_x0 = k_inter(X[0], v[0], x0)
        y_inter_x0 = x_or_y_inter(X[1], k_inter_x0, v[1])
        if if_inter(k_inter_x0, y_inter_x0, y0, y1):
            out = k_inter_x0
            return out 

        # vector: (x1, 0)
        k_inter_x1 = k_inter(X[0], v[0], x1)
        y_inter_x1 = x_or_y_inter(X[1], k_inter_x1, v[1])
        if if_inter(k_inter_x1, y_inter_x1, y0, y1):
            out = k_inter_x1
            return out 

        # vector (0, y0)
        k_inter_y0 = k_inter(X[1], v[1], y0)
        x_inter_y0 = x_or_y_inter(X[0], k_inter_y0, v[0])
        if if_inter(k_inter_y0, x_inter_y0, x0, x1):
            out = k_inter_y0
            return out 

        # vector (0, y1)
        k_inter_y1 = k_inter(X[1], v[1], y1)
        x_inter_y1 = x_or_y_inter(X[0], k_inter_y1, v[0])
        if if_inter(k_inter_y1, x_inter_y1, x0, x1):
            out = k_inter_y1
            return out 

        return out

class RandomAttack(AttackBase):
    def __init__(self, att_obj, criterion, eps, device):
        super().__init__(att_obj, criterion, eps, iter_num=0, alpha=0, device=device)
        self.restarts = 1

class PGDAttack(AttackBase):
    def __init__(self, att_obj, criterion, eps, iter_num, alpha, device, inv=False):
        super().__init__(att_obj, criterion, eps, iter_num, alpha, device, inv)
        self.mu = 0

class AttackGM:
    def __init__(self, att_obj, att_type, criterion, eps, iter_num, alpha, device, inv=False):
        '''A clean API for attack Deep Graph Matching
        Params: 
            att_obj (string): choices=['pixel', 'pos', 'struc', 'pos+struc', 'pixel+pos+struc']
            att_type (string): choices=['none', 'random', 'pgd', 'momentum']
            criterion (GMLoss):
            eps (tuple): containing the epsilon of vision feature and locality respectively
            iter_num (int): number of steps for PGD attack
            alpha (float): 
            device (torch.device):
            inv (bool): whether to conduct a inverse-gradient attack, i.e., lowering the loss 
        '''
        self.att_obj = att_obj
        self.att_type = att_type
        self.criterion = criterion
        if att_type == 'none':
            self.att_func = self.baseline
        elif att_type == 'random':
            self.att_func = RandomAttack(att_obj, criterion, eps, device)
        elif att_type == 'pgd':
            self.att_func = PGDAttack(att_obj, criterion, eps, iter_num, alpha, device,  inv=inv)
        elif att_type == 'momentum':
            self.att_func = AttackBase(att_obj, criterion, eps, iter_num, alpha, device,  inv=inv)
        elif att_type == 'fgsm':
            self.att_func = PGDAttack(att_obj, criterion, eps, iter_num=1, alpha=alpha, device=device,   inv=inv)
        elif att_type == 'pgd20':
            self.att_func = PGDAttack(att_obj, criterion, eps, iter_num=20, alpha=alpha, device=device,  inv=inv)
        elif att_type == 'pgd50':
            self.att_func = PGDAttack(att_obj, criterion, eps, iter_num=50, alpha=alpha, device=device,  inv=inv)
        elif att_type == 'pgd100':
            self.att_func = PGDAttack(att_obj, criterion, eps, iter_num=100, alpha=alpha, device=device,  inv=inv)
        else:
            raise NotImplementedError("No such attack type :" + att_type)

    def baseline(self, model, inputs, criterion=None):
        return inputs, None

    def __call__(self, models, inputs, criterion=None):
        # case for adaptive attack for AAR loss
        if criterion != None and criterion.name == 'ourloss':
            inputs_att, _ = self.att_func(models, inputs, GMLoss('perm', '2GM'))
            outputs_att   = models(inputs_att)
            criterion.update_outputs_att(outputs_att)
            
        if criterion != None:
            outcomes, pos = self.att_func(models, inputs, criterion)
        else:
            outcomes, pos = self.att_func(models, inputs, self.criterion)
        
        return outcomes, pos

class BlackAttackGM:
    def __init__(self, att_obj, att_type, criterion:GMLoss, device):
        import importlib
        from src.parallel import DataParallel
        mod = importlib.import_module(cfg.VICTIM_MODULE)
        Net = mod.Net
        self.model = Net().to(device)
        self.model = DataParallel(self.model, device_ids=cfg.GPUS)
        load_model(self.model, cfg.VICTIM_PATH, strict=False)
        self.attack = AttackGM(att_obj, att_type, criterion, 
                               eps=(cfg.ATTACK.EPSILON_FEATURE, cfg.ATTACK.EPSILON_LOCALITY), 
                               iter_num=cfg.ATTACK.EVAL_STEP, 
                               alpha=cfg.ATTACK.EVAL_ALPHA, 
                               device=device, 
                               inv=False)
       
    def __call__(self, inputs, criterion=None):
        inputs, _  = self.attack(self.model, inputs=inputs, criterion=criterion)
        return inputs
