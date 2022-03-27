import torch
import torch.nn as nn
import torch.nn.functional as F
from src.lap_solvers.hungarian import hungarian
from src.displacement_layer import Displacement
from torch import Tensor
from src.utils.config import cfg

from src.evaluation_metric import objective_score

class GMLoss:
    def __init__(self, name, problem_type):
        if name.lower() == 'offset':
            self.criterion = OffsetLoss(norm=cfg.TRAIN.RLOSS_NORM)
        elif name.lower() == 'perm':
            self.criterion = PermutationLoss()
        elif name.lower() == 'ce':
            self.criterion = CrossEntropyLoss()
        elif name.lower() == 'focal':
            self.criterion = FocalLoss(gamma=0.)
        elif name.lower() == 'hung':
            self.criterion = PermutationLossHung()
        elif name.lower() == 'hamming':
            self.criterion = HammingLoss()
        elif name.lower() == 'ourloss':
            self.criterion = OurPermutationLoss(reg_level=cfg.TRAIN.REG_LEVEL, reg_ratio=cfg.TRAIN.REG_RATIO)
        elif name.lower() == 'cw':
            self.criterion = CWLoss()
        else:
            raise ValueError('Unknown loss function {}'.format(name))

        self.name = name
        self.problem_type = problem_type

    def __call__(self, outputs, outputs_att=None, device=None, reduction='mean'):
        if self.problem_type == '2GM':
            if self.name == 'offset': 
                d_gt, grad_mask = displacement(outputs['gt_perm_mat'], *outputs['Ps'], outputs['ns'][0])
                d_pred, _ = displacement(outputs['ds_mat'], *outputs['Ps'], outputs['ns'][0])
                loss = self.criterion(d_pred, d_gt, grad_mask)
            elif self.name in ['perm', 'ce', 'soft_perm', 'cw']:
                loss = self.criterion(outputs['ds_mat'], outputs['gt_perm_mat'], *outputs['ns'], reduction=reduction)
            elif self.name == 'hung':
                loss = self.criterion(outputs['ds_mat'], outputs['gt_perm_mat'], *outputs['ns'])
            elif self.name == 'hamming':
                loss = self.criterion(outputs['perm_mat'], outputs['gt_perm_mat'])
            elif self.name == 'plain':
                loss = torch.sum(outputs['loss'])
            elif self.name == 'ourloss':
                if outputs_att == None:
                    outputs_att = self.outputs_att
                loss = self.criterion(outputs['ds_mat'], outputs['perm_mat'], outputs_att['perm_mat'], outputs['gt_perm_mat'], *outputs['ns'], reduction=reduction)
            else:
                raise ValueError('Unsupported loss function {} for problem type {}'.format(self.name, cfg.PROBLEM.TYPE))

        elif self.problem_type in ['MGM', 'MGMC']:
            assert 'ds_mat_list' in outputs
            assert 'graph_indices' in outputs
            assert 'perm_mat_list' in outputs
            assert 'gt_perm_mat_list' in outputs

            if self.name in ['perm', 'ce' 'hung']:
                loss = torch.zeros(1, device=device)
                ns = outputs['ns']
                for s_pred, x_gt, (idx_src, idx_tgt) in \
                        zip(outputs['ds_mat_list'], outputs['gt_perm_mat_list'], outputs['graph_indices']):
                    l = criterion(s_pred, x_gt, ns[idx_src], ns[idx_tgt])
                    loss += l
                loss /= len(outputs['ds_mat_list'])
            elif cfg.TRAIN.LOSS_FUNC == 'plain':
                loss = torch.sum(outputs['loss'])
            else:
                raise ValueError('Unsupported loss function {} for problem type {}'.format(cfg.TRAIN.LOSS_FUNC, cfg.PROBLEM.TYPE))

        return loss

    def get_criterion(self):
        return self.criterion

    def update_outputs_att(self, outputs_att):
        self.outputs_att = outputs_att


class OurPermutationLoss(nn.Module):
    def __init__(self, reg_level=1, reg_ratio=0.1):
        super(OurPermutationLoss, self).__init__()
        self.reg_level = reg_level
        self.reg_ratio = reg_ratio
    def forward(self, pred_dsmat: Tensor, pred_perm: Tensor, pred_perm_att: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor, reduction='mean') -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged permutation loss
        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]
        pred_dsmat = pred_dsmat.to(dtype=torch.float32)
        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        if reduction == 'mean':
            loss = torch.tensor(0.).to(pred_dsmat.device)
        elif reduction == 'none':
            loss = torch.zeros(batch_num).to(pred_dsmat.device)
        else:
            raise ValueError('Need use an appropriate way to compute the final loss. optional: mean|none')
        n_sum = torch.zeros_like(loss)
        
        with torch.no_grad():
            sim_batch = self.dig_similarity(pred_perm_att, gt_perm, src_ns, tgt_ns)

        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            cur_pred_dsmat = pred_dsmat[batch_slice]
            cur_gt_perm = gt_perm[batch_slice]
            # 1. mask the current pred perm via gt_perm 
            cur_mask = cur_gt_perm.detach().clone()
            # 2. filter out the similar keypoints using sim_array 
            cur_sim_array = sim_batch[b]
            for i in range(src_ns[b]):
                num_sim_kps = len(cur_sim_array[i])
                if num_sim_kps == 0: # current i gets matched correctly 
                    cur_mask[i, :] = 0.
                else:
                    # num_sim_kps = len(cur_sim_array[i])
                    max_sim_kps = min(num_sim_kps, self.reg_level)
                    sim_kps = [cur_sim_array[i][j] for j in range(max_sim_kps)]
                    cur_mask[i, sim_kps] = -1
            # 3. build the margin regularization loss(element wise multiplication)
            reg_loss = cur_pred_dsmat * cur_mask
            # reduction way: 'mean'(by default): first sum the loss all together, then average loss row by row; 
            # 'none': average values together and output batch-size loss
            if reduction == 'mean':
                loss -= self.reg_ratio * reg_loss.sum()
                loss += F.binary_cross_entropy(
                    cur_pred_dsmat,
                    cur_gt_perm,
                    reduction='sum')
                n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)
            elif reduction == 'none':
                loss[b] -= self.reg_ratio * reg_loss.mean()
                loss[b] += F.binary_cross_entropy(
                    cur_pred_dsmat,
                    cur_gt_perm,
                    reduction='mean')
        if reduction == 'mean':
            return loss / n_sum
        elif reduction == 'none':
            return loss

    def dig_similarity(self, pred_perm_att: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        # 1. mask the correctly matched pair 
        # 2. DFS to dig out the semantically similary keypoint groups (pairwise, triple, etc.)
        batch_num = pred_perm_att.shape[0]
        sim_batch = [[None] for _ in range(batch_num)]
        # TODO: optimize the func time cost later 
        pred_perm_att = pred_perm_att.detach().cpu()
        gt_perm = gt_perm.detach().cpu()

        def dfs(i, depth, visited, pred_perm, sim_loop=None):
            visited[i] = True
            next_i = pred_perm[i]
            if sim_loop is None and depth == 0:
                sim_loop = [i]
            if visited[next_i] or next_i == i:
                return sim_loop
            sim_loop.append(next_i.item())
            return dfs(next_i, depth+1, visited, pred_perm, sim_loop)
        # The main logic: 
        # 1. Map the matched G2 node index back to G1; 
        # 2. Perform DFS to find the sim loop on G1; (semantically ambiguous nodes)
        # 3. Map the G1 node index in the sim loop back to G2;
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            sim_array = [[None] for _ in range(src_ns[b])]
            sim_loop_array = []
            visited = [False for _ in range(tgt_ns[b])]
            cur_pred_perm_att = pred_perm_att[batch_slice]
            cur_gt_perm = gt_perm[batch_slice]
            col_index_gt = torch.nonzero(cur_gt_perm, as_tuple=True)[1]
            col_index_pred = torch.nonzero(cur_pred_perm_att, as_tuple=True)[1]
            col_index_gt_t = torch.nonzero(torch.t(cur_gt_perm), as_tuple=True)[1]
            # map the matched node index of g2 back to g1
            matching_on_g1 = [col_index_gt_t[i] for i in col_index_pred]
            for i in range(src_ns[b]):
                if not visited[i]:
                    sim_loop = dfs(i, 0, visited, matching_on_g1)
                    sim_loop_array.append(sim_loop)
            # decode sim_loop to similarity pair
            gt_mapping_g1_to_g2 = lambda index_g1, gt_mapping: [gt_mapping[i].item() for i in index_g1]
            for sim_loop in sim_loop_array:
                for loc, i in enumerate(sim_loop):
                    if loc == 0: sim_array[i] = gt_mapping_g1_to_g2(sim_loop[loc+1:], col_index_gt)
                    elif loc == len(sim_loop)-1: sim_array[i] = gt_mapping_g1_to_g2(sim_loop[:loc], col_index_gt)
                    else:
                        tmp_loop = sim_loop[loc+1:]
                        tmp_loop.extend(sim_loop[:loc])
                        sim_array[i] = gt_mapping_g1_to_g2(tmp_loop, col_index_gt)
            sim_batch[b] = sim_array
        return sim_batch

class PermutationLoss(nn.Module):
    r"""
    Binary cross entropy loss between two permutations, also known as "permutation loss".
    Proposed by `"Wang et al. Learning Combinatorial Embedding Networks for Deep Graph Matching. ICCV 2019."
    <http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Combinatorial_Embedding_Networks_for_Deep_Graph_Matching_ICCV_2019_paper.pdf>`_

    .. math::
        L_{perm} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left(\mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j} + (1-\mathbf{X}^{gt}_{i,j}) \log (1-\mathbf{S}_{i,j}) \right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self,):
        super(PermutationLoss, self).__init__()
    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor, reduction='mean') -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged permutation loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]
        pred_dsmat = pred_dsmat.to(dtype=torch.float32)
        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        if reduction == 'mean':
            loss = torch.tensor(0.).to(pred_dsmat.device)
        elif reduction == 'none':
            loss = torch.zeros(batch_num).to(pred_dsmat.device)
        else:
            raise ValueError('Need use an appropriate way to compute the final loss. optional: mean|none')
        n_sum = torch.zeros_like(loss)
        
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            if reduction == 'mean':
                loss += F.binary_cross_entropy(
                    pred_dsmat[batch_slice],
                    gt_perm[batch_slice],
                    reduction='sum')
                n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)
            elif reduction == 'none':
                loss[b] += F.binary_cross_entropy(
                    pred_dsmat[batch_slice],
                    gt_perm[batch_slice],
                    reduction='mean')
        if reduction == 'mean':
            return loss / n_sum
        elif reduction == 'none':
            return loss

class CWLoss(nn.Module):
    r"""
    Multi-class cross entropy loss between two permutations.

    .. math::
        L_{ce} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2} \left(\mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(CWLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor, reduction='sum') -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged cross-entropy loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            cur_pred_dsmat = pred_dsmat[batch_slice]
            cur_gt_perm = gt_perm[batch_slice]
            gt_index = torch.max(cur_gt_perm, dim=-1).indices
            top2_index = self.hard_label_mc_target(cur_pred_dsmat, gt_index)
            loss += F.nll_loss(
                torch.log(cur_pred_dsmat),
                gt_index,
                reduction=reduction)
            loss -= F.nll_loss(
                torch.log(cur_pred_dsmat),
                top2_index,
                reduction=reduction)
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum

    def hard_label_mc_target(self, outputs, y):
        y_t = torch.zeros((y.shape[0]), dtype=torch.int64).to(y.device)
        if isinstance(outputs, dict):
            outputs = outputs['logits']
        ind_sorted = outputs.sort(dim=-1)[1]
        # index for those correctly classified.
        ind = ind_sorted[:, -1] == y
        true_idcs = torch.nonzero(ind).squeeze(1)
        false_idcs = torch.nonzero(~ind).squeeze(1)
        y_t[true_idcs] = ind_sorted[:, -2][true_idcs]
        y_t[false_idcs] = ind_sorted[:, -1][false_idcs]
        return y_t
class CrossEntropyLoss(nn.Module):
    r"""
    Multi-class cross entropy loss between two permutations.

    .. math::
        L_{ce} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2} \left(\mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor, reduction='none') -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged cross-entropy loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            gt_index = torch.max(gt_perm[batch_slice], dim=-1).indices
            loss += F.nll_loss(
                torch.log(pred_dsmat[batch_slice]),
                gt_index,
                reduction=reduction)
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum

class PermutationLossHung(nn.Module):
    r"""
    Binary cross entropy loss between two permutations with Hungarian attention. The vanilla version without Hungarian
    attention is :class:`~src.loss_func.PermutationLoss`.

    .. math::
        L_{hung} &=-\sum_{i\in\mathcal{V}_1,j\in\mathcal{V}_2}\mathbf{Z}_{ij}\left(\mathbf{X}^\text{gt}_{ij}\log \mathbf{S}_{ij}+\left(1-\mathbf{X}^{\text{gt}}_{ij}\right)\log\left(1-\mathbf{S}_{ij}\right)\right) \\
        \mathbf{Z}&=\mathrm{OR}\left(\mathrm{Hungarian}(\mathbf{S}),\mathbf{X}^\text{gt}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    Hungarian attention highlights the entries where the model makes wrong decisions after the Hungarian step (which is
    the default discretization step during inference).

    Proposed by `"Yu et al. Learning deep graph matching with channel-independent embedding and Hungarian attention.
    ICLR 2020." <https://openreview.net/forum?id=rJgBd2NYPH>`_

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.

    A working example for Hungarian attention:

    .. image:: ../../images/hungarian_attention.png
    """
    def __init__(self):
        super(PermutationLossHung, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged permutation loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        dis_pred = hungarian(pred_dsmat, src_ns, tgt_ns)
        ali_perm = dis_pred + gt_perm
        ali_perm[ali_perm > 1.0] = 1.0 # Hung
        pred_dsmat = torch.mul(ali_perm, pred_dsmat)
        gt_perm = torch.mul(ali_perm, gt_perm)
        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_dsmat[b, :src_ns[b], :tgt_ns[b]],
                gt_perm[b, :src_ns[b], :tgt_ns[b]],
                reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)
        return loss / n_sum
class OffsetLoss(nn.Module):
    r"""
    OffsetLoss Criterion computes a robust loss function based on image pixel offset.
    Proposed by `"Zanfir et al. Deep Learning of Graph Matching. CVPR 2018."
    <http://openaccess.thecvf.com/content_cvpr_2018/html/Zanfir_Deep_Learning_of_CVPR_2018_paper.html>`_

    .. math::
        \mathbf{d}_i =& \sum_{j \in V_2} \left( \mathbf{S}_{i, j} P_{2j} \right)- P_{1i} \\
        L_{off} =& \sum_{i \in V_1} \sqrt{||\mathbf{d}_i - \mathbf{d}^{gt}_i||^2 + \epsilon}

    :math:`\mathbf{d}_i` is the displacement vector. See :class:`src.displacement_layer.Displacement` or more details

    :param epsilon: a small number for numerical stability
    :param norm: (optional) division taken to normalize the loss
    """
    def __init__(self, epsilon: float=1e-5, norm=None):
        super(OffsetLoss, self).__init__()
        self.epsilon = epsilon
        self.norm = norm

    def forward(self, d1: Tensor, d2: Tensor, mask: float=None) -> Tensor:
        """
        :param d1: predicted displacement matrix
        :param d2: ground truth displacement matrix
        :param mask: (optional) dummy node mask
        :return: computed offset loss
        """
        # Loss = Sum(Phi(d_i - d_i^gt))
        # Phi(x) = sqrt(x^T * x + epsilon)
        if mask is None:
            mask = torch.ones_like(mask)
        x = d1 - d2
        if self.norm is not None:
            x = x / self.norm

        xtx = torch.sum(x * x * mask, dim=-1)
        phi = torch.sqrt(xtx + self.epsilon)
        loss = torch.sum(phi) / d1.shape[0]

        return loss


class FocalLoss(nn.Module):
    r"""
    Focal loss between two permutations.

    .. math::
        L_{focal} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left((1-\mathbf{S}_{i,j})^{\gamma} \mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j} +
        \mathbf{S}_{i,j}^{\gamma} (1-\mathbf{X}^{gt}_{i,j}) \log (1-\mathbf{S}_{i,j}) \right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs, :math:`\gamma` is the focal loss
    hyper parameter.

    :param gamma: :math:`\gamma` parameter for focal loss
    :param eps: a small parameter for numerical stability

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self, gamma=0., eps=1e-15):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged focal loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            x = pred_dsmat[b, :src_ns[b], :tgt_ns[b]]
            y = gt_perm[b, :src_ns[b], :tgt_ns[b]]
            loss += torch.sum(
                - (1 - x) ** self.gamma * y * torch.log(x + self.eps)
                - x ** self.gamma * (1 - y) * torch.log(1 - x + self.eps)
            )
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum


class InnerProductLoss(nn.Module):
    r"""
    Inner product loss for self-supervised problems.

    .. math::
        L_{ce} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2} \left(\mathbf{X}^{gt}_{i,j} \mathbf{S}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(InnerProductLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged inner product loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            loss -= torch.sum(pred_dsmat[batch_slice] * gt_perm[batch_slice])
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum


class HammingLoss(torch.nn.Module):
    r"""
    Hamming loss between two permutations.

    .. math::
        L_{hamm} = \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left(\mathbf{X}_{i,j} (1-\mathbf{X}^{gt}_{i,j}) +  (1-\mathbf{X}_{i,j}) \mathbf{X}^{gt}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    Firstly adopted by `"Rolinek et al. Deep Graph Matching via Blackbox Differentiation of Combinatorial Solvers.
    ECCV 2020." <https://arxiv.org/abs/2003.11657>`_

    .. note::
        Hamming loss is defined between two discrete matrices, and discretization will in general truncate gradient. A
        workaround may be using the `blackbox differentiation technique <https://arxiv.org/abs/1912.02175>`_.
    """
    def __init__(self):
        super(HammingLoss, self).__init__()

    def forward(self, pred_perm: Tensor, gt_perm: Tensor) -> Tensor:
        r"""
        :param pred_perm: :math:`(b\times n_1 \times n_2)` predicted permutation matrix :math:`(\mathbf{X})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :return:
        """
        errors = pred_perm * (1.0 - gt_perm) + (1.0 - pred_perm) * gt_perm
        return errors.mean(dim=0).sum()

class CWLoss(nn.Module):
    r"""
    Multi-class cross entropy loss between two permutations.
    .. math::
        L_{ce} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2} \left(\mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j}\right)
    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.
    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(CWLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor, reduction='sum') -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged cross-entropy loss
        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            cur_pred_dsmat = pred_dsmat[batch_slice]
            cur_gt_perm = gt_perm[batch_slice]
            gt_index = torch.max(cur_gt_perm, dim=-1).indices
            top2_index = self.hard_label_mc_target(cur_pred_dsmat, gt_index)
            loss += F.nll_loss(
                torch.log(cur_pred_dsmat),
                gt_index,
                reduction=reduction)
            loss -= F.nll_loss(
                torch.log(cur_pred_dsmat),
                top2_index,
                reduction=reduction)
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum

    def hard_label_mc_target(self, outputs, y):
        y_t = torch.zeros((y.shape[0]), dtype=torch.int64).to(y.device)
        if isinstance(outputs, dict):
            outputs = outputs['logits']
        ind_sorted = outputs.sort(dim=-1)[1]
        # index for those correctly classified.
        ind = ind_sorted[:, -1] == y
        true_idcs = torch.nonzero(ind).squeeze(1)
        false_idcs = torch.nonzero(~ind).squeeze(1)
        y_t[true_idcs] = ind_sorted[:, -2][true_idcs]
        y_t[false_idcs] = ind_sorted[:, -1][false_idcs]
        return y_t
