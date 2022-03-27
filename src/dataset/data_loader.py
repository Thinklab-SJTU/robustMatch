import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torch_geometric as pyg
import numpy as np
import random
from src.build_graphs import build_graphs
from src.factorize_graph_matching import kronecker_sparse, kronecker_torch
from src.sparse_torch import CSRMatrix3d
from src.dataset import *

from src.utils.config import cfg

from itertools import combinations, product
import copy

class GMDataset(Dataset):
    def __init__(self, name, length, cls=None, problem='2GM', only_on_graph=0, 
                    stg_src=cfg.GRAPH.SRC_GRAPH_CONSTRUCT, stg_tgt=cfg.GRAPH.TGT_GRAPH_CONSTRUCT ,
                    **args):
        self.name = name
        self.ds = eval(self.name)(**args)
        self.length = length  # NOTE images pairs are sampled randomly, so there is no exact definition of dataset size
                              # length here represents the iterations between two checkpoints
        self.obj_size = self.ds.obj_resize
        self.cls = None if cls in ['none', 'all'] else cls

        if self.cls is None:
            if problem == 'MGMC':
                self.classes = list(combinations(self.ds.classes, cfg.PROBLEM.NUM_CLUSTERS))
            else:
                self.classes = self.ds.classes
        else:
            self.classes = [self.cls]

        self.problem_type = problem
        self.only_on_graph = only_on_graph
        self.stg_src = stg_src
        self.stg_tgt = stg_tgt

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.problem_type == '2GM':
            return self.get_pair(idx, self.cls)
        elif self.problem_type == 'MGM':
            return self.get_multi(idx, self.cls)
        elif self.problem_type == 'MGMC':
            return self.get_multi_cluster(idx)
        else:
            raise NameError("Unknown problem type: {}".format(self.problem_type))

    @staticmethod
    def to_pyg_graph(A, P):
        rescale = max(cfg.PROBLEM.RESCALE)

        edge_feat = 0.5 * (np.expand_dims(P, axis=1) - np.expand_dims(P, axis=0)) / rescale + 0.5  # from Rolink's paper
        edge_index = np.nonzero(A)
        edge_attr = edge_feat[edge_index]

        edge_attr = np.clip(edge_attr, 0, 1)
        assert (edge_attr > -1e-5).all(), P

        o3_A = np.expand_dims(A, axis=0) * np.expand_dims(A, axis=1) * np.expand_dims(A, axis=2)
        hyperedge_index = np.nonzero(o3_A)

        pyg_graph = pyg.data.Data(
            x=torch.tensor(P / rescale).to(torch.float32),
            edge_index=torch.tensor(np.array(edge_index), dtype=torch.long),
            edge_attr=torch.tensor(edge_attr).to(torch.float32),
            hyperedge_index=torch.tensor(np.array(hyperedge_index), dtype=torch.long),
        )
        return pyg_graph

    def get_pair(self, idx, cls):
        #anno_pair, perm_mat = self.ds.get_pair(self.cls if self.cls is not None else
        #                                       (idx % (cfg.BATCH_SIZE * len(self.classes))) // cfg.BATCH_SIZE)
        try:
            anno_pair, perm_mat = self.ds.get_pair(cls, tgt_outlier=cfg.PROBLEM.TGT_OUTLIER, src_outlier=cfg.PROBLEM.SRC_OUTLIER)
        except TypeError:
            anno_pair, perm_mat = self.ds.get_pair(cls)
        if min(perm_mat.shape[0], perm_mat.shape[1]) <= 2 or perm_mat.size >= cfg.PROBLEM.MAX_PROB_SIZE > 0:
            return self.get_pair(idx, cls)

        cls = [anno['cls'] for anno in anno_pair]
        # use perturbed node locations to build graphs
        # while using original node locations to extract features 
        
        # no attack: keypoints = keypoints_att 
        # normal locality attack: perturb keypoints and keypoints_att
        # only on graph: perturb keypoints_att only 
        # 'name': 'Top_Rudder', 'visible': '1', 'x': 202.69367088607595, 'y': 12.677365269461072, 'z': '0.00'},
        
        P1 = [(kp['x'], kp['y']) for kp in anno_pair[0]['keypoints']]
        P2 = [(kp['x'], kp['y']) for kp in anno_pair[1]['keypoints']]
        Names1 = { kp['name']: i for i, kp in enumerate(anno_pair[0]['keypoints'])}
        Names2 = { kp['name']: i for i, kp in enumerate(anno_pair[1]['keypoints'])}
        # P1_att = [(kp['x'], kp['y']) for kp in anno_pair[0]['keypoints_att']]
        # P2_att = [(kp['x'], kp['y']) for kp in anno_pair[1]['keypoints_att']]

        n1, n2 = len(P1), len(P2)
        univ_size = [anno['univ_size'] for anno in anno_pair]

        P1 = np.array(P1)
        P2 = np.array(P2)
        # P1_att = np.array(P1_att)
        # P2_att = np.array(P2_att)

        # if self.only_on_graph:
        A1, G1, H1, e1 = build_graphs(P1, n1, stg=self.stg_src, sym=cfg.GRAPH.SYM_ADJACENCY, namelist=Names1)
        if cfg.GRAPH.TGT_GRAPH_CONSTRUCT == 'same':
            G2 = perm_mat.transpose().dot(G1)
            H2 = perm_mat.transpose().dot(H1)
            A2 = G2.dot(H2.transpose())
            e2 = e1
        else:
            A2, G2, H2, e2 = build_graphs(P2, n2, stg=self.stg_tgt, sym=cfg.GRAPH.SYM_ADJACENCY, namelist=Names2)

        pyg_graph1 = self.to_pyg_graph(A1, P1)
        pyg_graph2 = self.to_pyg_graph(A2, P2)

        ret_dict = {'Ps': [torch.Tensor(x) for x in [P1, P2]],
                    'Ps_ori': [torch.Tensor(x) for x in [P1.copy(), P2.copy()]],
                    # 'P_names': [torch.Tensor(x) for x in [P1_names, P2_names]],
                    'ns': [torch.tensor(x) for x in [n1, n2]],
                    'es': [torch.tensor(x) for x in [e1, e2]],
                    'gt_perm_mat': perm_mat,
                    'Gs': [torch.Tensor(x) for x in [G1, G2]],
                    'Hs': [torch.Tensor(x) for x in [H1, H2]],
                    'As': [torch.Tensor(x) for x in [A1, A2]],
                    'pyg_graphs': [pyg_graph1, pyg_graph2],
                    'cls': [str(x) for x in cls],
                    'univ_size': [torch.tensor(int(x)) for x in univ_size],
                    }

        imgs = [anno['image'] for anno in anno_pair]
        if imgs[0] is not None:
            trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
                    ])
            imgs = [trans(img) for img in imgs]
            ret_dict['images'] = imgs
            ret_dict['images_ori'] = [img.clone().detach() for img in imgs]
        elif 'feat' in anno_pair[0]['keypoints'][0]:
            feat1 = np.stack([kp['feat'] for kp in anno_pair[0]['keypoints']], axis=-1)
            feat2 = np.stack([kp['feat'] for kp in anno_pair[1]['keypoints']], axis=-1)
            ret_dict['features'] = [torch.Tensor(x) for x in [feat1, feat2]]

        return ret_dict

    def get_multi(self, idx, cls):
        if (self.ds.sets == 'test' and cfg.PROBLEM.TEST_ALL_GRAPHS) or (self.ds.sets == 'train' and cfg.PROBLEM.TRAIN_ALL_GRAPHS):
            num_graphs = self.ds.len(cls)
        else:
            num_graphs = cfg.PROBLEM.NUM_GRAPHS
        anno_list, perm_mat_list = self.ds.get_multi(cls, num=num_graphs)

        assert isinstance(perm_mat_list, list)
        refetch = False
        for pm in perm_mat_list:
            if pm.shape[0] <= 2 or pm.shape[1] <= 2 or pm.size >= cfg.PROBLEM.MAX_PROB_SIZE > 0:
                refetch = True
                break
        if refetch:
            return self.get_multi(idx, cls)

        cls = [anno['cls'] for anno in anno_list]
        Ps = [[(kp['x'], kp['y']) for kp in anno_dict['keypoints']] for anno_dict in anno_list]

        ns = [len(P) for P in Ps]
        univ_size = [anno['univ_size'] for anno in anno_list]

        Ps = [np.array(P) for P in Ps]

        As = []
        Gs = []
        Hs = []
        As_tgt = []
        Gs_tgt = []
        Hs_tgt = []
        for P, n, perm_mat in zip(Ps, ns, perm_mat_list):
            # In multi-graph matching (MGM), when a graph is regarded as target graph, its topology may be different
            # from when it is regarded as source graph. These are represented by suffix "tgt".
            if cfg.GRAPH.TGT_GRAPH_CONSTRUCT == 'same' and len(Gs) > 0:
                G = perm_mat.dot(Gs[0])
                H = perm_mat.dot(Hs[0])
                A = G.dot(H.transpose())
                G_tgt = G
                H_tgt = H
                A_tgt = G_tgt.dot(H_tgt.transpose())
            else:
                A, G, H, _ = build_graphs(P, n, stg=cfg.GRAPH.SRC_GRAPH_CONSTRUCT)
                A_tgt, G_tgt, H_tgt, _ = build_graphs(P, n, stg=cfg.GRAPH.TGT_GRAPH_CONSTRUCT)
            As.append(A)
            Gs.append(G)
            Hs.append(H)
            As_tgt.append(A_tgt)
            Gs_tgt.append(G_tgt)
            Hs_tgt.append(H_tgt)

        pyg_graphs = [self.to_pyg_graph(A, P) for A, P in zip(As, Ps)]
        pyg_graphs_tgt = [self.to_pyg_graph(A, P) for A, P in zip(As_tgt, Ps)]

        ret_dict = {
            'Ps': [torch.Tensor(x) for x in Ps],
            'ns': [torch.tensor(x) for x in ns],
            'gt_perm_mat': perm_mat_list,
            'Gs': [torch.Tensor(x) for x in Gs],
            'Hs': [torch.Tensor(x) for x in Hs],
            'As': [torch.Tensor(x) for x in As],
            'Gs_tgt': [torch.Tensor(x) for x in Gs_tgt],
            'Hs_tgt': [torch.Tensor(x) for x in Hs_tgt],
            'As_tgt': [torch.Tensor(x) for x in As_tgt],
            'pyg_graphs': pyg_graphs,
            'pyg_graphs_tgt': pyg_graphs_tgt,
            'cls': [str(x) for x in cls],
            'univ_size': [torch.tensor(int(x)) for x in univ_size],
        }

        imgs = [anno['image'] for anno in anno_list]
        if imgs[0] is not None:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
            ])
            imgs = [trans(img) for img in imgs]
            ret_dict['images'] = imgs
        elif 'feat' in anno_list[0]['keypoints'][0]:
            feats = [np.stack([kp['feat'] for kp in anno_dict['keypoints']], axis=-1) for anno_dict in anno_list]
            ret_dict['features'] = [torch.Tensor(x) for x in feats]

        return ret_dict

    def get_multi_cluster(self, idx):
        dicts = []
        if self.cls is None or self.cls == 'none':
            cls_iterator = random.choice(self.classes)
        else:
            cls_iterator = self.cls
        for cls in cls_iterator:
            dicts.append(self.get_multi(idx, cls))
        ret_dict = {}
        for key in dicts[0]:
            ret_dict[key] = []
            for dic in dicts:
                ret_dict[key] += dic[key]
        return ret_dict
    
    @staticmethod
    def collate_batch(inputs, loc_att_mode='only_struc'):
        # 1. convert a batch of data to a list of single dicts 
        # 2. update each element of a list of single dicts
        # 3. convert a list of data back to a batch of data
        # inputs: dict: each value contains batch size data.
        batch_size = inputs['batch_size']
        single_data_list = []
        single_data = {}
        # temp_batch_data = {}
        pert_keys = ['Ps', 'Ps_ori', 'ns']
        temp_keys = ['gt_perm_mat', 'cls', 'univ_size', 'images', 'images_ori']
        # for key, input in inputs.items():
            # if key in pert_keys:
        # import pdb; pdb.set_trace()
        for i in range(batch_size):
            for key in pert_keys:
                if inputs[key][0].dim() == 1:
                    single_data[key] = [x[i] for x in inputs[key]]
                else:
                    single_data[key] = [x[i, :] for x in inputs[key]]
            # import pdb; pdb.set_trace()
            if loc_att_mode == 'only_struc':
                single_data_perted = GMDataset.pert_struc(single_data)
            elif loc_att_mode == 'both':
                single_data_perted = GMDataset.pert_both(single_data)
            else:
                raise ValueError('wrong locality attack mode. Optional: only_struc|both')
            single_data_list.append(single_data_perted)
        # import pdb; pdb.set_trace()
        ret = collate_fn(single_data_list, univ_size_reuse=True)
        # import pdb; pdb.set_trace()
        for key in temp_keys:
            ret[key] = inputs[key]
        return ret

    @staticmethod
    def pert_struc(inputs):
        n1, n2 = inputs['ns']
        n1, n2 = n1.detach().cpu().numpy(), n2.detach().cpu().numpy()
        # save the perturbed location 
        P1, P2 = inputs['Ps']
        # import pdb; pdb.set_trace();
        P1_att, P2_att = P1.detach().cpu().numpy(), P2.detach().cpu().numpy()
        P1_att, P2_att = P1_att[0:n1, :], P2_att[0:n2, :]
        # recover the original location
        inputs['Ps'] = [x[0:n1, :].clone().detach().cpu().numpy() for x in inputs['Ps_ori']]
        P1, P2 = inputs['Ps']
        # import pdb; pdb.set_trace();
        A1, G1, H1, e1 = build_graphs(P1_att, n1, stg=cfg.GRAPH.SRC_GRAPH_CONSTRUCT, sym=cfg.GRAPH.SYM_ADJACENCY)
        # if cfg.GRAPH.TGT_GRAPH_CONSTRUCT == 'same':
        #     G2 = perm_mat.transpose().dot(G1)
        #     H2 = perm_mat.transpose().dot(H1)
        #     A2 = G2.dot(H2.transpose())
        #     e2 = e1
        # else:
        A2, G2, H2, e2 = build_graphs(P2_att, n2, stg=cfg.GRAPH.TGT_GRAPH_CONSTRUCT, sym=cfg.GRAPH.SYM_ADJACENCY)

        pyg_graph1 = GMDataset.to_pyg_graph(A1, P1)
        pyg_graph2 = GMDataset.to_pyg_graph(A2, P2)
        inputs['es'] = [torch.tensor(x) for x in [e1, e2]]
        inputs['Gs'] =  [torch.Tensor(x) for x in [G1, G2]]
        inputs['Hs'] = [torch.Tensor(x) for x in [H1, H2]]
        inputs['As'] = [torch.Tensor(x) for x in [A1, A2]]
        inputs['pyg_graphs'] = [pyg_graph1, pyg_graph2]

        return copy.deepcopy(inputs)

    @staticmethod
    def pert_both(inputs):
        n1, n2 = inputs['ns']
        n1, n2 = n1.detach().cpu().numpy(), n2.detach().cpu().numpy()
        # inputs['Ps'] = [x[:n1, :] for x in inputs['Ps']]
        P1, P2 = inputs['Ps']
        P1, P2 = P1[:n1, :], P2[:n2, :]
        P1, P2 = P1.detach().cpu().numpy(), P2.detach().cpu().numpy()
        inputs['Ps'] = [torch.Tensor(x) for x in [P1, P2]]
        # import pdb; pdb.set_trace()
        # P1_att, P2_att = inputs['Ps_att']
        # P1_att, P2_att = P1.numpy(), P2.numpy()
        # P1_att, P2_att = P1_att[0:n1, :], P2_att[0:n2, :]
        # perm_mat = inputs['gt_perm_mat']
        # n1, n2 = len(P1), len(P2)
        A1, G1, H1, e1 = build_graphs(P1, n1, stg=cfg.GRAPH.SRC_GRAPH_CONSTRUCT, sym=cfg.GRAPH.SYM_ADJACENCY)
        # if cfg.GRAPH.TGT_GRAPH_CONSTRUCT == 'same':
        #     G2 = perm_mat.transpose().dot(G1)
        #     H2 = perm_mat.transpose().dot(H1)
        #     A2 = G2.dot(H2.transpose())
        #     e2 = e1
        # else:
        A2, G2, H2, e2 = build_graphs(P2, n2, stg=cfg.GRAPH.TGT_GRAPH_CONSTRUCT, sym=cfg.GRAPH.SYM_ADJACENCY)

        # import pdb; pdb.set_trace()

        pyg_graph1 = GMDataset.to_pyg_graph(A1, P1)
        pyg_graph2 = GMDataset.to_pyg_graph(A2, P2)
        inputs['es'] = [torch.tensor(x) for x in [e1, e2]]
        inputs['Gs'] =  [torch.Tensor(x) for x in [G1, G2]]
        inputs['Hs'] = [torch.Tensor(x) for x in [H1, H2]]
        inputs['As'] = [torch.Tensor(x) for x in [A1, A2]]
        inputs['pyg_graphs'] = [pyg_graph1, pyg_graph2]

        return copy.deepcopy(inputs)

class QAPDataset(Dataset):
    def __init__(self, name, length, pad=16, cls=None, **args):
        self.name = name
        # TODO: the function of eval? 
        # import QAPLIB class and use eval to load this class and initialize it? 
        # import pdb; pdb.set_trace()
        self.ds = eval(self.name)(**args, cls=cls)
        self.classes = self.ds.classes
        self.cls = None if cls == 'none' else cls
        self.length = length

    def get_cls_size(self):
        return len(self.ds.data_list)

    def __len__(self):
        #return len(self.ds.data_list)
        return self.length

    def __getitem__(self, idx):
        Fi, Fj, perm_mat, sol, name = self.ds.get_pair(idx % len(self.ds.data_list))
        if perm_mat.size <= 2 * 2 or perm_mat.size >= cfg.PROBLEM.MAX_PROB_SIZE > 0:
            return self.__getitem__(random.randint(0, len(self) - 1))

        #if np.max(ori_aff_mat) > 0:
        #    norm_aff_mat = ori_aff_mat / np.mean(ori_aff_mat)
        #else:
        #    norm_aff_mat = ori_aff_mat

        ret_dict = {'Fi': Fi,
                    'Fj': Fj,
                    'gt_perm_mat': perm_mat,
                    'ns': [torch.tensor(x) for x in perm_mat.shape],
                    'solution': torch.tensor(sol),
                    'name': name,
                    'univ_size': [torch.tensor(x) for x in perm_mat.shape],}

        return ret_dict


def collate_fn(data: list, univ_size_reuse=False):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """
    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            #pad_pattern = torch.from_numpy(np.asfortranarray(pad_pattern))
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, 'constant', 0))

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Keys mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == pyg.data.Data:
            ret = pyg.data.Batch.from_data_list(inp)
        elif type(inp[0]) == str:
            ret = inp
        else:
            raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    ret = stack(data)

    # compute CPU-intensive Kronecker product here to leverage multi-processing nature of dataloader
    if 'Gs' in ret and 'Hs' in ret:
        if cfg.PROBLEM.TYPE == '2GM' and len(ret['Gs']) == 2 and len(ret['Hs']) == 2:
            G1, G2 = ret['Gs']
            H1, H2 = ret['Hs']
            if cfg.FP16:
                sparse_dtype = np.float16
            else:
                sparse_dtype = np.float32
            K1G = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(G2, G1)]  # 1 as source graph, 2 as target graph
            K1H = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(H2, H1)]
            K1G = CSRMatrix3d(K1G)
            K1H = CSRMatrix3d(K1H).transpose()

            ret['KGHs'] = K1G, K1H
        elif cfg.PROBLEM.TYPE in ['MGM', 'MGMC'] and 'Gs_tgt' in ret and 'Hs_tgt' in ret:
            ret['KGHs'] = dict()
            for idx_1, idx_2 in product(range(len(ret['Gs'])), repeat=2):
                # 1 as source graph, 2 as target graph
                G1 = ret['Gs'][idx_1]
                H1 = ret['Hs'][idx_1]
                G2 = ret['Gs_tgt'][idx_2]
                H2 = ret['Hs_tgt'][idx_2]
                if cfg.FP16:
                    sparse_dtype = np.float16
                else:
                    sparse_dtype = np.float32
                KG = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(G2, G1)]
                KH = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(H2, H1)]
                KG = CSRMatrix3d(KG)
                KH = CSRMatrix3d(KH).transpose()
                ret['KGHs']['{},{}'.format(idx_1, idx_2)] = KG, KH
        else:
            raise ValueError('Data type not understood.')

    if 'Fi' in ret and 'Fj' in ret:
        Fi = ret['Fi']
        Fj = ret['Fj']
        # Fi.requires_grad()
        # Fj.requires_grad()
        aff_mat = kronecker_torch(Fj, Fi)
        ret['aff_mat'] = aff_mat

    ret['batch_size'] = len(data)
    if not univ_size_reuse:
        ret['univ_size'] = torch.tensor([max(*[item[b] for item in ret['univ_size']]) for b in range(ret['batch_size'])])

    for v in ret.values():
        if type(v) is list:
            ret['num_graphs'] = len(v)
            break

    return ret


def worker_init_fix(worker_id):
    """
    Init dataloader workers with fixed seed.
    """
    random.seed(cfg.RANDOM_SEED + worker_id)
    np.random.seed(cfg.RANDOM_SEED + worker_id)


def worker_init_rand(worker_id):
    """
    Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % 2 ** 32)


def get_dataloader(dataset, fix_seed=True, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.DATALOADER_NUM, collate_fn=collate_fn,
        pin_memory=False, worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )