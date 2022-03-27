import sys
pth = '/home/jiangshaofei/.local/lib/python3.7/site-packages'
if pth in sys.path:
    sys.path.remove(pth)

import torch
import numpy as np
from pathlib import Path

from src.dataset.data_loader import GMDataset, get_dataloader
from src.utils.model_sl import load_model
from src.parallel import DataParallel
from src.lap_solvers.hungarian import hungarian
from src.utils.data_to_cuda import data_to_cuda
import matplotlib
try:
    import _tkinter
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

plt.rcParams["font.family"] = "serif"

from src.utils.config import cfg

def vertical_subplt(a,b,c):
    plt.subplot(b, a, (c // b) + c % b * a)

def visualize_model(models, dataloader, device, num_images=6, set='test', cls=None, save_img=False):
    print('Visualizing model...')
    assert set in ('train', 'test')

    for model in models:
        model.eval()
    images_so_far = 0

    #names = ['source', 'GMN', 'PCA-GM', 'IPCA-GM']
    #names = ['source', 'GMN', 'PCA-GM', 'NGM', 'NGM-v2', 'NHGM-v2']
    names = ['source', 'GMN', 'PCA-GM', 'CIE', 'GANN-GM', 'GANN-MGM']
    num_cols = num_images #// 2 #+ 1

    old_cls = dataloader[set].dataset.cls
    if cls is not None:
        dataloader[set].dataset.cls = cls

    visualize_path = Path(cfg.OUTPUT_PATH) / 'visual'
    if save_img:
        if not visualize_path.exists():
            visualize_path.mkdir(parents=True)

    for cls in range(len(cfg.PascalVOC.CLASSES)):
        fig = plt.figure(figsize=(30, 30), dpi=120)
        dataloader[set].dataset.cls = cls

        print('class: {}'.format(cfg.PascalVOC.CLASSES[cls]))
        images_so_far = 0
        # import pdb; pdb.set_trace()
        for i, inputs in enumerate(dataloader[set]):
            if models[0].module.device != torch.device('cpu'):
                inputs = data_to_cuda(inputs)
            assert 'images' in inputs
            two_gm_inputs = {}
            for key, val in inputs.items():
                if key == 'gt_perm_mat':
                    # import pdb; pdb.set_trace()
                    two_gm_inputs[key] = val
                    # two_gm_inputs[key] = torch.bmm(val[0], val[1].transpose(1, 2))
                elif key == 'KGHs':
                    two_gm_inputs[key] = val
                    # two_gm_inputs[key] = val['0,1']
                elif key == 'num_graphs':
                    two_gm_inputs[key] = 2
                else:
                    if isinstance(val, list) and len(val) > 2:
                        two_gm_inputs[key] = val[0], val[1]
                    else:
                        two_gm_inputs[key] = val

            data1, data2 = two_gm_inputs['images']
            P1_gt, P2_gt = two_gm_inputs['Ps']
            n1_gt, n2_gt = two_gm_inputs['ns']
            perm_mat = two_gm_inputs['gt_perm_mat']

            pred_perms = []
            for model_id, model in enumerate(models):
                # if model_id != len(models) - 1:
                cfg.PROBLEM.TYPE = '2GM'
                outputs = model(two_gm_inputs)
                pred_perms.append(outputs['perm_mat'])
                # else:
                #     cfg.PROBLEM.TYPE = 'MGM'
                #     outputs = model(inputs)
                #     for (idx1, idx2), pred_perm in zip(outputs['graph_indices'], outputs['perm_mat_list']):
                #         if idx1 == 0 and idx2 == 1:
                #             pred_perms.append(pred_perm)
                #             break
                #         elif idx1 == 1 and idx2 == 0:
                #             pred_perms.append(pred_perm.transpose(1, 2))
                #             break
            # import pdb; pdb.set_trace()
            for j in range(inputs['batch_size']):
                if n1_gt[j] <= 4:
                    print('graph too small.')
                    continue

                matched = []
                for idx, pred_perm in enumerate(pred_perms):
                    matched_num = torch.sum(pred_perm[j, :n1_gt[j], :n2_gt[j]] * perm_mat[j, :n1_gt[j], :n2_gt[j]])
                    matched.append(matched_num)
                # import pdb; pdb.set_trace()
                #if random.choice([0, 1, 2]) >= 1:
                # if not (matched[4] >= matched[3] >= matched[1] > matched[0]):
                #         print('performance not good.')
                #         continue

                images_so_far += 1
                print(chr(13) + 'Visualizing {:4}/{:4}'.format(images_so_far, num_images))  # chr(13)=CR

                colorset = np.random.rand(n1_gt[j], 3)
                #ax = plt.subplot(1 + len(s_pred_perms), num_cols, images_so_far + 1)
                #ax.axis('off')
                #plt.title('source')
                #plot_helper(data1[j], P1_gt[j], n1_gt[j], ax, colorset)

                for idx, pred_perm in enumerate(pred_perms):
                    ax = plt.subplot(len(pred_perms), num_cols, idx * num_cols + images_so_far)
                    #if images_so_far > num_cols:
                    #    ax = plt.subplot(len(pred_perms) * 2, num_cols, (idx + len(pred_perms)) * num_cols + images_so_far - num_cols)
                    #else:
                    #    ax = plt.subplot(len(pred_perms) * 2, num_cols, idx * num_cols + images_so_far)
                    ax.axis('off')
                    #plt.title('predict')
                    #plot_helper(data2[j], P2_gt[j], n1_gt[j], ax, colorset, 'tgt', s_pred_perm[j], perm_mat[j])
                    plot_2graph_helper(data1[j], data2[j], P1_gt[j], P2_gt[j], n1_gt[j], n2_gt[j], ax, colorset, pred_perm[j], perm_mat[j], cls, names[idx+1])

                #ax = plt.subplot(2 + len(s_pred_perms), num_images + 1, (len(s_pred_perms) + 1) * num_images + images_so_far)
                #ax.axis('off')
                #plt.title('groundtruth')
                #plot_helper(data2[j], P2_gt[j], n1_gt[j], ax, colorset, 'tgt', perm_mat[j])

                if not save_img:
                    plt.show()
                    print("Press Enter to continue...", end='', flush=True)  # prevent new line
                    input()

                if images_so_far >= num_images:
                    fig.savefig(str(visualize_path / '{}_{:0>4}.jpg'.format(dataloader[set].dataset.cls, images_so_far)), bbox_inches='tight')
                    break

                dataloader[set].dataset.cls += 1
            if images_so_far >= num_images:
                break

    dataloader[set].dataset.cls = old_cls


def plot_helper(img, P, n, ax, colorset, mode='src', pmat=None, gt_pmat=None):
    assert mode in ('src', 'tgt')
    if mode == 'tgt':
        assert pmat is not None
    img = tensor2np(img.cpu())
    plt.imshow(img)

    P = P.cpu().numpy()
    if mode == 'src':
        for i in range(n):
            mark = plt.Circle(P[i], 7, edgecolor='w', facecolor=colorset[i])
            ax.add_artist(mark)
    else:
        pmat = pmat.cpu().numpy()
        gt_pmat = gt_pmat.cpu().numpy()
        idx = np.argmax(pmat, axis=-1)
        idx_gt = np.argmax(gt_pmat, axis=-1)
        matched = 0
        for i in range(n):
            mark = plt.Circle(P[idx[i]], 7, edgecolor='w' if idx[i] == idx_gt[i] else 'r', facecolor=colorset[i])
            ax.add_artist(mark)
            if idx[i] == idx_gt[i]:
                matched += 1
        plt.title('{:d}/{:d}'.format(matched, n), y=-0.2, fontsize=25)

def plot_2graph_helper(imgsrc, imgtgt, Psrc, Ptgt, nsrc, ntgt, ax, colorset, pmat, gt_pmat, cls, method=""):
    imgcat = torch.cat((imgsrc, imgtgt), dim=2)
    imgcat = tensor2np(imgcat.cpu())
    plt.imshow(imgcat)

    Psrc = Psrc.cpu().numpy()
    Ptgt = Ptgt.cpu().numpy()
    Ptgt[:, 0] += imgsrc.shape[2]
    pmat = pmat.cpu().numpy()
    gt_pmat = gt_pmat.cpu().numpy()
    matched = 0
    for i in range(nsrc):
        if pmat.sum(axis=-1)[i] == 0:
            mark = plt.Circle(Psrc[i], 7, edgecolor='b', facecolor="None")
            ax.add_artist(mark)
    for j in range(ntgt):
        if pmat.sum(axis=-2)[j] == 0:
            mark = plt.Circle(Ptgt[j], 7, edgecolor='b', facecolor="None")
            ax.add_artist(mark)
    for i in range(nsrc):
        for j in range(ntgt):
            if pmat[i, j] == 1:
                # src
                mark = plt.Circle(Psrc[i], 7, edgecolor='g' if gt_pmat[i, j] == 1 else 'r', facecolor="None")
                ax.add_artist(mark)
                #tgt
                mark = plt.Circle(Ptgt[j], 7, edgecolor='g' if gt_pmat[i, j] == 1  else 'r', facecolor="None")
                ax.add_artist(mark)
                l = matplotlib.lines.Line2D([Psrc[i][0], Ptgt[j][0]], [Psrc[i][1], Ptgt[j][1]], color='g' if gt_pmat[i, j] == 1 else 'r')
                ax.add_line(l)
                if gt_pmat[i, j] == 1:
                    matched += 1
    # import pdb; pdb.set_trace()
    plt.title('{} {}: {}/{}'.format(method, cfg.PascalVOC.CLASSES[cls], matched, round(gt_pmat.sum())), y=-0.3, fontsize=20)
    # plt.title('{} {}: {:d}/{:d}'.format(method, cfg.PascalVOC.CLASSES['test'][dataloader['test'].dataset.cls], matched, round(gt_pmat.sum())), y=-0.3, fontsize=20)


def tensor2np(inp):
    """Tensor to numpy array for plotting"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(cfg.NORM_MEANS)
    std = np.array(cfg.NORM_STD)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


if __name__ == '__main__':
    from src.utils.parse_args import parse_args
    args = parse_args('Deep learning of graph matching visualization code.')

    import importlib
    from src.utils.config import cfg_from_file

    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    image_dataset = {
        x: GMDataset(cfg.DATASET_FULL_NAME,
                     sets=x,
                     problem=cfg.PROBLEM.TYPE,
                     length=dataset_len[x],
                     cls=cfg.TRAIN.CLASS if x == 'train' else cfg.EVAL.CLASS,
                     obj_resize=cfg.PROBLEM.RESCALE)
        for x in ('train', 'test')}
    cfg.DATALOADER_NUM = 0
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == 'test'))
        for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_paths = [#'/mnt/nas/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_gmn_imcpt.pt',
                   #'/mnt/nas/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_pca_imcpt.pt',
                   #'/mnt/nas/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_cie_imcpt.pt',
                   #'/mnt/nas/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_gann-gm_imcpt.pt',
                   #'/mnt/nas/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_gann-mgm_imcpt.pt',
                   '/mnt/nas/home/jiangshaofei/RobustMatch/pretrained_model/pretrained_params_vgg16_ngm_voc.pt',
                   ]

    cfg_files = [#'experiments/vgg16_gmn_imcpt.yaml',
                 #'experiments/vgg16_pca_imcpt.yaml',
                 #'experiments/vgg16_cie_imcpt.yaml',
                 #'experiments/vgg16_gann-gm_imcpt.yaml',
                 #'experiments/vgg16_gann-mgm_imcpt.yaml',
                 'experiments/vgg16_ngm_voc_attack.yaml',
                 ]
    models = []

    for i, (model_path, cfg_file) in enumerate(zip(model_paths, cfg_files)):
        cfg_from_file(cfg_file)

        mod = importlib.import_module(cfg.MODULE)
        Net = mod.Net

        model = Net()
        model = model.to(device)
        model = DataParallel(model, device_ids=cfg.GPUS)

        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)
        models.append(model)

    visualize_model(models, dataloader, device,
                    num_images=cfg.VISUAL.NUM_IMGS,
                    cls=cfg.VISUAL.CLASS if cfg.VISUAL.CLASS != 'none' else None,
                    save_img=cfg.VISUAL.SAVE)

59 104