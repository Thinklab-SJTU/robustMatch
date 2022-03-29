import time
from datetime import datetime
from pathlib import Path
import xlwt
import numpy as np
import logging

from src.dataset.data_loader import GMDataset, get_dataloader
from src.evaluation_metric import *
from src.parallel import DataParallel
from src.utils.model_sl import load_model
from src.utils.data_to_cuda import data_to_cuda
from src.utils.timer import Timer

from src.utils.config import cfg

from attack_utils import AttackGM, BlackAttackGM
from src.loss_func import *
from src.dataset.pascal_voc import KPT_NAMES

def eval_util(model, wb, dataloader, device=None):
    if device is None:
        device = next(model.parameters()).device
    # FGSM, PGD-10
    for obj_type in ['pixel', 'pos+struc', 'pixel+pos+struc']:
        xls_sheet = wb.add_sheet(obj_type)

        criterion_att = GMLoss('perm', '2GM')

        print('FGSM ATTACK with ' + obj_type)
        eval_att = AttackGM(obj_type, 'fgsm', 
                            criterion=criterion_att, 
                            eps=(cfg.ATTACK.EPSILON_FEATURE, cfg.ATTACK.EPSILON_LOCALITY), 
                            iter_num=1, 
                            alpha=1., 
                            device=device, 
                            inv=False)
        accs = eval_model(model, dataloader['test'], criterion=criterion_att, attack=eval_att, xls_sheet=xls_sheet, xls_row=1)

        print('PGD-10 ATTACK with ' + obj_type)
        eval_att = AttackGM(obj_type, 'pgd', 
                    criterion=criterion_att, 
                    eps=(cfg.ATTACK.EPSILON_FEATURE, cfg.ATTACK.EPSILON_LOCALITY), 
                    iter_num=10,
                    alpha=0.25,
                    device=device, 
                    inv=False)
        accs = eval_model(model, dataloader['test'], criterion=criterion_att, attack=eval_att, xls_sheet=xls_sheet, xls_row=2)

    # PGD-50 combo attack
    print('PGD-50 ATTACK with combo' )
    xls_sheet = wb.add_sheet('pgd50')
    eval_att = AttackGM('pixel+pos+struc', 'pgd50', 
                    criterion=criterion_att, 
                    eps=(cfg.ATTACK.EPSILON_FEATURE, cfg.ATTACK.EPSILON_LOCALITY), 
                    iter_num=50, 
                    alpha=cfg.ATTACK.EVAL_ALPHA, 
                    device=device, 
                    inv=False)
    accs = eval_model(model, dataloader['test'], criterion=criterion_att, attack=eval_att, xls_sheet=xls_sheet, xls_row=1)
    return

def evaluation(model, wb, dataloader, device=None):
    if device is None:
        device = next(model.parameters()).device
    if len(cfg.PRETRAINED_PATH) > 0:
        model_path = cfg.PRETRAINED_PATH
    if len(model_path) > 0:
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path, strict=False)

    if cfg.EVAL.MODE == 'clean':
        xls_sheet = wb.add_sheet('clean')
        accs = eval_model(model, dataloader['test'], xls_sheet=xls_sheet)
    elif cfg.EVAL.MODE == 'single':
        xls_sheet = wb.add_sheet(cfg.ATTACK.OBJ_TYPE)
        criterion_att = GMLoss(cfg.ATTACK.LOSS_FUNC.lower(), cfg.PROBLEM.TYPE)
        eval_att = AttackGM(cfg.ATTACK.OBJ_TYPE, cfg.ATTACK.TYPE, 
                            criterion=criterion_att, 
                            eps=(cfg.ATTACK.EPSILON_FEATURE, cfg.ATTACK.EPSILON_LOCALITY), 
                            iter_num=cfg.ATTACK.EVAL_STEP,
                            alpha=cfg.ATTACK.EVAL_ALPHA, 
                            device=device, 
                            inv=False)
        accs = eval_model(model, dataloader['test'], criterion=criterion_att, attack=eval_att, xls_sheet=xls_sheet)
    elif cfg.EVAL.MODE == 'all':
        eval_util(model, wb, dataloader, device)

def eval_model(model, dataloader, criterion=None, attack=None, verbose=False, xls_sheet=None, xls_row=None):
    print('Start evaluation...')
    since = time.time()

    device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    ds = dataloader.dataset
    classes = ds.classes

    recalls = []
    precisions = []
    f1s = []
    pred_time = []
    objs = torch.zeros(len(classes), device=device)
    cluster_acc = []
    cluster_purity = []
    cluster_ri = []

    timer = Timer()
    neg_gt_obj = []
    gt_obj = [ [] for i in range(len(classes))]

    for i, cls in enumerate(classes):
        if verbose:
            print('Evaluating class {}: {}/{}'.format(cls, i, len(classes)))

        neg_gt_obj.append(False)

        running_since = time.time()
        iter_num = 0

        ds.cls = cls
        recall_list = []
        precision_list = [] 
        f1_list = []
        pred_time_list = []
        obj_total_num = torch.zeros(1, device=device)
        cluster_acc_list = []
        cluster_purity_list = []
        cluster_ri_list = []

        for inputs in dataloader:
            if model.module.device != torch.device('cpu'):
                inputs = data_to_cuda(inputs)
            batch_num = inputs['batch_size']

            iter_num = iter_num + 1

            with torch.set_grad_enabled(False):
                timer.tick()

                if attack is not None:
                    if isinstance(attack, BlackAttackGM):
                        inputs = attack(inputs, criterion)
                    else:
                        inputs, _ = attack(model, inputs, criterion)

                outputs = model(inputs)
                pred_time_list.append(torch.full((batch_num,), timer.toc() / batch_num))

            # Evaluate matching accuracy
            if cfg.PROBLEM.TYPE == '2GM':
                assert 'perm_mat' in outputs
                assert 'gt_perm_mat' in outputs

                recall = matching_recall(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'][0])
                recall_list.append(recall)
                precision = matching_precision(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'][0])
                precision_list.append(precision)
                f1 = 2 * (precision * recall) / (precision + recall)
                f1[torch.isnan(f1)] = 0
                f1_list.append(f1)

                if 'aff_mat' in outputs:
                    pred_obj_score = objective_score(outputs['perm_mat'], outputs['aff_mat'])
                    gt_obj_score = objective_score(outputs['gt_perm_mat'], outputs['aff_mat'])
                    objs[i] += torch.sum(pred_obj_score / gt_obj_score)
                    obj_total_num += batch_num
                    
                    # GT QAP score research
                    gt_obj[i] += gt_obj_score.cpu().detach().numpy().tolist()
                    if not neg_gt_obj[i]:
                        neg_gt_obj[i] = min(gt_obj_score) < 0
                        if neg_gt_obj[i]:
                            logging.warning('GT_obj_score : {}'.format(min(gt_obj_score).cpu().numpy()))
                    
            elif cfg.PROBLEM.TYPE in ['MGM', 'MGMC']:
                assert 'graph_indices' in outputs
                assert 'perm_mat_list' in outputs
                assert 'gt_perm_mat_list' in outputs

                ns = outputs['ns']
                for x_pred, x_gt, (idx_src, idx_tgt) in \
                        zip(outputs['perm_mat_list'], outputs['gt_perm_mat_list'], outputs['graph_indices']):
                    recall = matching_recall(x_pred, x_gt, ns[idx_src])
                    recall_list.append(recall)
                    precision = matching_precision(x_pred, x_gt, ns[idx_src])
                    precision_list.append(precision)
                    f1 = 2 * (precision * recall) / (precision + recall)
                    f1[torch.isnan(f1)] = 0
                    f1_list.append(f1)
            else:
                raise ValueError('Unknown problem type {}'.format(cfg.PROBLEM.TYPE))

            # Evaluate clustering accuracy
            if cfg.PROBLEM.TYPE == 'MGMC':
                assert 'pred_cluster' in outputs
                assert 'cls' in outputs

                pred_cluster = outputs['pred_cluster']
                cls_gt_transpose = [[] for _ in range(batch_num)]
                for batched_cls in outputs['cls']:
                    for b, _cls in enumerate(batched_cls):
                        cls_gt_transpose[b].append(_cls)
                cluster_acc_list.append(clustering_accuracy(pred_cluster, cls_gt_transpose))
                cluster_purity_list.append(clustering_purity(pred_cluster, cls_gt_transpose))
                cluster_ri_list.append(rand_index(pred_cluster, cls_gt_transpose))

            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                print('Class {} Iteration {:<4} {:>4.2f}sample/s'.format(cls, iter_num, running_speed))
                running_since = time.time()


        recalls.append(torch.cat(recall_list))
        precisions.append(torch.cat(precision_list))
        f1s.append(torch.cat(f1_list))
        objs[i] = objs[i] / obj_total_num
        pred_time.append(torch.cat(pred_time_list))
        if cfg.PROBLEM.TYPE == 'MGMC':
            cluster_acc.append(torch.cat(cluster_acc_list))
            cluster_purity.append(torch.cat(cluster_purity_list))
            cluster_ri.append(torch.cat(cluster_ri_list))

        if verbose:
            print('Class {} {}'.format(cls, format_accuracy_metric(precisions[i], recalls[i], f1s[i])))
            print('Class {} norm obj score = {:.4f}'.format(cls, objs[i]))
            if neg_gt_obj[i]:
                logging.warning('The obj score metric in class {} is not reliable due to negative ground truth obj score detected'.format(cls))
            print('Class {} pred time = {}s'.format(cls, format_metric(pred_time[i])))

            if cfg.PROBLEM.TYPE == 'MGMC':
                print('Class {} cluster acc={}'.format(cls, format_metric(cluster_acc[i])))
                print('Class {} cluster purity={}'.format(cls, format_metric(cluster_purity[i])))
                print('Class {} cluster rand index={}'.format(cls, format_metric(cluster_ri[i])))

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)

    exp_setup_list = ['type', 'step', 'eps_feature', 'eps_locality']
    xls_col = len(exp_setup_list)
    if xls_sheet and attack is None:
        for idx, str in enumerate(exp_setup_list):
            xls_sheet.write(0, idx, str)
        for idx, cls in enumerate(classes):
            xls_sheet.write(0, idx+xls_col, cls)
        xls_sheet.write(0, idx+xls_col+1, 'mean')

    if xls_row is None:
        xls_row = 1
    # show result
    print('Matching accuracy')
    if xls_sheet:
        if attack is None:
            exp_setup_stats = ['baseline', 0, 0, 0]
        else:
            exp_setup_stats = [cfg.ATTACK.OBJ_TYPE+cfg.ATTACK.TYPE, cfg.ATTACK.EVAL_STEP, cfg.ATTACK.EPSILON_FEATURE, cfg.ATTACK.EPSILON_LOCALITY]
        for idx, str in enumerate(exp_setup_stats):
            xls_sheet.write(xls_row, idx, str)

    for idx, (cls, cls_p, cls_r, cls_f1) in enumerate(zip(classes, precisions, recalls, f1s)):
        print('{}: {}'.format(cls, format_accuracy_metric(cls_p, cls_r, cls_f1)))
        if xls_sheet:
            xls_sheet.write(xls_row, idx+xls_col, torch.mean(cls_p).item())
    print('average accuracy: {}'.format(format_accuracy_metric(torch.cat(precisions), torch.cat(recalls), torch.cat(f1s))))
    if xls_sheet:
        xls_sheet.write(xls_row, idx+xls_col+1, torch.mean(torch.cat(precisions)).item())
        xls_row += 1

    if not torch.any(torch.isnan(objs)):
        print('Normalized objective score')
    #     if xls_sheet: xls_sheet.write(xls_row, 0, 'norm objscore')
        for idx, (cls, cls_obj) in enumerate(zip(classes, objs)):
            print('{} = {:.4f}'.format(cls, cls_obj))
    #         if xls_sheet: xls_sheet.write(xls_row, idx+1, cls_obj.item())
        print('average objscore = {:.4f}'.format(torch.mean(objs)))
    #     if xls_sheet:
    #         xls_sheet.write(xls_row, idx+2, torch.mean(objs).item())
    #         xls_row += 1

    if cfg.PROBLEM.TYPE == 'MGMC':
        print('Clustering accuracy')
        if xls_sheet: xls_sheet.write(xls_row, 0, 'cluster acc')
        for idx, (cls, cls_acc) in enumerate(zip(classes, cluster_acc)):
            print('{} = {}'.format(cls, format_metric(cls_acc)))
            if xls_sheet: xls_sheet.write(xls_row, idx+1, torch.mean(cls_acc).item())
        print('average clustering accuracy = {}'.format(format_metric(torch.cat(cluster_acc))))
        if xls_sheet:
            xls_sheet.write(xls_row, idx+2, torch.mean(torch.cat(cluster_acc)).item())
            xls_row += 1

        print('Clustering purity')
        if xls_sheet: xls_sheet.write(xls_row, 0, 'cluster purity')
        for idx, (cls, cls_acc) in enumerate(zip(classes, cluster_purity)):
            print('{} = {}'.format(cls, format_metric(cls_acc)))
            if xls_sheet: xls_sheet.write(xls_row, idx+1, torch.mean(cls_acc).item())
        print('average clustering purity = {}'.format(format_metric(torch.cat(cluster_purity))))
        if xls_sheet:
            xls_sheet.write(xls_row, idx+2, torch.mean(torch.cat(cluster_purity)).item())
            xls_row += 1

        print('Clustering rand index')
        if xls_sheet: xls_sheet.write(xls_row, 0, 'rand index')
        for idx, (cls, cls_acc) in enumerate(zip(classes, cluster_ri)):
            print('{} = {}'.format(cls, format_metric(cls_acc)))
            if xls_sheet: xls_sheet.write(xls_row, idx+1, torch.mean(cls_acc).item())
        print('average rand index = {}'.format(format_metric(torch.cat(cluster_ri))))
        if xls_sheet:
            xls_sheet.write(xls_row, idx+2, torch.mean(torch.cat(cluster_ri)).item())
            xls_row += 1        

    return torch.Tensor(list(map(torch.mean, recalls)))

if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args
    from src.utils.print_easydict import print_easydict
    from src.utils.count_model_params import count_parameters

    args = parse_args('Deep learning of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    image_dataset = GMDataset(cfg.DATASET_FULL_NAME,
                              sets='test',
                              problem=cfg.PROBLEM.TYPE,
                              length=cfg.EVAL.SAMPLES,
                              cls=cfg.EVAL.CLASS,
                              #cls='chair',
                              obj_resize=cfg.PROBLEM.RESCALE)
    dataloader = get_dataloader(image_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)
    model = DataParallel(model, device_ids=cfg.GPUS)

    if cfg.ATTACK.BLACK:
        print('BlackBoxAttack' + cfg.ATTACK.TYPE + ' from ' + cfg.VICTIM_MODEL_NAME + ' to ' + cfg.MODEL_NAME)
        criterion = GMLoss(cfg.ATTACK.LOSS_FUNC.lower(), cfg.PROBLEM.TYPE)
        attack =  BlackAttackGM(cfg.ATTACK.OBJ_TYPE, cfg.ATTACK.TYPE, criterion, device)
    else:
        attack = None
        criterion = None

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    wb = xlwt.Workbook()
    ws = wb.add_sheet('epoch{}'.format(cfg.EVAL.EPOCH))
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        print('Number of parameters: {:.2f}M'.format(count_parameters(model) / 1e6))

        model_path = ''
        if cfg.EVAL.EPOCH is not None and cfg.EVAL.EPOCH > 0:
            model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(cfg.EVAL.EPOCH))
        if len(cfg.PRETRAINED_PATH) > 0:
            model_path = cfg.PRETRAINED_PATH
        if len(model_path) > 0:
            print('Loading model parameters from {}'.format(model_path))
            load_model(model, model_path, strict=False)

        eval_model(
            model, dataloader,
            criterion=criterion, attack=attack,
            verbose=True,
            xls_sheet=ws,
        )
    wb.save(str(Path(cfg.OUTPUT_PATH) / ('eval_result_' + now_time + '.xls')))
