import torch.optim as optim
import time
import xlwt
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from tensorboardX import SummaryWriter

from src.dataset.data_loader import GMDataset, get_dataloader
from src.build_graphs import build_graphs
from src.displacement_layer import Displacement
from src.loss_func import *
from src.evaluation_metric import matching_recall
from src.parallel import DataParallel
from src.utils.model_sl import load_model, save_model
from eval import eval_model, eval_util, evaluation
from src.utils.data_to_cuda import data_to_cuda, cuda_copy

from src.utils.config import cfg

from attack_utils import AttackGM

def train_eval_model(model,
                     criterion,
                     optimizer,
                     dataloader,
                     tfboard_writer,
                     num_epochs=25,
                     start_epoch=0,
                     attacks=None,
                     xls_wb=None,
                     criterion_burnin=None):
    print('Start training...')

    since = time.time()
    dataset_size = len(dataloader['train'].dataset)
    displacement = Displacement()

    device = next(model.parameters()).device
    print('model on device: {}'.format(device))

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    model_path, optim_path = '',''
    if start_epoch > 0:
        model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
        optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
    if len(cfg.PRETRAINED_PATH) > 0:
        model_path = cfg.PRETRAINED_PATH
    if len(model_path) > 0:
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path, strict=False)
    if len(optim_path) > 0:
        print('Loading optimizer state from {}'.format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg.TRAIN.LR_STEP,
                                               gamma=cfg.TRAIN.LR_DECAY,
                                               last_epoch=cfg.TRAIN.START_EPOCH - 1)

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        if epoch == cfg.TRAIN.BURN_IN_PERIOD:
            print('BURN-IN PERIOD ENDS.')
        print('-' * 10)

        model.train()  # Set model to training mode

        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0
        running_since = time.time()
        iter_num = 0

        # Iterate over data.
        for inputs in dataloader['train']:
            if model.module.device != torch.device('cpu'):
                inputs = data_to_cuda(inputs)
            # start_time = time.time()
            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                inputs_att = dict()
                inputs_att = cuda_copy(inputs, inputs_att)
                if epoch < cfg.TRAIN.BURN_IN_PERIOD:
                    inputs_att, _ = attacks[0](model, inputs_att)
                else:
                    inputs_att, _ = attacks[1](model, inputs_att)

                outputs_att = model(inputs_att)
                    

                if cfg.PROBLEM.TYPE == '2GM':
                    assert 'ds_mat' in outputs_att
                    assert 'perm_mat' in outputs_att
                    assert 'gt_perm_mat' in outputs_att
                    # compute loss

                    if cfg.TRAIN.LOSS_FUNC == 'ourloss':
                        loss = criterion(outputs_att, outputs_att)
                    else:
                        loss = criterion(outputs_att)

                    # compute accuracy
                    acc = matching_recall(outputs_att['perm_mat'], outputs_att['gt_perm_mat'], outputs_att['ns'][0])

                else:
                    raise ValueError('Unknown problem type {}'.format(cfg.PROBLEM.TYPE))

                # backward + optimize
                loss.backward()
                optimizer.step()
                
                batch_num = inputs['batch_size']

                # tfboard writer
                loss_dict = dict()
                loss_dict['loss'] = loss.item()
                tfboard_writer.add_scalars('loss', loss_dict, epoch * cfg.TRAIN.EPOCH_ITERS + iter_num)

                accdict = dict()
                accdict['matching accuracy'] = torch.mean(acc)
                tfboard_writer.add_scalars(
                    'training accuracy',
                    accdict,
                    epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                )

                # statistics
                running_loss += loss.item() * batch_num
                epoch_loss += loss.item() * batch_num

                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                    print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f} Training Acc={:<8.2f}'
                          .format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / batch_num, torch.mean(acc)))
                    tfboard_writer.add_scalars(
                        'speed',
                        {'speed': running_speed},
                        epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    )

                    tfboard_writer.add_scalars(
                        'learning rate',
                        {'lr_{}'.format(i): x['lr'] for i, x in enumerate(optimizer.param_groups)},
                        epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    )

                    running_loss = 0.0
                    running_since = time.time()

        epoch_loss = epoch_loss / dataset_size

        print('Epoch {:<4} Loss: {:.4f} \n'.format(epoch, epoch_loss))
        # Eval Clean Accuracy every n epochs
        if epoch == 0 or (epoch + 1) % cfg.EVAL.NUM_EPOCH == 0:
            xls_sheet = xls_wb.add_sheet('epoch{}'.format(epoch + 1))
            accs = eval_model(model, dataloader['test'], xls_sheet=xls_sheet)
            acc_dict = {"{}".format(cls): single_acc for cls, single_acc in zip(dataloader['test'].dataset.classes, accs)}
            acc_dict['average'] = torch.mean(accs)
            tfboard_writer.add_scalars(
                'Eval acc',
                acc_dict,
                (epoch + 1) * cfg.TRAIN.EPOCH_ITERS
            )

            if epoch > 0:
                save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
                torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))

        scheduler.step()

    # Eval Robustness in the final epoch 
    eval_util(model, xls_wb, dataloader)
    wb.save(wb.__save_path)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args
    from src.utils.print_easydict import print_easydict
    from src.utils.count_model_params import count_parameters

    args = parse_args('Deep learning of graph matching training & evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed(cfg.RANDOM_SEED)

    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    image_dataset = {
        x: GMDataset(cfg.DATASET_FULL_NAME,
                     sets=x,
                     problem=cfg.PROBLEM.TYPE,
                     length=dataset_len[x],
                     cls=cfg.TRAIN.CLASS if x == 'train' else cfg.EVAL.CLASS,
                     obj_resize=cfg.PROBLEM.RESCALE)
        for x in ('train', 'test')}
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == 'test'))
        for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)

    criterion_train = GMLoss(cfg.TRAIN.LOSS_FUNC.lower(), cfg.PROBLEM.TYPE)

    if cfg.TRAIN.SYNC_MINMAX:
        criterion_att = GMLoss(cfg.TRAIN.LOSS_FUNC.lower(), cfg.PROBLEM.TYPE)
    else:
        if criterion_train.name == 'hamming':        
            criterion_att = GMLoss('hamming', cfg.PROBLEM.TYPE)
        else:
            criterion_att = GMLoss(cfg.ATTACK.LOSS_FUNC.lower(), cfg.PROBLEM.TYPE)

    if cfg.TRAIN.MODE == 'eval': 
        pass
    elif cfg.TRAIN.MODE == 'at':
        attack1 = AttackGM(cfg.ATTACK.OBJ_TYPE, cfg.ATTACK.TYPE, 
                            criterion=criterion_att, 
                            eps=(cfg.ATTACK.EPSILON_FEATURE, cfg.ATTACK.EPSILON_LOCALITY), 
                            iter_num=cfg.ATTACK.STEP, 
                            alpha=cfg.ATTACK.ALPHA, 
                            device=device, 
                            inv=False)
        attacks = [attack1, attack1]
    elif cfg.TRAIN.MODE == '2step':
        attack_warm = AttackGM(cfg.ATTACK2.OBJ_TYPE, cfg.ATTACK2.TYPE, 
                                criterion=criterion_att, 
                                eps=(cfg.ATTACK.EPSILON_FEATURE, cfg.ATTACK.EPSILON_LOCALITY), 
                                iter_num=cfg.ATTACK2.STEP, 
                                alpha=cfg.ATTACK.ALPHA, 
                                device=device, 
                                inv=False)
        attack1     = AttackGM(cfg.ATTACK.OBJ_TYPE, cfg.ATTACK.TYPE, 
                                criterion=criterion_att, 
                                eps=(cfg.ATTACK.EPSILON_FEATURE, cfg.ATTACK.EPSILON_LOCALITY), 
                                iter_num=cfg.ATTACK.STEP, 
                                alpha=cfg.ATTACK.ALPHA, 
                                device=device, 
                                inv=False)
        attacks = [attack_warm, attack1]
    else:
        raise NotImplementedError

    if cfg.TRAIN.SEPARATE_BACKBONE_LR:
        backbone_ids = [id(item) for item in model.backbone_params]
        other_params = [param for param in model.parameters() if id(param) not in backbone_ids]

        model_params = [
            {'params': other_params},
            {'params': model.backbone_params, 'lr': cfg.TRAIN.BACKBONE_LR}
        ]
    else:
        model_params = model.parameters()

    if cfg.TRAIN.OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(model_params, lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
    elif cfg.TRAIN.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(model_params, lr=cfg.TRAIN.LR)
    else:
        raise ValueError('Unknown optimizer {}'.format(cfg.TRAIN.OPTIMIZER))

    if cfg.FP16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to enable FP16.")
        model, optimizer = amp.initialize(model, optimizer)

    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH) / 'tensorboard' / 'training_{}'.format(now_time)))
    wb = xlwt.Workbook()
    wb.__save_path = str(Path(cfg.OUTPUT_PATH) / ( now_time + '.xls'))

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / (now_time + '.log'))) as _:
        print_easydict(cfg)
        print('Number of parameters: {:.2f}M'.format(count_parameters(model) / 1e6))
        if cfg.TRAIN.MODE == 'eval':
            evaluation(model, wb, dataloader)
        else:
            model = train_eval_model(model, criterion_train,  optimizer, dataloader, tfboardwriter,
                                    attacks=attacks,
                                    num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                    start_epoch=cfg.TRAIN.START_EPOCH,
                                    xls_wb=wb
                                    )
        
    wb.save(wb.__save_path)