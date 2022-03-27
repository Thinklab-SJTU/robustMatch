import argparse
from src.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_dir_new
from pathlib import Path

def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    ### Universal param
    parser.add_argument('--cfg', '--config', dest='cfg_file', action='append',
                        help='an optional config file', default=None, type=str)
    parser.add_argument('--batch', dest='batch_size',
                        help='batch size', default=None, type=int)
    ### Attack param
    parser.add_argument('--alpha', dest='alpha',
                        help='alpha', default=None, type=float)
    parser.add_argument('--num_iter', dest='num_iter',
                        help='iteration number', default=None, type=int)
    parser.add_argument('--warm_num_iter', default=None, type=int)
    parser.add_argument('--eps_feature', default=None, type=float)
    parser.add_argument('--eps_locality', default=None, type=float)
    parser.add_argument('--momentum_mu', default=0., type=float)

    parser.add_argument('--attack_type', default=None, type=str)
    parser.add_argument('--obj_type', default=None, type=str)
    parser.add_argument('--att_loss_func', default=None, type=str)
    parser.add_argument('--black',action='store_true', default=False)
    parser.add_argument('--inv',  action='store_true', default=False)

    ### Model-concerning params
    parser.add_argument('--pretrained_path', default=None, type=str)

    ### Eval param
    parser.add_argument('--eval_num_iter', 
                        help='eval iteration number', default=None, type=int)
    parser.add_argument('--eval_alpha', 
                        help='eval step size per iteration', default=None, type=float)
    parser.add_argument('--eval_mode', default=None, type=str, choices=['clean', 'all', 'single', 'strong', 'supple', 'supple_s'], help='evaluation mode.')

    ### Train param
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='start epoch number for resume training', default=None, type=int)
    parser.add_argument('--eval_epoch', default=None, type=int)
    parser.add_argument('--eval_per_num_epoch', default=None, type=int)

    parser.add_argument('--attack2_type', default=None, type=str)
    parser.add_argument('--obj2_type', default=None, type=str)
    parser.add_argument('--mode', default=None, type=str, choices=['at', '2step', 'eval'], help='training mode. pixel+loc: first perturb locations of points to generate new graph structure, then perturb features.')
    parser.add_argument('--burn_in_period', default=None, type=int)

    parser.add_argument('--sync_minmax', default=None, type=int, help='whether to use the same loss function or not.')
    parser.add_argument('--loss_func', default=None, type=str)

    parser.add_argument('--reg_level', default=None, type=int, help='regularization level')
    parser.add_argument('--reg_ratio', default=None, type=float)

    parser.add_argument('--affinity_mask', default=None, type=int)

    ### Utils params
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--prefix", default='', type=str, help='prefix to the experiment subdir name')
    parser.add_argument("--datetime", default=None, type=str)
    args = parser.parse_args()

    # load cfg from file
    if args.cfg_file is not None:
        for f in args.cfg_file:
            cfg_from_file(f)

    # load cfg from arguments
    if args.loss_func is not None:
        cfg_from_list(['TRAIN.LOSS_FUNC', args.loss_func])
    if args.sync_minmax is not None:
        cfg_from_list(['TRAIN.SYNC_MINMAX', args.sync_minmax])
    if args.affinity_mask is not None:
        cfg_from_list(['EVAL.AFF_MASK', args.affinity_mask])
    if args.eval_mode is not None:
        cfg_from_list(['EVAL.MODE', args.eval_mode])
    if args.batch_size is not None:
        cfg_from_list(['BATCH_SIZE', args.batch_size])
    if args.start_epoch is not None:
        cfg_from_list(['TRAIN.START_EPOCH', args.start_epoch])
    if args.eval_epoch is not None:
        cfg_from_list(['EVAL.EPOCH', args.eval_epoch])
    if args.eval_per_num_epoch is not None:
        cfg_from_list(['EVAL.NUM_EPOCH', args.eval_per_num_epoch])
    if args.alpha is not None:
        cfg_from_list(['ATTACK.ALPHA', args.alpha])
    if args.eval_alpha is not None:
        cfg_from_list(['ATTACK.EVAL_ALPHA', args.eval_alpha])
    if args.att_loss_func is not None:
        cfg_from_list(['ATTACK.LOSS_FUNC', args.att_loss_func])
    if args.num_iter is not None:
        cfg_from_list(['ATTACK.STEP', args.num_iter])
    if args.eval_num_iter is not None:
        cfg_from_list(['ATTACK.EVAL_STEP', args.eval_num_iter])
    if args.eps_feature is not None:
        cfg_from_list(['ATTACK.EPSILON_FEATURE', args.eps_feature])
    if args.eps_locality is not None:
        cfg_from_list(['ATTACK.EPSILON_LOCALITY', args.eps_locality])
    if args.eps_feature is not None:
        cfg_from_list(['ATTACK.EVAL_EPSILON_FEATURE', args.eps_feature])
    if args.eps_locality is not None:
        cfg_from_list(['ATTACK.EVAL_EPSILON_LOCALITY', args.eps_locality])
    if args.momentum_mu is not None:
        cfg_from_list(['ATTACK.MU', args.momentum_mu])
    if args.attack_type is not None:
        cfg_from_list(['ATTACK.TYPE', args.attack_type])
    if args.obj_type is not None:
        cfg_from_list(['ATTACK.OBJ_TYPE', args.obj_type])
    if args.attack2_type is not None:
        cfg_from_list(['ATTACK2.TYPE', args.attack2_type])
    if args.obj2_type is not None:
        cfg_from_list(['ATTACK2.OBJ_TYPE', args.obj2_type])
    if args.warm_num_iter is not None:
        cfg_from_list(['ATTACK2.STEP', args.warm_num_iter])
    if args.inv is not None:
        cfg_from_list(['ATTACK.INV', args.inv])
    if args.black is not None:
        cfg_from_list(['ATTACK.BLACK', args.black])
    if args.pretrained_path is not None:
        cfg_from_list(['PRETRAINED_PATH', args.pretrained_path])

    if args.mode is not None:
        cfg_from_list(['TRAIN.MODE', args.mode])
    if args.reg_level is not None:
        cfg_from_list(['TRAIN.REG_LEVEL', args.reg_level])
    if args.reg_ratio is not None:
        cfg_from_list(['TRAIN.REG_RATIO', args.reg_ratio])
    if args.burn_in_period is not None:
        cfg_from_list(['TRAIN.BURN_IN_PERIOD', args.burn_in_period])

    if len(cfg.MODEL_NAME) != 0 and len(cfg.DATASET_NAME) != 0:
        # outp_path = get_output_dir(cfg.MODEL_NAME, cfg.DATASET_NAME)
        inv_str = 'Inv-' if args.inv else ''
        black_str = 'BB-' if args.black else ''
        if cfg.TRAIN.MODE == 'at':
            mode_str = cfg.TRAIN.MODE
            attack_str = black_str+inv_str+cfg.ATTACK.LOSS_FUNC+'_'+cfg.ATTACK.OBJ_TYPE+'_'+cfg.ATTACK.TYPE
        elif cfg.TRAIN.MODE == '2step':
            mode_str = cfg.TRAIN.MODE+'_warmup_'+str(cfg.TRAIN.BURN_IN_PERIOD)+'_'+str(cfg.TRAIN.REG_RATIO)
            attack_str = black_str+inv_str+cfg.ATTACK.LOSS_FUNC+'_att1_'+cfg.ATTACK.OBJ_TYPE+'_'+cfg.ATTACK.TYPE+'_s_'+str(cfg.ATTACK.STEP)+'_att2_'+cfg.ATTACK2.OBJ_TYPE+'_'+cfg.ATTACK2.TYPE+'_s_'+str(cfg.ATTACK2.STEP)
        elif cfg.TRAIN.MODE == 'eval':
            mode_str = cfg.TRAIN.MODE+'_'+cfg.EVAL.MODE
            attack_str = black_str+inv_str+cfg.ATTACK.LOSS_FUNC+'_'+cfg.ATTACK.OBJ_TYPE+'_'+cfg.ATTACK.TYPE+'_s_'+str(cfg.ATTACK.EVAL_STEP)

        prefix_head=args.prefix+'_'+mode_str+'_'+attack_str+'_'+cfg.MODEL_NAME+'_'+cfg.DATASET_NAME

        outp_path = get_output_dir_new(prefix_head, now_day=args.datetime, exp_name=args.exp_name)
        cfg_from_list(['OUTPUT_PATH', outp_path])
    #assert len(cfg.OUTPUT_PATH) != 0, 'Invalid OUTPUT_PATH! Make sure model name and dataset name are specified.'
    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    return args
