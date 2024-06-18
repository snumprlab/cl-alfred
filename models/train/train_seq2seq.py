import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))

import os
import torch
import pprint
import json
from data.preprocess import Dataset
from importlib import import_module
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from models.utils.helper_utils import optimizer_to

torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--data', help='dataset folder', default='data/json_feat_2.1.0')
    parser.add_argument('--splits', help='json file containing train/dev/test splits', default='data/splits/oct21.json')
    parser.add_argument('--preprocess', help='store preprocessed data to json files', action='store_true')
    parser.add_argument('--pp_folder', help='folder name for preprocessed data', default='pp')
    parser.add_argument('--save_every_epoch', help='save model after every epoch (warning: consumes a lot of space)', action='store_true', default=True)
    parser.add_argument('--model_arch', help='model to use', default='seq2seq_im_mask')
    parser.add_argument('--gpu', help='use gpu', action='store_true', default=True)
    parser.add_argument('--dout', help='where to save model')
    parser.add_argument('--resume', help='load a checkpoint')

    # hyper parameters
    parser.add_argument('--dhid', help='hidden layer size', default=512, type=int)
    parser.add_argument('--dframe', help='image feature vec size', default=5*7*7, type=int)
    parser.add_argument('--demb', help='language embedding size', default=100, type=int)
    parser.add_argument('--pframe', help='image pixel size (assuming square shape eg: 300x300)', default=300, type=int)
    parser.add_argument('--mask_loss_wt', help='weight of mask loss', default=1., type=float)
    parser.add_argument('--action_loss_wt', help='weight of action loss', default=1., type=float)
    parser.add_argument('--subgoal_aux_loss_wt', help='weight of subgoal completion predictor', default=0, type=float)
    parser.add_argument('--pm_aux_loss_wt', help='weight of progress monitor', default=0.2, type=float)

    # dropouts
    parser.add_argument('--zero_goal', help='zero out goal language', action='store_true')
    parser.add_argument('--zero_instr', help='zero out step-by-step instr language', action='store_true')
    parser.add_argument('--lang_dropout', help='dropout rate for language (goal + instr)', default=0., type=float)
    parser.add_argument('--input_dropout', help='dropout rate for concatted input feats', default=0., type=float)
    parser.add_argument('--vis_dropout', help='dropout rate for Resnet feats', default=0.3, type=float)
    parser.add_argument('--hstate_dropout', help='dropout rate for LSTM hidden states during unrolling', default=0.3, type=float)
    parser.add_argument('--attn_dropout', help='dropout rate for attention', default=0., type=float)
    parser.add_argument('--actor_dropout', help='dropout rate for actor fc', default=0., type=float)

    # other settings
    parser.add_argument('--dec_teacher_forcing', help='use gpu', action='store_true')
    parser.add_argument('--temp_no_history', help='use gpu', action='store_true')
    parser.add_argument('--panoramic', help='use panoramic', action='store_true', default=True)
    parser.add_argument('--orientation', help='use orientation features', action='store_true')
    parser.add_argument('--panoramic_concat', help='use panoramic', action='store_true', default=True)

    # CL args
    parser.add_argument("--mode", type=str, help="Select CIL method")
    parser.add_argument("--n_tasks", type=int, help="The number of tasks")
    parser.add_argument("--rnd_seed", type=int, default=1, help="Random seed number.")
    parser.add_argument("--stream_seed", type=int, help="Random seed number for stream.")
    parser.add_argument("--memory_size", type=int, default=500, help="Episodic memory size (# videos)")
    parser.add_argument("--incremental_setup", type=str, choices=['behavior_il', 'environment_il', 'environment_il_nosampling'])

    # Dataset
    parser.add_argument("--log_path", type=str, default="results", help="The path logs are saved.")

    # Model
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name")

    # Train
    parser.add_argument("--opt_name", type=str, default="adam", help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument("--batchsize", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

    # Regularization
    parser.add_argument("--reg_coef", type=int, default=100, help="weighting for the regularization loss term")
    parser.add_argument("--data_dir", type=str, help="location of the dataset")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Turn on Debug mode")

    # Eval period
    parser.add_argument("--eval_period", type=int, default=500000, help="evaluation period for true online setup")

    parser.add_argument("--temp_batchsize", type=int, help="temporary batch size, for true online")
    parser.add_argument("--online_iter", type=float, default=1, help="number of model updates per samples seen.") # 1.182575531717415 0.8456119488179613

    # CLIB
    parser.add_argument("--imp_update_period", type=int, default=1, help="period between importance update, in units of model updates (increase for heavy datasets like ImageNet)")
    parser.add_argument('--lr_step', type=float, default=0.95, help='step of iterating lr for adaptive LR')
    parser.add_argument('--lr_length', type=int, default=10, help='period of iterating lr for adaptive LR')
    parser.add_argument('--lr_period', type=int, default=10, help='period of iterating lr for adaptive LR')

    # MIR
    parser.add_argument('--mir_cands', type=int, default=50, help='# candidates to use for MIR')

    # CAMA
    parser.add_argument('--N', type=int, default=5, help='# the recent confidence scores for CAMA')

    # XDER
    parser.add_argument("--simclr_batch_size", type=int, default=64)
    parser.add_argument("--simclr_num_aug", type=int, default=2)
    parser.add_argument("--simclr_temp", type=int, default=5)
    parser.add_argument("--total_classes", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument("--param_m", type=float, default=0.7)
    parser.add_argument("--lambd", type=float, default=0.05)

    # args and init
    args = parser.parse_args()
    args.dout = args.dout.format(**vars(args))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CL-ALFRED n_task
    if args.incremental_setup in ['behavior_il']:
        args.n_tasks = 7
    elif args.incremental_setup in ['environment_il']:
        args.n_tasks = 4
    else:
        raise Exception("args.n_task should be either 7 (Behavior-IL) or 4 (Environment-IL).")

    # check if dataset has been preprocessed
    if not os.path.exists(os.path.join(args.data, "%s.vocab" % args.pp_folder)) and not args.preprocess:
        raise Exception("Dataset not processed; run with --preprocess")

    # make output dir
    pprint.pprint(args)
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    # load train/valid/tests splits
    with open(args.splits) as f:
        splits = json.load(f)
        pprint.pprint({k: len(v) for k, v in splits.items()})

    # preprocess and save
    if args.preprocess:
        print("\nPreprocessing dataset and saving to %s folders ... This will take a while. Do this once as required." % args.pp_folder)
        dataset = Dataset(args, None)
        dataset.preprocess_splits(splits)
        vocab = torch.load(os.path.join(args.dout, "%s.vocab" % args.pp_folder))
    else:
        vocab = torch.load(os.path.join(args.data, "%s.vocab" % args.pp_folder))

    # load model
    M = import_module('model.{}'.format(args.model_arch))
    if args.resume:
        print("Loading: " + args.resume)
        model, optimizer = M.Module.load(args.resume)
    else:
        model = M.Module(args, vocab)
        optimizer = None

    # to gpu
    if args.gpu:
        model = model.to(torch.device('cuda'))
        if not optimizer is None:
            optimizer_to(optimizer, torch.device('cuda'))

    # start train loop
    model.run_train(args=args)
