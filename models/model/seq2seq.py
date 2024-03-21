import os
import sys

import random
import json
import torch
import collections
import numpy as np
from torch import nn
from tqdm import trange, tqdm

from utils.method_manager import select_method
from collections import defaultdict



class Module(nn.Module):

    def __init__(self, args, vocab):
        '''
        Base Seq2Seq agent with common train and val loops
        '''
        super().__init__()

        # sentinel tokens
        self.pad = 0
        self.seg = 1

        # args and vocab
        self.args = args
        self.vocab = vocab

        # emb modules
        self.emb_word = nn.Embedding(len(vocab['word']), args.demb)
        self.emb_action_low = nn.Embedding(len(vocab['action_low']), args.demb)

        # end tokens
        self.stop_token = self.vocab['action_low'].word2index("<<stop>>", train=False)
        self.seg_token = self.vocab['action_low'].word2index("<<seg>>", train=False)

        # set random seed (Note: this is not the seed used to initialize THOR object locations)
        random.seed(a=args.seed)
        np.random.seed(args.seed)


    def run_train(self, args=None):
        '''
        training loop
        '''

        # args
        args = args or self.args

        # dump config
        fconfig = os.path.join(args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(args), f, indent=4)

        # display dout
        print("Saving to: %s" % self.args.dout)

        cl_method = select_method(args=args, n_classes=args.n_tasks, model=self)

        test_datalist_seen = json.load(open(f'embodied_split/{args.incremental_setup}/valid_seen.json', 'r'))
        test_datalist_seen = [(s, False) for s in test_datalist_seen]
        test_datalist_unseen = json.load(open(f'embodied_split/{args.incremental_setup}/valid_unseen.json', 'r'))
        test_datalist_unseen = [(s, False) for s in test_datalist_unseen]

        samples_cnt = 0
        eval_results_seen = defaultdict(list)
        eval_results_unseen = defaultdict(list)
        for cur_iter in range(args.n_tasks):
            cur_train_datalist = json.load(open(f'embodied_split/{args.incremental_setup}/embodied_data_disjoint_rand{args.stream_seed}_cls1_task{cur_iter}.json', 'r'))

            cl_method.online_before_task(cur_iter)
            for i, data in enumerate(tqdm(cur_train_datalist)):
                samples_cnt += 1
                traj_data = self.load_task_json(data['task'])
                data['num_frames'] = len([aa for a in traj_data['num']['action_low'] for aa in a])

                cl_method.online_step(data, samples_cnt)

                if samples_cnt % args.eval_period == 0:
                    # valid_seen
                    eval_dict_seen = cl_method.online_evaluate(test_datalist_seen, samples_cnt, args.batchsize, tag='valid_seen')
                    eval_results_seen["data_cnt"].append(samples_cnt)
                    for k in eval_results_seen:
                        if k not in ['data_cnt']:
                            eval_results_seen[k].append(eval_dict_seen[k])

                    # valid_unseen
                    eval_dict_unseen = cl_method.online_evaluate(test_datalist_unseen, samples_cnt, args.batchsize, tag='valid_unseen')
                    eval_results_unseen["data_cnt"].append(samples_cnt)
                    for k in eval_results_unseen:
                        if k not in ['data_cnt']:
                            eval_results_unseen[k].append(eval_dict_unseen[k])
            cl_method.online_after_task(cur_iter)

            torch.save({
                'metric': {'samples_cnt': samples_cnt},
                'model': self.state_dict(),
                'optim': cl_method.optimizer.state_dict(),
                'args': self.args,
                'vocab': self.vocab,
            }, os.path.join(args.dout, 'net_epoch_%09d_%s.pth' % (samples_cnt, data['klass'])))


    def run_pred(self, dev, batch_size=32, name='dev', iter=0):
        '''
        validation loop
        '''
        m_dev = collections.defaultdict(list)
        p_dev = {}
        self.eval()
        total_loss = list()
        dev_iter = iter
        for batch, feat in self.iterate(dev, batch_size):
            out = self.forward(feat)
            preds = self.extract_preds(out, batch, feat)
            p_dev.update(preds)
            loss = self.compute_loss(out, batch, feat)
            for k, v in loss.items():
                ln = 'loss_' + k
                m_dev[ln].append(v.item())
            sum_loss = sum(loss.values())
            total_loss.append(float(sum_loss.detach().cpu()))
            dev_iter += len(batch)

        m_dev = {k: sum(v) / len(v) for k, v in m_dev.items()}
        total_loss = sum(total_loss) / len(total_loss)
        return p_dev, dev_iter, total_loss, m_dev

    def featurize(self, batch):
        raise NotImplementedError()

    def forward(self, feat, max_decode=100):
        raise NotImplementedError()

    def extract_preds(self, out, batch, feat):
        raise NotImplementedError()

    def compute_loss(self, out, batch, feat):
        raise NotImplementedError()

    def compute_metric(self, preds, data):
        raise NotImplementedError()

    def get_task_and_ann_id(self, ex):
        '''
        single string for task_id and annotation repeat idx
        '''
        return "%s_%s" % (ex['task_id'], str(ex['ann']['repeat_idx']))

    def make_debug(self, preds, data):
        '''
        readable output generator for debugging
        '''
        debug = {}
        for task in data:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            debug[i] = {
                'lang_goal': ex['turk_annotations']['anns'][ex['ann']['repeat_idx']]['task_desc'],
                'action_low': [a['discrete_action']['action'] for a in ex['plan']['low_actions']],
                'p_action_low': preds[i]['action_low'].split(),
            }
        return debug

    def load_task_json(self, task):
        '''
        load preprocessed json from disk
        '''
        json_path = os.path.join(self.args.data, task['task'], '%s' % self.args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)
        return data

    def get_task_root(self, ex):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(self.args.data, ex['split'], *(ex['root'].split('/')[-2:]))

    def iterate(self, data, batch_size):
        '''
        breaks dataset into batch_size chunks for training
        '''
        for i in range(0, len(data), batch_size):
            tasks = data[i:i+batch_size]
            batch = [(self.load_task_json(task), swapColor) for task, swapColor in tasks]
            feat = self.featurize(batch)
            yield batch, feat

    def zero_input(self, x, keep_end_token=True):
        '''
        pad input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        return list(np.full_like(x[:-1], self.pad)) + end_token

    def zero_input_list(self, x, keep_end_token=True):
        '''
        pad a list of input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        lz = [list(np.full_like(i, self.pad)) for i in x[:-1]] + end_token
        return lz

    @staticmethod
    def adjust_lr(optimizer, init_lr, epoch, decay_epoch=5):
        '''
        decay learning rate every decay_epoch
        '''
        lr = init_lr * (0.1 ** (epoch // decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @classmethod
    def load(cls, fsave):
        '''
        load pth model from disk
        '''
        save = torch.load(fsave)
        model = cls(save['args'], save['vocab'])
        model.load_state_dict(save['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(save['optim'])
        return model, optimizer

    @classmethod
    def has_interaction(cls, action):
        '''
        check if low-level action is interactive
        '''
        non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>', '<<pad>>', '<<seg>>']
        if any(a in action for a in non_interact_actions):
            return False
        else:
            return True
