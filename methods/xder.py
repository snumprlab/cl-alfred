import torch
import numpy as np
from torch.nn import functional as F

import random
random.seed(100)

from methods.der import DER
from utils.data_loader import DistillationMemory_XDER



class XDER(DER):
    def __init__(self, n_classes, model, **kwargs):
        super().__init__(n_classes, model, **kwargs)
        if kwargs["temp_batchsize"] is None:
            self.temp_batchsize = self.batch_size - 2 * self.batch_size // 3

        self.simclr_temp = kwargs["simclr_temp"]
        self.simclr_batch_size = kwargs["simclr_batch_size"]
        self.gamma = kwargs["gamma"]
        self.tasks = kwargs["n_tasks"]
        self.m = kwargs["param_m"]
        self.eta = kwargs["eta"]
        self.simclr_num_aug = kwargs["simclr_num_aug"]
        self.lambd = kwargs["lambd"]

        self.update_counter = torch.zeros(self.memory_size).to(self.device)
        self.memory = DistillationMemory_XDER(cls_list=self.exposed_classes, data_dir=self.data_dir)

        self.mask_actions = [[] for _ in range(self.tasks)]
        self.mask_classes = [[] for _ in range(self.tasks)]
        self.mask_actions[0].extend([0, 1])
        self.mask_classes[0].extend([0])


    def online_before_task(self, cur_iter):
        super().online_before_task(cur_iter)
        self.cur_task = cur_iter

    def online_after_task(self, cur_iter, batchsize=512):
        super().online_after_task(cur_iter)

        # Update future past logits
        if cur_iter > 0:
            if len(self.memory) > 0:
                self.model.eval()
                with torch.no_grad():
                    for i in range(len(self.memory.images)):
                        if self.memory.labels[i] < self.cur_task:
                            logit = self.memory.logits[i]

                            feat = self.model.featurize(
                                [(self.model.load_task_json(self.memory.images[i]['task']), False)]
                            )
                            out = self.model.forward(feat)

                            # actions
                            to_transplant = self.update_memory_logits(
                                logit[4][logit[2].nonzero().view(-1)],
                                logit[0][logit[2].nonzero().view(-1)],
                                out['out_action_low'][0],
                                self.cur_task, n_tasks=self.tasks - self.cur_task, masks=self.mask_actions
                            )
                            logit[0][logit[2].nonzero().view(-1)] = to_transplant

                            # classes
                            to_transplant = self.update_memory_logits(
                                logit[5],
                                logit[1][logit[3].nonzero().view(-1)],
                                out['out_action_low_mask'][0][feat['action_low_valid_interact'][0].nonzero().view(-1)],
                                self.cur_task, n_tasks=self.tasks - self.cur_task, masks=self.mask_classes
                            )
                            logit[1][logit[3].nonzero().view(-1)] = to_transplant

                            self.memory.update_logits([i], [logit])
                            self.memory.update_task_ids([i], self.cur_task)

        self.update_counter = torch.zeros(self.memory_size).to(self.device)

    def online_step(self, sample, sample_num):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.temp_batch.append(sample)
        self.num_updates += self.online_iter

        if len(self.temp_batch) == self.temp_batchsize:
            info, logits, task_ids = self.online_train(
                self.temp_batch, self.batch_size,
                iterations=int(self.num_updates),
                stream_batch_size=self.temp_batchsize
            )
            self.report_training(sample_num, info)

            for i, stored_sample in enumerate(self.temp_batch):
                self.update_memory(stored_sample, logit=logits[i], task_ids=task_ids)

            self.temp_batch = []
            self.num_updates -= int(self.num_updates)

    def get_logit_constraint_loss(self, loss_stream, logit, stream_batch_size):
        outputs_actions = [l[0][l[2].nonzero().view(-1)] for l in logit[:stream_batch_size]]
        outputs_classes = [l[1][l[3].nonzero().view(-1)] for l in logit[:stream_batch_size]]
        buf_outputs_actions = [l[0][l[2].nonzero().view(-1)] for l in logit[stream_batch_size:]]
        buf_outputs_classes = [l[1][l[3].nonzero().view(-1)] for l in logit[stream_batch_size:]]
        buf_labels_actions = [l[4][l[2].nonzero().view(-1)] for l in logit[stream_batch_size:]]
        buf_labels_classes = [l[5] for l in logit[stream_batch_size:]]

        loss_constr_past = torch.tensor(0.).type(loss_stream.dtype).to(self.device)
        if self.cur_task > 0:
            loss_constr_past_actions = []
            loss_constr_past_classes = []
            for i in range(stream_batch_size):
                # actions
                outputs = outputs_actions[i]
                mask_total = [j for j in sorted(list(set([j for m in self.mask_actions[:self.cur_task + 1] for j in m])))]
                chead = F.softmax(outputs[:, mask_total], 1)

                mask_curr = [mask_total.index(j) for j in self.mask_actions[self.cur_task]]
                mask_prev = [mask_total.index(j) for j in mask_total if j not in self.mask_actions[self.cur_task]]
                good_head = chead[:, sorted(mask_curr)]
                bad_head = chead[:, sorted(mask_prev)]

                loss_constr = bad_head.max(1)[0].detach() + self.m
                if good_head.size(1) > 0:
                    loss_constr -= good_head.max(1)[0]

                mask = loss_constr > 0

                if mask.any():
                    _loss_constr_past_actions = self.eta * loss_constr[mask].mean()
                    loss_constr_past_actions.append(_loss_constr_past_actions)

                # classes
                outputs = outputs_classes[i]
                mask_total = [j for j in sorted(list(set([j for m in self.mask_classes[:self.cur_task + 1] for j in m])))]
                chead = F.softmax(outputs[:, mask_total], 1)

                mask_curr = [mask_total.index(j) for j in self.mask_classes[self.cur_task]]
                mask_prev = [mask_total.index(j) for j in mask_total if j not in self.mask_classes[self.cur_task]]
                good_head = chead[:, sorted(mask_curr)]
                bad_head = chead[:, sorted(mask_prev)]

                loss_constr = bad_head.max(1)[0].detach() + self.m
                if good_head.size(1) > 0:
                    loss_constr -= good_head.max(1)[0]

                mask = loss_constr > 0

                if mask.any():
                    _loss_constr_past_classes = self.eta * loss_constr[mask].mean()
                    loss_constr_past_classes.append(_loss_constr_past_classes)

            if len(loss_constr_past_actions) > 0:
                loss_constr_past += torch.stack(loss_constr_past_actions).mean()
            if len(loss_constr_past_classes) > 0:
                loss_constr_past += torch.stack(loss_constr_past_classes).mean()

        loss_constr_futu = torch.tensor(0.).to(self.device)
        if self.cur_task < self.tasks - 1:
            loss_constr_futu_actions = []
            loss_constr_futu_classes = []
            for i in range(stream_batch_size):
                outputs = outputs_actions[i]

                mask_bad = [j for j in sorted(list(
                    set(range(outputs.size(1))) - set([j for m in self.mask_actions[:self.cur_task+1] for j in m])
                ))]
                mask_good = self.mask_actions[self.cur_task]
                bad_head = outputs[:, mask_bad]
                good_head = outputs[:, mask_good]
                if len(self.memory.images) > 0:
                    pass

                loss_constr = 0
                if bad_head.size(1) > 0:
                    loss_constr += bad_head.max(1)[0].detach()
                if good_head.size(1) > 0:
                    loss_constr -= good_head.max(1)[0]

                if type(loss_constr) != int:
                    loss_constr += self.m
                    mask = loss_constr > 0
                    if mask.any():
                        _loss_constr_futu_actions = self.eta * loss_constr[mask].mean()
                        loss_constr_futu_actions.append(_loss_constr_futu_actions)

            if len(self.memory.images):
                for i in range(len(buf_outputs_actions)):
                    buf_outputs = buf_outputs_actions[i]
                    buf_labels = buf_labels_actions[i]

                    mask_bad = [j for j in sorted(list(
                        set(range(outputs.size(1))) - set([j for m in self.mask_actions[:self.cur_task + 1] for j in m])
                    ))]
                    mask_good = []
                    for l in buf_labels:
                        for j in range(len(self.mask_actions)):
                            if l in self.mask_actions[j]:
                                mask_good.append(j)
                                break

                    bad_head = buf_outputs[:, mask_bad]
                    good_head = [buf_outputs[j][mask_good[j]] for j in range(len(buf_outputs))]
                    loss_constr = 0
                    if bad_head.size(1) > 0:
                        loss_constr += bad_head.max(1)[0]
                    if len(good_head) > 0:
                        loss_constr -= torch.stack([good_head[j].max() for j in range(len(good_head))])

                    if type(loss_constr) != int:
                        loss_constr += self.m
                        mask = loss_constr > 0
                        if mask.any():
                            _loss_constr_futu_actions = self.eta * loss_constr[mask].mean()
                            loss_constr_futu_actions.append(_loss_constr_futu_actions)

                for i in range(len(buf_outputs_classes)):
                    buf_outputs = buf_outputs_classes[i]
                    buf_labels = buf_labels_classes[i]

                    mask_bad = [j for j in sorted(list(
                        set(range(outputs.size(1))) - set([j for m in self.mask_classes[:self.cur_task + 1] for j in m])
                    ))]
                    mask_good = []
                    for l in buf_labels:
                        for j in range(len(self.mask_classes)):
                            if l in self.mask_classes[j]:
                                mask_good.append(j)
                                break

                    bad_head = buf_outputs[:, mask_bad]
                    good_head = [buf_outputs[j][mask_good[j]] for j in range(len(buf_outputs))]
                    loss_constr = 0
                    if bad_head.size(1) > 0:
                        loss_constr += bad_head.max(1)[0]
                    if len(good_head) > 0:
                        loss_constr -= torch.stack([good_head[j].max() for j in range(len(good_head))])

                    if type(loss_constr) != int:
                        loss_constr += self.m
                        mask = loss_constr > 0
                        if mask.any():
                            _loss_constr_futu_classes = self.eta * loss_constr[mask].mean()
                            loss_constr_futu_classes.append(_loss_constr_futu_classes)

            if len(loss_constr_futu_actions) > 0:
                loss_constr_futu += torch.stack(loss_constr_futu_actions).mean()
            if len(loss_constr_futu_classes) > 0:
                loss_constr_futu += torch.stack(loss_constr_futu_classes).mean()

        return loss_constr_past, loss_constr_futu


    def online_train(self, sample, batch_size, iterations=1, stream_batch_size=1, alpha=0.5, beta=0.5):
        if len(sample) > 0:
            self.memory.register_stream(sample)
        memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

        info = {}
        for i in range(iterations):
            self.model.train()
            data = self.memory.get_batch(batch_size, stream_batch_size)
            batch = [(self.model.load_task_json(task), swapColor) for task, swapColor in data['batch']]
            feat = self.model.featurize(batch)
            out = self.model.forward(feat)

            p_alow = out['out_action_low']
            l_alow = feat['action_low']
            p_alow_mask = out['out_action_low_mask']
            l_alow_mask = feat['action_low_mask_label_unflattened']
            valid = feat['action_low_valid_interact']
            pad_valid = (l_alow != self.model.pad)
            buf_idx = data['indices'].to(self.device)
            task_ids = data['task_id'].to(self.device)

            self.mask_actions[self.cur_task].extend(l_alow[:stream_batch_size].unique().tolist())
            self.mask_actions[self.cur_task] = [j for j in sorted(list(
                set(self.mask_actions[self.cur_task]) - set([l for m in self.mask_actions[:self.cur_task] for l in m])
            ))]
            self.mask_classes[self.cur_task].extend(torch.cat(l_alow_mask[:stream_batch_size], dim=0).unique().tolist())
            self.mask_classes[self.cur_task] = [j for j in sorted(list(
                set(self.mask_classes[self.cur_task]) - set([l for m in self.mask_classes[:self.cur_task] for l in m])
            ))]

            logit = [(p_alow[j], p_alow_mask[j], pad_valid[j], valid[j], l_alow[j], l_alow_mask[j]) for j in range(len(p_alow))]

            distill_size = memory_batch_size // 2

            self.optimizer.zero_grad()

            if distill_size > 0:
                cls_loss = self.model.compute_loss_unsummed(out, batch, feat)[:-distill_size]
                loss = cls_loss[:self.temp_batchsize].mean() + 0.5 * cls_loss[self.temp_batchsize:].mean()
                if 'cls_loss' not in info:
                    info['cls_loss'] = 0
                info['cls_loss'] += loss.item()

                if 'logit' in data:
                    distilled_losses_actions = []
                    distilled_losses_classes = []
                    for pred, gt in zip(logit[-distill_size:], data['logit'][-distill_size:]):
                        distilled_loss_action = (pred[0][pred[2].nonzero().view(-1)] - gt[0][gt[2].nonzero().view(-1)]) ** 2
                        distilled_loss_class = (pred[1][pred[3].nonzero().view(-1)] - gt[1][gt[3].nonzero().view(-1)]) ** 2
                        distilled_losses_actions.append(distilled_loss_action.mean())
                        distilled_losses_classes.append(distilled_loss_class.mean())
                    distilled_loss_actions = beta * sum(distilled_losses_actions) / (len(distilled_losses_actions) + 1e-8)
                    distilled_loss_classes = beta * sum(distilled_losses_classes) / (len(distilled_losses_classes) + 1e-8)
                    distilled_loss = distilled_loss_actions + distilled_loss_classes
                    loss += distilled_loss

                    if 'distilled_loss_actions' not in info:
                        info['distilled_loss_actions'] = 0
                    info['distilled_loss_actions'] += distilled_loss_actions.item()
                    if 'distilled_loss_classes' not in info:
                        info['distilled_loss_classes'] = 0
                    info['distilled_loss_classes'] += distilled_loss_classes.item()
                    if 'distilled_loss' not in info:
                        info['distilled_loss'] = 0
                    info['distilled_loss'] += distilled_loss.item()

                    loss_constr_past, loss_constr_futu = self.get_logit_constraint_loss(loss, logit, stream_batch_size)

                    if 'loss_constr_past' not in info:
                        info['loss_constr_past'] = 0
                    info['loss_constr_past'] += loss_constr_past.item()
                    if 'loss_constr_futu' not in info:
                        info['loss_constr_futu'] = 0
                    info['loss_constr_futu'] += loss_constr_futu.item()

                    loss += loss_constr_futu + loss_constr_past

            else:
                loss = sum(self.model.compute_loss(out, batch, feat).values())
                if 'cls_loss' not in info:
                    info['cls_loss'] = 0
                info['cls_loss'] += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            self.update_schedule()

            if self.cur_task > 0:

                logit = [(l[0].detach(), l[1].detach(), l[2].detach(), l[3].detach(), l[4].detach(), l[5].detach())
                          for l in logit]

                with torch.no_grad():
                    for j in range(stream_batch_size):
                        # actions
                        logit[j][0][logit[j][2].nonzero().view(-1)] = self.update_memory_logits(
                            logit[j][4][logit[j][2].nonzero().view(-1)],
                            logit[j][0][logit[j][2].nonzero().view(-1)],
                            logit[j][0][logit[j][2].nonzero().view(-1)],
                            0, self.cur_task, self.mask_actions
                        )
                        # classes
                        logit[j][1][logit[j][3].nonzero().view(-1)] = self.update_memory_logits(
                            logit[j][5],
                            logit[j][1][logit[j][3].nonzero().view(-1)],
                            logit[j][1][logit[j][3].nonzero().view(-1)],
                            0, self.cur_task, self.mask_classes
                        )

                    chosen = torch.tensor([self.memory.labels[j] < self.cur_task for j in buf_idx]).to(self.device)
                    self.update_counter[buf_idx[chosen]] += 1
                    c = chosen.clone()
                    chosen[c] = torch.rand_like(chosen[c].float()) * self.update_counter[buf_idx[c]] < 1

                    if chosen.any():
                        for j in range(len(data['logit'])):
                            if chosen[j]:
                                # actions
                                to_transplant = self.update_memory_logits(
                                    logit[stream_batch_size:][j][4][logit[stream_batch_size:][j][2].nonzero().view(-1)],
                                    data['logit'][j][0][logit[stream_batch_size:][j][2].nonzero().view(-1)],
                                    logit[stream_batch_size:][j][0][logit[stream_batch_size:][j][2].nonzero().view(-1)].detach(),
                                    self.cur_task, n_tasks=self.tasks-self.cur_task, masks=self.mask_actions
                                )
                                logit[stream_batch_size:][j][0][logit[stream_batch_size:][j][2].nonzero().view(-1)] = to_transplant

                                # classes
                                to_transplant = self.update_memory_logits(
                                    logit[stream_batch_size:][j][5],
                                    data['logit'][j][1][logit[stream_batch_size:][j][3].nonzero().view(-1)],
                                    logit[stream_batch_size:][j][1][logit[stream_batch_size:][j][3].nonzero().view(-1)].detach(),
                                    self.cur_task, n_tasks=self.tasks-self.cur_task, masks=self.mask_classes
                                )
                                logit[stream_batch_size:][j][1][logit[stream_batch_size:][j][3].nonzero().view(-1)] = to_transplant

                                self.memory.update_logits([buf_idx[j]], [logit[stream_batch_size:][j]])
                                self.memory.update_task_ids([buf_idx[j]], self.cur_task)

        logits = [(l[0].detach(), l[1].detach(), l[2].detach(), l[3].detach(), l[4].detach(), l[5].detach()) for l in logit[:stream_batch_size]]

        info = {k: v / iterations for k, v in info.items()}
        return info, logits[:stream_batch_size], task_ids[stream_batch_size:]

    def update_memory_logits(self, gt, old, new, cur_task, n_tasks=1, masks=None):
        mask_transplant = [i for i in sorted(list(set([j for m in masks[cur_task:cur_task+n_tasks] for j in m])))]
        if len(mask_transplant) > 0:
            transplant = new[:, mask_transplant]
            gt_values = old[torch.arange(len(gt)), gt]
            max_values = transplant.max(1).values
            coeff = self.gamma * gt_values / max_values
            coeff = coeff.unsqueeze(1).repeat(1, len(mask_transplant))
            mask = (max_values > gt_values).unsqueeze(1).repeat(1, len(mask_transplant))
            transplant[mask] *= coeff[mask]
            old[:, mask_transplant] = transplant
        return old

    def update_memory(self, sample, logit=None, task_ids=None):
        self.reservoir_memory(sample, logit)

    def reservoir_memory(self, sample, logit=None):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)
                self.memory.save_logit(logit, j)
                self.memory.save_task_id(self.cur_task, j)
        else:
            self.memory.replace_sample(sample)
            self.memory.save_logit(logit)
            self.memory.save_task_id(self.cur_task)