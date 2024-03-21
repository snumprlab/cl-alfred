import torch
import numpy as np
import torch.nn.functional as F

from methods.er_baseline import ER
from utils.data_loader import DistillationMemory


class CAMA(ER):
    def __init__(self, n_classes, model, **kwargs):
        super().__init__(n_classes, model, **kwargs)
        if kwargs["temp_batchsize"] is None:
            self.temp_batchsize = self.batch_size - 2 * self.batch_size // 3
        self.memory = DistillationMemory(self.exposed_classes, data_dir=self.data_dir)

        self.cls_pred_actions = [[] for _ in range(15)]
        self.cls_pred_actions_length = kwargs['N']
        self.cls_pred_classes = [[] for _ in range(119)]
        self.cls_pred_classes_length = kwargs['N']

    def online_step(self, sample, sample_num):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.temp_batch.append(sample)
        self.num_updates += self.online_iter

        if len(self.temp_batch) == self.temp_batchsize:
            info, logits = self.online_train(
                self.temp_batch,
                self.batch_size,
                iterations=int(self.num_updates),
                stream_batch_size=self.temp_batchsize
            )
            self.report_training(sample_num, info)

            for i, stored_sample in enumerate(self.temp_batch):
                self.update_memory(stored_sample, logits[i])

                p_alow = logits[i][0][logits[i][2].nonzero().view(-1)]
                l_alow = logits[i][5][logits[i][2].nonzero().view(-1)]
                for j in range(p_alow.shape[0]):
                    label_idx = l_alow[j].item()
                    confidence = F.softmax(p_alow[j])[label_idx].item()
                    self.cls_pred_actions[label_idx].append(confidence)
                    if len(self.cls_pred_actions[label_idx]) > self.cls_pred_actions_length:
                        del self.cls_pred_actions[label_idx][0]

                p_alow_mask = logits[i][1][logits[i][3].nonzero().view(-1)]
                l_alow_mask = logits[i][6]
                for j in range(p_alow_mask.shape[0]):
                    label_idx = l_alow_mask[j].item()
                    confidence = F.softmax(p_alow_mask[j])[label_idx].item()
                    self.cls_pred_classes[label_idx].append(confidence)
                    if len(self.cls_pred_classes[label_idx]) > self.cls_pred_classes_length:
                        del self.cls_pred_classes[label_idx][0]

            for i in range(len(self.temp_batch), self.batch_size):
                if i < len(logits):
                    p_alow = logits[i][0][logits[i][2].nonzero().view(-1)]
                    l_alow = logits[i][5][logits[i][2].nonzero().view(-1)]
                    for j in range(p_alow.shape[0]):
                        label_idx = l_alow[j].item()
                        confidence = F.softmax(p_alow[j])[label_idx].item()
                        self.cls_pred_actions[label_idx].append(confidence)
                        if len(self.cls_pred_actions[label_idx]) > self.cls_pred_actions_length:
                            del self.cls_pred_actions[label_idx][0]

                    # pred. vs. GT classes
                    p_alow_mask = logits[i][1][logits[i][3].nonzero().view(-1)]
                    l_alow_mask = logits[i][6]
                    for j in range(p_alow_mask.shape[0]):
                        label_idx = l_alow_mask[j].item()
                        confidence = F.softmax(p_alow_mask[j])[label_idx].item()
                        self.cls_pred_classes[label_idx].append(confidence)
                        if len(self.cls_pred_classes[label_idx]) > self.cls_pred_classes_length:
                            del self.cls_pred_classes[label_idx][0]

            for i in range(len(self.temp_batch), len(logits)):
                task_id = logits[i][4]

                task_ids_in_memory = [l[4] for l in self.memory.logits]
                if task_id in task_ids_in_memory:
                    idx = task_ids_in_memory.index(task_id)
                    min_len = min(len(self.memory.logits[idx][0]), len(logits[i][0]))
                    
                    updated_logits_actions = self.memory.logits[idx][0][:min_len]
                    for j in range(min_len):
                        if logits[i][2][j] == 1:
                            c = np.mean(self.cls_pred_actions[logits[i][5][j].item()]) if len(self.cls_pred_actions[logits[i][5][j].item()]) > 0 else 0
                            c = np.clip(c - 1/13, 0, 1)
                            updated_logits_actions[j] = (1 - c)*updated_logits_actions[j] + c*logits[i][0][j]
                    
                    updated_logits_classes = self.memory.logits[idx][1][:min_len]
                    cc = 0
                    for j in range(min_len):
                        if logits[i][2][j] == 1 and logits[i][3][j] == 1:
                            c = np.mean(self.cls_pred_classes[logits[i][6][cc].item()]) if len(self.cls_pred_classes[logits[i][6][cc].item()]) > 0 else 0
                            c = np.clip(c - 1/119, 0, 1)
                            updated_logits_classes[j] = (1 - c)*updated_logits_classes[j] + c*logits[i][1][j]
                            cc += 1

                    self.memory.logits[idx] = (
                        updated_logits_actions,
                        updated_logits_classes,
                        logits[i][2],
                        logits[i][3],
                        logits[i][4],
                    )
            self.temp_batch = []
            self.num_updates -= int(self.num_updates)

    def online_train(self, sample, batch_size, iterations=1, stream_batch_size=1, alpha=0.5, beta=0.5):
        if len(sample) > 0:
            self.memory.register_stream(sample)
        memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

        info = {}
        for i in range(iterations):
            self.model.train()

            data = self.memory.get_batch(batch_size, stream_batch_size)
            batch = [(self.model.load_task_json(task), swapColor) for task, swapColor in data['batch']]
            task_ids = [f'{task["task"]}/{task["repeat_idx"]}' for task, swapColor in data['batch']]
            feat = self.model.featurize(batch)
            out = self.model.forward(feat)

            # logit
            p_alow = out['out_action_low']
            l_alow = feat['action_low']
            p_alow_mask = out['out_action_low_mask']
            l_alow_mask = feat['action_low_mask_label_unflattened']
            valid = feat['action_low_valid_interact']
            pad_valid = (l_alow != self.model.pad)

            logit = [(p_alow[j], p_alow_mask[j], pad_valid[j], valid[j], task_ids[j], l_alow[j], l_alow_mask[j]) for j
                     in range(len(p_alow))]

            distill_size = memory_batch_size // 2

            self.optimizer.zero_grad()

            if distill_size > 0:
                cls_loss = self.model.compute_loss_unsummed(out, batch, feat)[:-distill_size]
                loss = cls_loss[:self.temp_batchsize].mean() + alpha * cls_loss[self.temp_batchsize:].mean()
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
                    distilled_loss_actions = beta * sum(distilled_losses_actions) / len(distilled_losses_actions)
                    distilled_loss_classes = beta * sum(distilled_losses_classes) / len(distilled_losses_classes)
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
            else:
                loss = sum(self.model.compute_loss(out, batch, feat).values())
                if 'cls_loss' not in info:
                    info['cls_loss'] = 0
                info['cls_loss'] += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            self.update_schedule()

        logits = [(l[0].detach(), l[1].detach(), l[2].detach(), l[3].detach(), l[4], l[5], l[6]) for l in logit]

        info = {k: v / iterations for k, v in info.items()}
        return info, logits

    def update_memory(self, sample, logit=None):
        self.reservoir_memory(sample, logit)

    def reservoir_memory(self, sample, logit=None):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)
                self.memory.save_logit(logit, j)
        else:
            self.memory.replace_sample(sample)
            self.memory.save_logit(logit)
