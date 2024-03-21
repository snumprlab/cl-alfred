################################
# This code is referred by
# https://github.com/GT-RIPL/Continual-Learning-Benchmark
################################

import torch

from methods.er_baseline import ER
from utils.data_loader import StreamDataset

class EWCpp(ER):

    def __init__(self, n_classes, model, **kwargs):
        super().__init__(n_classes, model, **kwargs)

        # except for last layers.
        self.params = {n: p for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad}
        self.regularization_terms = {}
        self.task_count = 0
        self.reg_coef = kwargs["reg_coef"]
        self.online_reg = True

        self.score = []
        self.fisher = []
        self.n_fisher_sample = None
        self.empFI = False
        self.alpha = 0.5
        self.epoch_score = {}
        self.epoch_fisher = {}
        for n, p in self.params.items():
            self.epoch_score[n] = (p.clone().detach().fill_(0).to(self.device))
            self.epoch_fisher[n] = (p.clone().detach().fill_(0).to(self.device))

    def regularization_loss(
        self,
    ):
        reg_loss = 0
        if len(self.regularization_terms) > 0:
            # Calculate the reg_loss only when the regularization_terms exists
            for _, reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term["importance"]
                task_param = reg_term["task_param"]

                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()

                max_importance = 0
                max_param_change = 0
                for n, p in self.params.items():
                    max_importance = max(max_importance, importance[n].max())
                    max_param_change = max(
                        max_param_change, ((p - task_param[n]) ** 2).max()
                    )
                reg_loss += task_reg_loss
            reg_loss = self.reg_coef * reg_loss

        return reg_loss

    def online_train(self, sample, batch_size, iterations=1, stream_batch_size=1):
        self.model.train()
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(
                datalist=sample,
                cls_list=self.exposed_classes,
                data_dir=self.data_dir)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

        info = {}
        for i in range(iterations):
            self.model.train()

            data = []
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                data += stream_data['batch']
            if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                data += memory_data['batch']

            batch = [(self.model.load_task_json(task), swapColor) for task, swapColor in data]
            feat = self.model.featurize(batch)

            out = self.model.forward(feat)
            self.optimizer.zero_grad()

            old_params = {n: p.clone().detach() for n, p in self.params.items()}
            old_grads = {n: p.grad.clone().detach() for n, p in self.params.items() if p.grad is not None}

            loss = sum(self.model.compute_loss(out, batch, feat).values())
            if 'cls_loss' not in info:
                info['cls_loss'] = 0
            info['cls_loss'] += loss.item()

            reg_loss = self.regularization_loss()
            loss += reg_loss
            if 'reg_loss' not in info:
                info['reg_loss'] = 0
            info['reg_loss'] += reg_loss if type(reg_loss) == int else reg_loss.item()

            loss.backward()
            self.optimizer.step()

            self.update_schedule()
            new_params = {n: p.clone().detach() for n, p in self.params.items()}
            new_grads = {
                n: p.grad.clone().detach() for n, p in self.params.items() if p.grad is not None
            }
            self.update_fisher_and_score(new_params, old_params, new_grads, old_grads)

        info = {k: v / iterations for k, v in info.items()}
        return info

    def online_after_task(self, cur_iter):
        # 2.Backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()

        # 3.Calculate the importance of weights for current task
        importance = self.calculate_importance()

        # Save the weight and importance of weights of current task
        self.task_count += 1

        # Use a new slot to store the task-specific information
        if self.online_reg and len(self.regularization_terms) > 0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {
                "importance": importance,
                "task_param": task_param,
            }
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {
                "importance": importance,
                "task_param": task_param,
            }

    def update_fisher_and_score(self, new_params, old_params, new_grads, old_grads, epsilon=0.001):
        for n, _ in self.params.items():
            if n in old_grads:
                new_p = new_params[n]
                old_p = old_params[n]
                new_grad = new_grads[n]
                old_grad = old_grads[n]
                if torch.isinf(new_p).sum()+torch.isinf(old_p).sum()+torch.isinf(new_grad).sum()+torch.isinf(old_grad).sum():
                    continue
                if torch.isnan(new_p).sum()+torch.isnan(old_p).sum()+torch.isnan(new_grad).sum()+torch.isnan(old_grad).sum():
                    continue
                self.epoch_score[n] += (old_grad-new_grad) * (new_p - old_p) / (
                    0.5 * self.epoch_fisher[n] * (new_p - old_p) ** 2 + epsilon
                )
                if (self.epoch_fisher[n] == 0).all():  # First time
                    self.epoch_fisher[n] = new_grad ** 2
                else:
                    self.epoch_fisher[n] = (1 - self.alpha) * self.epoch_fisher[n] + self.alpha * new_grad ** 2

    def calculate_importance(self):
        importance = {}
        self.fisher.append(self.epoch_fisher)
        if self.task_count == 0:
            self.score.append(self.epoch_score)
        else:
            score = {}
            for n, p in self.params.items():
                score[n] = 0.5 * self.score[-1][n] + 0.5 * self.epoch_score[n]
            self.score.append(score)

        for n, p in self.params.items():
            importance[n] = self.fisher[-1][n]
            self.epoch_score[n] = self.params[n].clone().detach().fill_(0)
        return importance