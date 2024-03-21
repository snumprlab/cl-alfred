import copy
import torch

from methods.er_baseline import ER
from utils.data_loader import StreamDataset


class MIR(ER):
    def __init__(self, n_classes, model, **kwargs):
        super().__init__(n_classes, model, **kwargs)
        self.cand_size = kwargs['mir_cands']

    def online_train(self, sample, batch_size, iterations=1, stream_batch_size=1):
        self.model.train()
        assert stream_batch_size > 0
        sample_dataset = StreamDataset(datalist=sample, cls_list=self.exposed_classes, data_dir=self.data_dir)

        info = {}
        for i in range(iterations):
            stream_data = sample_dataset.get_data()['batch']
            batch = [(self.model.load_task_json(task), swapColor) for task, swapColor in stream_data]
            feat = self.model.featurize(batch)
            out = self.model.forward(feat)
            loss = sum(self.model.compute_loss(out, batch, feat).values())
            self.optimizer.zero_grad()
            loss.backward()
            grads = {}
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                grads[name] = param.grad.data

            if len(self.memory) > 0:
                memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

                lr = self.optimizer.param_groups[0]['lr']
                new_model = copy.deepcopy(self.model)
                for name, param in new_model.named_parameters():
                    if name not in grads:
                        continue
                    param.data = param.data - lr * grads[name]

                memory_cands, memory_cands_test = self.memory.get_two_batches(min(self.cand_size, len(self.memory)))
                memory_cands_batch = [(self.model.load_task_json(task), swapColor) for task, swapColor in memory_cands_test['batch']]
                memory_cands_feat = self.model.featurize(memory_cands_batch)
                with torch.no_grad():
                    out_pre = self.model.forward(memory_cands_feat)
                    out_post = new_model.forward(memory_cands_feat)
                    pre_loss = self.model.compute_loss_unsummed(out_pre, memory_cands_batch, memory_cands_feat)
                    post_loss = new_model.compute_loss_unsummed(out_post, memory_cands_batch, memory_cands_feat)
                    scores = post_loss - pre_loss
                    scores = torch.tensor(scores)
                selected_samples = torch.argsort(scores, descending=True)[:memory_batch_size]
                batch = []
                batch = batch + [(self.model.load_task_json(task), swapColor) for task, swapColor in stream_data]
                batch = batch + [(self.model.load_task_json(task), swapColor) for task, swapColor in [memory_cands['batch'][bi] for bi in selected_samples.tolist()]]
                feat = self.model.featurize(batch)
            else:
                batch = [(self.model.load_task_json(task), swapColor) for task, swapColor in stream_data]
                feat = self.model.featurize(batch)

            self.optimizer.zero_grad()
            out = self.model.forward(feat)
            loss = sum(self.model.compute_loss(out, batch, feat).values())
            if 'cls_loss' not in info:
                info['cls_loss'] = 0
            info['cls_loss'] += loss.item()

            loss.backward()
            self.optimizer.step()
            self.update_schedule()

        info = {k: v / iterations for k, v in info.items()}
        return info