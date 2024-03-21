import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.data_loader import StreamDataset, MemoryDataset


class ER:
    def __init__(self, n_classes, model, **kwargs):
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.n_classes = n_classes
        self.exposed_classes = []
        self.seen = 0

        self.device = torch.device("cuda")
        self.model_name = kwargs["model_name"]
        self.opt_name = kwargs["opt_name"]
        self.sched_name = kwargs["sched_name"]
        if self.sched_name == "default":
            self.sched_name = 'exp_reset'
        self.lr = kwargs["lr"]

        self.memory_size = kwargs["memory_size"]
        self.data_dir = kwargs["data_dir"]

        self.online_iter = kwargs["online_iter"]
        self.batch_size = kwargs["batchsize"]
        self.temp_batchsize = kwargs["temp_batchsize"]
        if self.temp_batchsize is None:
            self.temp_batchsize = self.batch_size//2
        if self.temp_batchsize > self.batch_size:
            self.temp_batchsize = self.batch_size
        self.memory_size -= self.temp_batchsize

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_gamma = 0.9999
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda iter: 1)

        self.memory = MemoryDataset(cls_list=self.exposed_classes, data_dir=self.data_dir)
        self.temp_batch = []
        self.num_updates = 0
        self.train_count = 0
        self.batch_size = kwargs["batchsize"]

        self.writer = SummaryWriter(log_dir=kwargs['dout'])

    def online_step(self, sample, sample_num):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.temp_batch.append(sample)
        self.num_updates += self.online_iter

        if len(self.temp_batch) == self.temp_batchsize:
            info = self.online_train(
                self.temp_batch, self.batch_size,
                iterations=int(self.num_updates),
                stream_batch_size=self.temp_batchsize
            )
            self.report_training(sample_num, info)
            for stored_sample in self.temp_batch:
                self.update_memory(stored_sample)
            self.temp_batch = []
            self.num_updates -= int(self.num_updates)

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        self.memory.add_new_class(cls_list=self.exposed_classes)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_train(self, sample, batch_size, iterations=1, stream_batch_size=1):
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(
                datalist=sample,
                cls_list=self.exposed_classes,
                data_dir=self.data_dir
            )
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
            loss = self.model.compute_loss(out, batch, feat)
            sum_loss = sum(loss.values())
            if 'cls_loss' not in info:
                info['cls_loss'] = 0
            info['cls_loss'] += sum_loss.item()
            sum_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            self.update_schedule()

        info = {k: v / iterations for k, v in info.items()}
        return info


    def report_training(self, sample_num, info):
        for k in info:
            self.writer.add_scalar(f'train/{k}', info[k], sample_num)

        # memory / frame sizes
        self.writer.add_scalar('train/memory_size', len(self.memory.images), sample_num)
        self.writer.add_scalar('train/frame_size', sum([m['num_frames'] for m in self.memory.images]), sample_num)

    def report_test(self, sample_num, eval_dict, tag=None):
        for k in eval_dict:
            self.writer.add_scalar(f"{tag}/{k}", eval_dict[k], sample_num)

        # memory / frame sizes
        self.writer.add_scalar('train/memory_size', len(self.memory.images), sample_num)
        self.writer.add_scalar('train/frame_size', sum([m['num_frames'] for m in self.memory.images]), sample_num)

    def update_memory(self, sample):
        self.reservoir_memory(sample)

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda iter: 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()

    def online_evaluate(self, test_list, sample_num, batch_size, tag=None):
        eval_dict = self.evaluation(test_list, sample_num, batch_size)
        self.report_test(sample_num, eval_dict, tag=tag)
        return eval_dict

    def online_before_task(self, cur_iter):
        # Task-Free
        pass

    def online_after_task(self, cur_iter):
        # Task-Free
        pass

    def reservoir_memory(self, sample):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)
        else:
            self.memory.replace_sample(sample)

    def reset_opt(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda iter: 1)

    def evaluation(self, test_list, sample_num, batch_size=32):
        eval_dict = {}
        p_valid, _, total_valid_loss, m_valid = self.model.run_pred(test_list, batch_size=batch_size, iter=sample_num)
        eval_dict['cls_loss'] = float(total_valid_loss)
        eval_dict.update(self.model.compute_metric(p_valid, test_list))
        return eval_dict

    def _interpret_pred(self, y, pred):
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects

