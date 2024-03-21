import torch
import numpy as np
from torch.utils.data import Dataset

import random
random.seed(100)

from collections import defaultdict


class StreamDataset(Dataset):
    def __init__(self, datalist, cls_list, data_dir=None):
        self.images = []
        self.labels = []
        self.cls_list = cls_list
        self.data_dir = data_dir

        for data in datalist:
            self.images.append(data)
            self.labels.append(self.cls_list.index(data['klass']))

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def get_data(self):
        data = dict()
        data['batch'] = [(img['task'], random.choice(range(7))) for img in self.images]
        return data


class MemoryDataset(Dataset):
    def __init__(self, cls_list=None, data_dir=None, save_test=None, keep_history=False):
        self.cls_seen_count = defaultdict(int)

        self.datalist = []
        self.labels = []
        self.images = []
        self.cls_list = []
        self.cls_dict = {cls_list[i]:i for i in range(len(cls_list))}
        self.cls_count = []
        self.cls_idx = []
        self.cls_train_cnt = np.array([])
        self.score = []
        self.others_loss_decrease = np.array([])
        self.previous_idx = np.array([], dtype=int)
        self.data_dir = data_dir
        self.keep_history = keep_history

        self.save_test = save_test
        if self.save_test is not None:
            self.device_img = []

    def __len__(self):
        return len(self.images)

    def add_new_class(self, cls_list):
        self.cls_list = cls_list
        self.cls_count.append(0)
        self.cls_idx.append([])
        self.cls_dict = {self.cls_list[i]:i for i in range(len(self.cls_list))}
        self.cls_train_cnt = np.append(self.cls_train_cnt, 0)

    def update_gss_score(self, score, idx=None):
        if idx is None:
            self.score.append(score)
        else:
            self.score[idx] = score

    def replace_sample(self, sample, idx=None):
        self.cls_count[self.cls_dict[sample['klass']]] += 1
        if idx is None:
            self.cls_idx[self.cls_dict[sample['klass']]].append(len(self.images))
            self.datalist.append(sample)
            self.images.append(sample)
            self.labels.append(self.cls_dict[sample['klass']])
            if self.save_test is not None:
                self.device_img.append(sample)
            if self.cls_count[self.cls_dict[sample['klass']]] == 1:
                self.others_loss_decrease = np.append(self.others_loss_decrease, 0)
            else:
                self.others_loss_decrease = np.append(self.others_loss_decrease, np.mean(self.others_loss_decrease[self.cls_idx[self.cls_dict[sample['klass']]][:-1]]))
        else:
            self.cls_count[self.labels[idx]] -= 1
            self.cls_idx[self.labels[idx]].remove(idx)
            self.datalist[idx] = sample
            self.cls_idx[self.cls_dict[sample['klass']]].append(idx)
            self.images[idx] = sample #img
            self.labels[idx] = self.cls_list.index(sample['klass'])
            if self.save_test is not None:
                self.device_img[idx] = sample #img
            if self.cls_count[self.cls_dict[sample['klass']]] == 1:
                self.others_loss_decrease[idx] = np.mean(self.others_loss_decrease)
            else:
                self.others_loss_decrease[idx] = np.mean(self.others_loss_decrease[self.cls_idx[self.cls_dict[sample['klass']]][:-1]])

    def get_weight(self):
        weight = np.zeros(len(self.images))
        for i, indices in enumerate(self.cls_idx):
            weight[indices] = 1/self.cls_count[i]
        return weight

    @torch.no_grad()
    def get_batch(self, batch_size, use_weight=False, transform=None):
        if use_weight:
            weight = self.get_weight()
            indices = np.random.choice(range(len(self.images)), size=batch_size, p=weight/np.sum(weight), replace=False)
        else:
            indices = np.random.choice(range(len(self.images)), size=batch_size, replace=False)
        data = dict()
        batch = []
        for i in indices:
            batch.append((self.images[i]['task'], random.choice(range(7))))
            self.cls_train_cnt[self.labels[i]] += 1
        data['batch'] = batch
        if self.keep_history:
            self.previous_idx = np.append(self.previous_idx, indices)
        return data

    def update_loss_history(self, loss, prev_loss, ema_ratio=0.90, dropped_idx=None):
        if dropped_idx is None:
            loss_diff = np.mean(loss - prev_loss)
        elif len(prev_loss) > 0:
            mask = np.ones(len(loss), bool)
            mask[dropped_idx] = False
            loss_diff = np.mean((loss[:len(prev_loss)] - prev_loss)[mask[:len(prev_loss)]])
        else:
            loss_diff = 0
        difference = loss_diff - np.mean(self.others_loss_decrease[self.previous_idx]) / len(self.previous_idx)
        self.others_loss_decrease[self.previous_idx] -= (1 - ema_ratio) * difference
        self.previous_idx = np.array([], dtype=int)

    def get_two_batches(self, batch_size):
        indices = np.random.choice(range(len(self.images)), size=batch_size, replace=False)

        data_1 = dict()
        data_2 = dict()

        data_1['batch'] = [(self.images[i]['task'], random.choice(range(7))) for i in indices]
        data_2['batch'] = [(self.images[i]['task'], 0) for i in indices]

        return data_1, data_2


class DistillationMemory(MemoryDataset):
    def __init__(self, cls_list=None, data_dir=None, save_test=None, keep_history=False, use_logit=True, use_feature=False):
        super().__init__(cls_list, data_dir, save_test, keep_history)
        self.logits = []
        self.features = []
        self.logits_mask = []
        self.use_logit = use_logit
        self.use_feature = use_feature

        self.stream_images = []
        self.stream_labels = []

    def register_stream(self, datalist):
        self.stream_images = []
        self.stream_labels = []
        for data in datalist:
            self.stream_images.append(data)
            self.stream_labels.append(self.cls_dict[data['klass']])

    def remove_logit(self, idx):
        self.logits.pop(idx)

    def save_logit(self, logit, idx=None):
        if idx is None:
            self.logits.append(logit)
        else:
            self.logits[idx] = logit

    def save_feature(self, feature, idx=None):
        if idx is None:
            self.features.append(feature)
        else:
            self.features[idx] = feature

    def add_new_class(self, cls_list):
        self.cls_list = cls_list
        self.cls_count.append(0)
        self.cls_idx.append([])
        self.cls_dict = {self.cls_list[i]:i for i in range(len(self.cls_list))}
        self.cls_train_cnt = np.append(self.cls_train_cnt, 0)

    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=False, transform=None):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_images))
        batch_size = min(batch_size, stream_batch_size + len(self.images))
        memory_batch_size = batch_size - stream_batch_size
        if memory_batch_size > 0:
            if use_weight:
                weight = self.get_weight()
                indices = np.random.choice(range(len(self.images)), size=memory_batch_size, p=weight / np.sum(weight),
                                           replace=False)
            else:
                indices = np.random.choice(range(len(self.images)), size=memory_batch_size, replace=False)
        if stream_batch_size > 0:
            if len(self.stream_images) > stream_batch_size:
                stream_indices = np.random.choice(range(len(self.stream_images)), size=stream_batch_size, replace=False)
            else:
                stream_indices = np.arange(len(self.stream_images))

        data = dict()
        data['batch'] = []
        if stream_batch_size > 0:
            data['batch'] += [(self.stream_images[i]['task'], random.choice(range(7))) for i in stream_indices]
        if memory_batch_size > 0:
            data['batch'] += [(self.images[i]['task'], random.choice(range(7))) for i in indices]
            if self.use_logit:
                data['logit'] = [self.logits[i] for i in indices]

        if self.keep_history:
            self.previous_idx = np.append(self.previous_idx, indices)
        return data


class DistillationMemory_XDER(DistillationMemory):
    def __init__(self, cls_list=None, data_dir=None, save_test='cpu', keep_history=False, use_logit=True, use_feature=False, use_task_id=True):
        super().__init__(cls_list, data_dir, save_test, keep_history)
        self.logits = []
        self.task_ids = []
        self.features = []
        self.logits_mask = []
        self.use_logit = use_logit
        self.use_feature = use_feature
        self.use_task_id = use_task_id

    def update_logits(self, indices, new_logits):
        for new_logit, indice in zip(new_logits, indices):
            self.logits[indice] = new_logit

    def update_task_ids(self, indices, new_task_id):
        for indice in indices:
            self.task_ids[indice] = new_task_id

    def save_task_id(self, task_id, idx=None):
        if idx is None:
            self.task_ids.append(task_id)
        else:
            self.task_ids[idx] = task_id

    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=False, transform=None, get_not_aug_img = False):
        indices= []
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_images))
        batch_size = min(batch_size, stream_batch_size + len(self.images))
        memory_batch_size = batch_size - stream_batch_size
        if memory_batch_size > 0:
            if use_weight:
                weight = self.get_weight()
                indices = np.random.choice(range(len(self.images)), size=memory_batch_size, p=weight / np.sum(weight),
                                           replace=False)
            else:
                indices = np.random.choice(range(len(self.images)), size=memory_batch_size, replace=False)
        if stream_batch_size > 0:
            if len(self.stream_images) > stream_batch_size:
                stream_indices = np.random.choice(range(len(self.stream_images)), size=stream_batch_size, replace=False)
            else:
                stream_indices = np.arange(len(self.stream_images))

        data = dict()
        task_ids = []

        data['batch'] = []
        data['batch_noaug'] = []
        if stream_batch_size > 0:
            data['batch'] += [(self.stream_images[i]['task'], random.choice(range(7))) for i in stream_indices]
            data['batch_noaug'] += [(self.stream_images[i]['task'], False) for i in stream_indices]
        if memory_batch_size > 0:
            data['batch'] += [(self.images[i]['task'], random.choice(range(7))) for i in indices]
            data['batch_noaug'] += [(self.images[i]['task'], False) for i in indices]
            if self.use_logit:
                data['logit'] = [self.logits[i] for i in indices]

            for i in indices:
                if self.use_task_id:
                    task_ids.append(self.task_ids[i])

        data['indices'] = torch.LongTensor(indices)

        if memory_batch_size > 0:
            if self.use_task_id:
                data['task_id'] = torch.LongTensor(task_ids)
        else:
            if self.use_task_id:
                data['task_id'] = torch.zeros(1)

        if self.keep_history:
            self.previous_idx = np.append(self.previous_idx, indices)
        return data

