import json
import pprint
import random
import time
import torch
import torch.multiprocessing as mp
from models.nn.resnet import Resnet
from data.preprocess import Dataset
from importlib import import_module

from gen import constants

class Eval(object):

    # tokens
    STOP_TOKEN = "<<stop>>"
    SEQ_TOKEN = "<<seg>>"
    TERMINAL_TOKENS = [STOP_TOKEN, SEQ_TOKEN]

    def __init__(self, args, manager):
        # args and manager
        self.args = args
        self.manager = manager

        # load splits
        with open(self.args.splits) as f:
            self.splits = json.load(f)
            pprint.pprint({k: len(v) for k, v in self.splits.items()})
            
            #############################################################################
            if self.args.incremental_setup in ['behavior_il']:
                # filtering based on task types
                print(self.args.stream_seed)
                task_types = constants.BEHAVIOR_TYPES[self.args.stream_seed]
                seen_tasks = task_types[:task_types.index(self.args.incremental_type) + 1]
                print('seen_tasks:', len(seen_tasks))
                self.splits['valid_seen'] = [t for t in self.splits['valid_seen'] if any(st in t['task'] for st in seen_tasks)]
                self.splits['valid_unseen'] = [t for t in self.splits['valid_unseen'] if any(st in t['task'] for st in seen_tasks)]
            elif self.args.incremental_setup in ['environment_il', 'environment_il_nosampling']:
                # filtering based on task types
                if self.args.incremental_setup == 'environment_il':
                    scene_types = constants.ENVIRONMENT_TYPES[self.args.stream_seed]
                elif self.args.incremental_setup == 'environment_il_nosampling':
                    scene_types = constants.IMBALANCED_ENVIRONMENT_TYPES[self.args.stream_seed]
                scene_types = scene_types[:scene_types.index(constants.ENVIRONMENT2NUM[self.args.incremental_type]) + 1]

                def task2scene(task):
                    return int(task['task'].split('/')[0].split('-')[-1]) // 100

                print('seen_scenes:', len(scene_types))
                # reload eval set for environment_il
                self.splits = {
                    'valid_seen': json.load(open(f'embodied_split/{self.args.incremental_setup}/valid_seen.json', 'r')),
                    'valid_unseen': json.load(open(f'embodied_split/{self.args.incremental_setup}/valid_unseen.json', 'r')),
                }
                self.splits['valid_seen'] = [t for t in self.splits['valid_seen'] if task2scene(t) in scene_types]
                self.splits['valid_unseen'] = [t for t in self.splits['valid_unseen'] if task2scene(t) in scene_types]
            else:
                print('Invalid incremental setup for evaluation:', self.args.incremental_setup)
                exit(0)
            
            pprint.pprint({k: len(v) for k, v in self.splits.items()})
            #############################################################################

        # load model
        print("Loading: ", self.args.model_path)
        M = import_module(self.args.model)
        self.model, optimizer = M.Module.load(self.args.model_path)
        self.model.share_memory()
        self.model.eval()
        self.model.test_mode = True

        # updated args
        self.model.args.dout = self.args.model_path.replace(self.args.model_path.split('/')[-1], '')
        self.model.args.data = self.args.data if self.args.data else self.model.args.data

        # preprocess and save
        if args.preprocess:
            print("\nPreprocessing dataset and saving to %s folders ... This is will take a while. Do this once as required:" % self.model.args.pp_folder)
            self.model.args.fast_epoch = self.args.fast_epoch
            dataset = Dataset(self.model.args, self.model.vocab)
            dataset.preprocess_splits(self.splits)

        # load resnet
        args.visual_model = 'resnet18'
        self.resnet = Resnet(args, eval=True, share_memory=True, use_conv_feat=True)

        # gpu
        if self.args.gpu:
            self.model = self.model.to(torch.device('cuda'))

        # success and failure lists
        self.create_stats()

        # set random seed for shuffling
        random.seed(int(time.time()))

    def queue_tasks(self):
        '''
        create queue of trajectories to be evaluated
        '''
        task_queue = self.manager.Queue()
        files = self.splits[self.args.eval_split]

        # debugging: fast epoch
        if self.args.fast_epoch:
            files = files[:16]

        if self.args.shuffle:
            random.shuffle(files)
        for traj in files:
            task_queue.put(traj)
        return task_queue

    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''
        task_queue = self.queue_tasks()

        # start threads
        threads = []
        lock = self.manager.Lock()
        for n in range(self.args.num_threads):
            thread = mp.Process(target=self.run, args=(self.model, self.resnet, task_queue, self.args, lock,
                                                       self.successes, self.failures, self.results))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        # save
        self.save_results()

    @classmethod
    def setup_scene(cls, env, traj_data, r_idx, args, reward_type='dense'):
        '''
        intialize the scene and agent from the task info
        '''
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        env.step(dict(traj_data['scene']['init_action']))

        # print goal instr
        print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

        # setup task for reward
        env.set_task(traj_data, args, reward_type=reward_type)

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures):
        raise NotImplementedError()

    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures):
        raise NotImplementedError()

    def save_results(self):
        raise NotImplementedError()

    def create_stats(self):
        raise NotImplementedError()
