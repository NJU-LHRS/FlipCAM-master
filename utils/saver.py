import os
import shutil
import torch
from collections import OrderedDict
from utils import pyutils
import glob

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)              # 模型保存根目录
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0              # 这种方法可以保证训练不会覆盖之前的结果

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))     # 保存路径
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        pyutils.Logger(self.experiment_dir + '/print.log')



    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        # models = glob.glob(os.path.join(self.experiment_dir, 'checkpoint_*.pth.tar'))
        # model_id = sorted([(int(i.split('.')[0].split('_')[-1]) + 1) for i in models])[-1] if models else 0
        filename = os.path.join(self.experiment_dir, 'checkpoint_{}.pth.tar'.format(state['epoch']))
        torch.save(state, filename)
        # if is_best:
        #     best_pred = state['best_pred']
        #     with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
        #         f.write(str(best_pred))
        #     if self.runs:
        #         previous_miou = [0.0]
        #         for run in self.runs:
        #             run_id = run.split('_')[-1]
        #             path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
        #             if os.path.exists(path):
        #                 with open(path, 'r') as f:
        #                     miou = float(f.readline())
        #                     previous_miou.append(miou)
        #             else:
        #                 continue
        #         max_miou = max(previous_miou)
        #         if best_pred > max_miou:
        #             shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
        #     else:
        #         shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        # p = OrderedDict()           # 有序字典
        # p['data_root'] = self.args.data_root
        # p['size'] = self.args.size

        p = vars(self.args)     # args转为列表

        for key, val in p.items():
            log_file.write(key + ': ' + str(val) + '\n')
        log_file.close()