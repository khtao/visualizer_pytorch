import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import torch
import time
import cv2
import numpy as np


def create_file(filename):
    filename += time.strftime('_%m%d_%H%M%S')
    if os.path.exists(filename):
        i = 1
        while os.path.exists(filename + '_' + str(i)):
            i += 1
        filename = filename + '_' + str(i)
    return filename


def create_image(filename):
    if os.path.exists(filename):
        i = 1
        while os.path.exists(filename[:-4] + '_' + str(i) + filename[-4:]):
            i += 1
        filename = filename[:-4] + '_' + str(i) + filename[-4:]
    return filename


class Logger(object):
    def __init__(self, filename='running.log', stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()


class Visualizer(object):
    def __init__(self, env='default'):
        self.vis = create_file(os.path.join('visualizer', env))
        self.log_dir = os.path.join(self.vis, 'log')
        self.image_dir = os.path.join(self.vis, 'image')
        self.loss_dir = os.path.join(self.vis, 'loss')
        matplotlib.use('Agg')
        os.makedirs(self.log_dir)
        os.makedirs(self.image_dir)
        os.makedirs(self.loss_dir)
        self.loss_fs = open(os.path.join(self.loss_dir, 'loss.txt'), 'a')
        sys.stdout = Logger(os.path.join(self.log_dir, 'running.log'), stream=sys.stdout)
        sys.stderr = Logger(os.path.join(self.log_dir, 'system.log'), stream=sys.stderr)
        self.index = {}
        self.data = {}

    @staticmethod
    def print_args(args):
        for k, v in args.__dict__.items():
            print(k, '=', v)

    def plot_many(self, d):
        """
        plot multi values
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            if v is not None:
                self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        if name in self.data.keys():
            self.data[name].append([x, y])
        else:
            self.data[name] = [[x, y]]
        self.loss_fs.write(f'{name}: {x}, {y}\n')
        self.loss_fs.flush()
        line_data = np.array(self.data[name])
        plt.plot(line_data[:, 0], line_data[:, 1], label=name)
        plt.title(name)
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.loss_dir, name + '.jpg'))
        plt.close()
        self.index[name] = x + 1

    def plot_many_in_one(self, name, d):
        """
        self.plot('loss',1.00)
        """
        for k, v in d.items():
            x = self.index.get(k, 0)
            if k in self.data.keys():
                self.data[k].append([x, v])
            else:
                self.data[k] = [[x, v]]
            self.loss_fs.write(f'{k}: {x}, {v}\n')
            line_data = np.array(self.data[k])
            self.index[k] = x + 1
            plt.plot(line_data[:, 0], line_data[:, 1], label=k)
        plt.title(name)
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.loss_dir, name + '.jpg'))
        plt.close()

    def img(self, name, img_):
        file_name = create_image(os.path.join(self.image_dir, name + '.jpg'))
        if type(img_) is figure.Figure:
            img_.savefig(file_name)
            plt.close(img_)
        else:
            # If list of images, convert to a 4D tensor
            if isinstance(img_, torch.Tensor):
                img_ = img_.detach().cpu()

            if isinstance(img_, list):
                img_ = np.stack(img_, 0)
            if img_.ndim == 2:  # single image H x W
                img_ = np.expand_dims(img_, 0)
            if img_.ndim == 3:  # single image
                if img_.shape[0] == 1:  # if single-channel, convert to 3-channel
                    img_ = np.repeat(img_, 3, 0)
            cv2.imwrite(file_name, (img_.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))


if __name__ == '__main__':
    from tqdm import tqdm

    fig = plt.figure()
    plt.plot([0, 1, 2, 3], [0, 1, 2, 3])
    plt.title('ssss')
    vis = Visualizer()
    vis.img('fig', fig)
    for i in tqdm(range(10)):
        time.sleep(1)
        pass
    vis.plot_many_in_one('many_loss', {'d1': 0.1, 'd2': 0.05})
    vis.plot_many_in_one('many_loss', {'d1': 0.2, 'd2': 0.1})
    vis.plot_many_in_one('many_loss', {'d1': 0.8, 'd2': 0.3})
    vis.plot('loss', 0.1)
    vis.plot('loss', 0.2)
    vis.plot('loss', 0.5)
    vis.img('000', np.zeros((256, 256), dtype=float))

    print('ok')
    print('hahah')
