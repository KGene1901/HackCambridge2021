'''
Tutorial followed: 
https://www.mindspore.cn/tutorial/training/en/r1.1/quick_start/quick_start.html
https://www.mindspore.cn/tutorial/training/en/r1.0/advanced_use/cv_resnet50.html
'''
import os
import argparse
from mindspore import context
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore import dtype as mstype
import mindspore.nn as nn
from mindspore.common.initializer import Normal
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore import Model
from mindspore import load_checkpoint, load_param_into_net
from model_zoo.official.cv.resnet.src.resnet import resnet50 
from mindspore.train.callback import Callback 

class Evalcb(cb):
    def __init__(self, model, eval_ds, eval_per_epoch, epoch_per_eval):
        self.model = model
        self.eval_dataset = eval_ds
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = epoch_per_eval

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            self.epoch_per_eval["epoch"].append(cur_epoch)
            self.epoch_per_eval["acc"].append(acc["Accuracy"])
            print(acc)

def create_dataset(training, data_path, batch_size=32, repeat_size=1, num_parallel_workers=1):
    # define dataset
    cifar_ds = ds.Cifar10Dataset(data_path)

    # define operation parameters
    resize_height, resize_width = 224, 224
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    random_crop_op = C.RandomCrop((32, 32), (4, 4, 4, 4)) # padding_mode default CONSTANT
    random_horizontal_op = C.RandomHorizontalFlip()
    resize_op = C.Resize((resize_height, resize_width)) # interpolation default BILINEAR
    rescale_op = C.Rescale(rescale, shift)
    normalize_op = C.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = C.HWC2CHW()
    type_cast_op = C2.TypeCast(mstype.int32)

    c_trans = []

    if training:
        c_trans = [random_crop_op, random_horizontal_op]

    c_trans += [resize_op, rescale_op, normalize_op, changeswap_op]

    # apply map operations on images
    cifar_ds = cifar_ds.map(operations=type_cast_op, input_columns="label")
    cifar_ds = cifar_ds.map(operations=c_trans, input_columns="image")

    # apply shuffle ops
    cifar_ds = cifar_ds.shuffle(buffer_size=10)

    # apply batch ops
    cifar_ds = cifar_ds.batch(batch_size=batch_size, drop_remainder=True)

    # apply repeat operators
    cifar_ds = cifar_ds.repeat(repeat_size)

    return cifar_ds

def train_net(epoch_size, data_path, eval_per_epoch, repeat_size, ckpoint_cb, sink_mode):
    """define the training method"""
    print("============== Starting Training ==============")
    # Create training dataset
    ds_train = create_dataset(True, training_path, 32, repeat_size)
    # Initialise model
    model = Model(resnet, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    # model = Model(resnet, net_loss, net_opt, metrics={"Accuracy": Accuracy()}, amp_level="O3") # this will not work for CPU
    epoch_per_eval = {"epoch":[], "acc":[]}
    eval_cb =Evalcb(model, ds_train, eval_per_epoch, epoch_per_eval)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(), eval_cb], dataset_sink_mode=sink_mode)

if __name__ == '__main__':
    
    # Initialise important training params
    parser = argparse.ArgumentParser(description='MindSpore Resnet50 Training')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'], help='device where the code will be implemented (default: CPU)')
    parser.add_argument('--datapath', type=str)
    args = parser.parse_args()
    if args.debug:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    # this next line can't be ran on a CPU
    context.set_context(enable_graph_kernel=True)
    dataset_sink_mode = not args.device_target == "CPU"
    learning_r = 0.01
    momentum = 0.9
    epoch_size = 4
    training_path = args.datapath
    dataset_size = 1
    eval_per_epoch = 2

    # define network to use
    resnet = resnet50()

    # loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # optimisation function
    net_opt = nn.Momentum(filter(lambda x: x.requires_grad, resnet.get_parameters()), learning_r, momentum)

    # set params at checkpoint
    config_ck = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=10)
    # apply params at checkpoint
    ckpoint = ModelCheckpoint(prefix="checkpoint_resnet_cifar10", config=config_ck)

    train_net(epoch_size, training_path, eval_per_epoch, dataset_size, ckpoint, dataset_sink_mode)