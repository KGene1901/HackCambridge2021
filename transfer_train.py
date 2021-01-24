import os
import argparse
from mindspore import context
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore import dtype as mstype
import mindspore.nn as nn
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore import Model
from mindspore import load_checkpoint, load_param_into_net
from model_zoo.official.cv.resnet.src.resnet import resnet50 
from mindspore.train.callback import Callback  

def transfer_train(chkpt, opt):
	param_dict = load_checkpoint("resnet50-2_32.ckpt")
	resnet = resnet50()
	# load the parameter into net
	load_param_into_net(resnet, param_dict)
	# load the parameter into optimizer
	load_param_into_net(opt, param_dict)
	loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
	model = Model(resnet, loss, opt)
	model.train(epoch, dataset)

if __name__ == '__main__':
    
    # Initialise important training params
    parser = argparse.ArgumentParser(description='MindSpore Resnet50 Training')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'], help='device where the code will be implemented (default: CPU)')
    parser.add_argument('--chkpt', type=str)
    args = parser.parse_args()
    if args.debug:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    # this next line can't be ran on a CPU
    # context.set_context(enable_graph_kernel=True)
    dataset_sink_mode = not args.device_target == "CPU"
    learning_r = 0.01
    momentum = 0.9
    epoch_size = 3
    dataset_size = 1
    eval_per_epoch = 2

    # loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # optimisation function
    net_opt = nn.Momentum(filter(lambda x: x.requires_grad, resnet.get_parameters()), learning_r, momentum)

    # set params at checkpoint
    config_ck = CheckpointConfig(save_checkpoint_steps=10, keep_checkpoint_max=5)
    # apply params at checkpoint
    ckpoint = ModelCheckpoint(prefix="checkpoint_resnet_cifar10", config=config_ck)

    transfer_train(args.chkpt, net_opt, dataset_sink_mode)