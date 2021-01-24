from resnet import resnet50 
from main import create_dataset
from gpt.py import CrossEntropyLoss
from mindspore import Model, load_checkpoint
from mindspore.nn.metrics import Accuracy
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

def test_net(network, data_path):
    """define the evaluation method"""
    print("============== Starting Testing ==============")
    #load the saved model for evaluation
    load_checkpoint("checkpoint_resnet_cifar10-1_50.ckpt", net=network)
    #load testing dataset
    ds_eval = create_dataset(False, data_path)

    # config = GPTConfig(batch_size=4,
    #                    seq_length=1024,
    #                    vocab_size=50257,
    #                    embedding_size=1024,
    #                    num_layers=24,
    #                    num_heads=16,
    #                    expand_ratio=4,
    #                    post_layernorm_residual=False,
    #                    dropout_rate=0.1,
    #                    compute_dtype=mstype.float16,
    #                    use_past=False) 
    # loss = CrossEntropyLoss(config)

    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model = Model(resnet, net_loss, metrics={"Accuracy": Accuracy()}, amp_level="O3")
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))

if __name__ == '__main__':
    resnet = resnet50()
    testing_path = r'./cifar-testing'
    test_net(resnet, testing_path)
