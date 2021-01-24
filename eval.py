from resnet import resnet50 
from main import create_dataset
from gpt.py import CrossEntropyLoss

def test_net(network, data_path):
    """define the evaluation method"""
    print("============== Starting Testing ==============")
    #load the saved model for evaluation
    load_checkpoint("checkpoint_resnet_cifar10_50.ckpt", net=network)
    #load testing dataset
    ds_eval = create_dataset(False, data_path) 
    loss = CrossEntropyLoss()
    model = Model(resnet, loss, metrics={"Accuracy": Accuracy()})
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))

if __name__ == '__main__':
    resnet = resnet50()
    testing_path = './'
    test_net(resnet, testing_path)
