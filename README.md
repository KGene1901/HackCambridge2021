# **HackCambridge2021 - Huawei Image Classification Challenge**

## Model Training
For this challenge, we have utilised `resnet50()` as our image classification model (it is imported from Mindspore's Github page inside the model zoo). We have managed to implement some optimisation code after reading up on the documentation and example codes provided by Mindspore but many of which couldn't really be tested - this included graph kernel fusion and enabling automated mixed precision. To create the training dataset, we have randomly cropped and shuffled the training data in order to prevent the model from following a trend. We have also resized and normalise the data to ensure to reduce any non-uniformity of data.

## Evaluation:
Due to hardware factors (i.e. our laptops generally kept freezing or slowing down when running training scripts) and time constraints, we could only achieve an accurary of about 27-28. We used SoftmaxCrossEntropyWithLogits as a loss function when building the model up again with the checkpoint made before running the evaluation.

