# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Link to github repo: https://github.com/mindspore-ai/mindspore/blob/ca053f0cbcafc2a0107bda6a7016140a01437d13/model_zoo/official/nlp/gpt/src/gpt.py

"""GPT model"""

import math
import numpy as np
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype
from mindspore.common.initializer import TruncatedNormal, initializer, Normal
from mindspore.ops import operations as P
from mindspore.ops import functional as F

class GPTConfig:
    """
    GPT config class which defines the model size
    """
    def __init__(self,
                 batch_size=32,
                 seq_length=1024,
                 vocab_size=50257,
                 embedding_size=768,
                 num_layers=12,
                 num_heads=12,
                 expand_ratio=4,
                 post_layernorm_residual=False,
                 dropout_rate=0.1,
                 compute_dtype=mstype.float16,
                 use_past=False):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.post_layernorm_residual = post_layernorm_residual
        self.dropout_rate = dropout_rate
        self.compute_dtype = compute_dtype
        self.use_past = use_past

class CrossEntropyLoss(nn.Cell):
    """
    Calculate the cross entropy loss
    Args:
        config(GPTConfig): the config of the network
    Inputs:
        logits: the output logits of the backbone
        label: the ground truth label of the sample
        input_mask: the mask indicating whether each position is a valid input
    Returns:
        loss: Tensor, the corrsponding cross entropy loss
    """
    def __init__(self, config):
        super(CrossEntropyLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.mean = P.ReduceMean()
        self.sum = P.ReduceSum()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.vocab_size = config.vocab_size

    def construct(self, logits, label, input_mask):
        logits = self.log_softmax(P.Cast()(logits, mstype.float32))
        label = P.Reshape()(label, (-1,))
        one_hot_label = self.onehot(label, self.vocab_size, self.on_value, self.off_value)
        loss_sum = P.Neg()(self.sum(logits*one_hot_label, (-1,)))
        input_mask = P.Reshape()(input_mask, (-1,))
        numerator = self.sum(loss_sum*input_mask)
        denominator = self.sum(input_mask) + P.Cast()(F.tuple_to_array((1e-5,)), mstype.float32)
        loss = numerator / denominator
        return loss
