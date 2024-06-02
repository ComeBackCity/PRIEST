import torch
from torch import nn
import torch.nn.functional as F
import math
# import gensim

EMBEDDING_SIZE = 100
NUM_FILTERS = 10

class CnnTextClassifier(nn.Module):
    def __init__(self, num_classes, window_sizes=(1,2,3,)):
        super(CnnTextClassifier, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(1, NUM_FILTERS, [window_size, EMBEDDING_SIZE], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        self.linear_final = nn.Linear(len(window_sizes) \
                      * NUM_FILTERS, num_classes)

        self.dropout_final = nn.Dropout(0.2)

        self.attention_weights = nn.Parameter(torch.ones(3))
        self.attention_weights.requires_grad = True

    def forward(self, src, debug=False):
        # x = self.embedding(x) # [B, T, E]

        # print(x.shape)

        # Apply a convolution + max_pool layer for each window size
        # src = x
        attention_outputs = [self.attention(src)[0], self.additive_attention(src, src, src)[0], \
            self.scaled_dot_product_attention(src, src, src)[0]]

        attention_weights = self.attention_weights / self.attention_weights.sum()
        weighted_attention_outputs = [attention_outputs[i] * attention_weights[i] for i in range(3)]
        attention_output = sum(weighted_attention_outputs)

        src = src.add(attention_output)
        src = src.unsqueeze(1)
    
        # src = self.attention(src)[0]

        conved = [F.relu(conv(src)).squeeze(3) 
                  for conv in self.convs]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) 
                  for conv in conved]

        cat = self.dropout_final(torch.cat(pooled, dim = 1))
        output = self.linear_final(cat)

        return output, 0

    def attention(self, x):
        # x is of shape (batch_size, seq_len, num_filters)
        # Compute the dot product between each pair of elements
        attention_scores = torch.matmul(x, x.transpose(-1, -2))
        # Normalize the dot product with a softmax function
        attention_scores = F.softmax(attention_scores, dim=-1)
        # Use the attention scores to weight the element-wise product of the original sequence
        weighted_sum = torch.matmul(attention_scores, x)
        return weighted_sum, attention_scores

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        d_k = k.shape[-1]
        # Scale the dot product of the queries and keys by the square root of the key dimension
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        # Optionally apply a mask to the attention scores
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        # Normalize the attention scores with a softmax function
        attention_weights = F.softmax(attention_scores, dim=-1)
        # Use the attention weights to weight the element-wise product of the values
        weighted_sum = torch.matmul(attention_weights, v)
        return weighted_sum, attention_weights

    def additive_attention(self, q, k, v):
        # Reshape q and k into shapes [batch_size, Tq, 1, dim] and [batch_size, 1, Tv, dim] respectively
        q = q.unsqueeze(2)
        k = k.unsqueeze(1)
        # Calculate scores with shape [batch_size, Tq, Tv] as a non-linear sum: scores = tf.reduce_sum(tf.tanh(query + key), axis=-1)
        scores = torch.sum(torch.tanh(q + k), dim=-1)
        # Use scores to calculate a distribution with shape [batch_size, Tq, Tv]: distribution = tf.nn.softmax(scores)
        distribution = F.softmax(scores, dim=-1)
        # Use distribution to create a linear combination of value with shape [batch_size, Tq, dim]: return tf.matmul(distribution, value)
        weighted_sum = torch.matmul(distribution, v)
        return weighted_sum, distribution





