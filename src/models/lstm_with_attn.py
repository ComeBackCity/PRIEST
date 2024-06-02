import torch
from torch import nn
import torch.nn.functional as F
import math
# import gensim

class LSTMSeriesClassifier(nn.Module):
    def __init__(self, seq_length, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTMSeriesClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.bn11 = torch.nn.BatchNorm1d(num_features=input_size)
        self.bn12 = torch.nn.BatchNorm1d(num_features=input_size)
        self.bn13 = torch.nn.BatchNorm1d(num_features=input_size)

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.lstm3 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)

        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size)

        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout_final = nn.Dropout(0.3)

        self.attention_weights = nn.Parameter(torch.ones(3, seq_length, input_size))
        self.cat_weights = nn.Parameter(torch.ones(3, seq_length, hidden_size))
        self.attention_weights.requires_grad = True
        self.cat_weights.requires_grad = True 
        self.device = device

    def forward(self, src, debug=False):
        attention_outputs = [self.attention(src)[0], self.additive_attention(src, src, src)[0], \
            self.scaled_dot_product_attention(src, src, src)[0]]

        attention_weights = self.attention_weights / self.attention_weights.sum(dim=0)
        weighted_attention_outputs = [attention_outputs[i] * attention_weights[i] for i in range(3)]
        attention_output = sum(weighted_attention_outputs)

        line1 = src.transpose(1, 2)
        line2 = torch.add(src, attention_output).transpose(1, 2)
        line3 = attention_output.transpose(1, 2)

        line1 = self.bn11(line1).transpose(1, 2)
        line2 = self.bn12(line2).transpose(1, 2)
        line3 = self.bn13(line3).transpose(1, 2)

        h1 = torch.zeros(self.num_layers, src.size(0), self.hidden_size).to(self.device)
        c1 = torch.zeros(self.num_layers, src.size(0), self.hidden_size).to(self.device)

        h2 = torch.zeros(self.num_layers, src.size(0), self.hidden_size).to(self.device)
        c2 = torch.zeros(self.num_layers, src.size(0), self.hidden_size).to(self.device)

        h3 = torch.zeros(self.num_layers, src.size(0), self.hidden_size).to(self.device)
        c3 = torch.zeros(self.num_layers, src.size(0), self.hidden_size).to(self.device)
    
        line1, _ = self.lstm1(line1, (h1, c1))     
        line2, _ = self.lstm2(line2, (h2, c2)) 
        line3, _ = self.lstm3(line3, (h3, c3))  

        cat_weights = self.cat_weights / self.cat_weights.sum(dim=0)
        cat = line1 * cat_weights[0] + line2 * cat_weights[1] + line3 * cat_weights[2]
        cat = cat.transpose(1, 2)
        cat = self.bn2(cat).transpose(1, 2)

        # cat = self.dropout_final(torch.cat(pooled, dim = 1))
        src = self.dropout_final(cat[:, -1, :])
        output = self.fc(src)

        return output, 0

    def attention(self, x):
        # x is of shape (batch_size, seq_len, num_filters)
        # Compute the dot product between each pair of elements
        attention_scores = torch.matmul(x, x.transpose(-1, -2))
        # Normalize the dot product with a softmax function
        mask = torch.tril(torch.ones_like(attention_scores))
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = F.softmax(attention_scores, dim=-1)
        # Use the attention scores to weight the element-wise product of the original sequence
        weighted_sum = torch.matmul(attention_scores, x)
        return weighted_sum, attention_scores

    def scaled_dot_product_attention(self, q, k, v):
        d_k = k.shape[-1]
        # Scale the dot product of the queries and keys by the square root of the key dimension
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        # Optionally apply a mask to the attention scores
        mask = torch.tril(torch.ones_like(attention_scores))
        # Mask the attention scores with the created mask
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
        # Create a mask with 1s for all timesteps before the current timestep, and 0s for all timesteps after the current timestep
        mask = torch.tril(torch.ones_like(scores))
        # Mask the scores with the created mask
        scores = scores.masked_fill(mask == 0, -1e9)
        # Use scores to calculate a distribution with shape [batch_size, Tq, Tv]: distribution = tf.nn.softmax(scores)
        distribution = F.softmax(scores, dim=-1)
        # Use distribution to create a linear combination of value with shape [batch_size, Tq, dim]: return tf.matmul(distribution, value)
        weighted_sum = torch.matmul(distribution, v)
        return weighted_sum, distribution





