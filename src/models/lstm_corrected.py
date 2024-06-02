import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
# import gensim


class LSTMSeriesClassifier(nn.Module):
    def __init__(self, seq_length, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTMSeriesClassifier, self).__init__()

        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.bn11 = torch.nn.BatchNorm1d(num_features=input_size)
        self.bn12 = torch.nn.BatchNorm1d(num_features=input_size)
        self.bn13 = torch.nn.BatchNorm1d(num_features=input_size)

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.lstm3 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)

        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_size)

        self.fc = nn.Linear(hidden_size, num_classes)

        self.dropout_final = nn.Dropout(0.3)

        self.attention_weights = nn.Parameter(torch.ones(3, seq_length, input_size))
        self.cat_weights = nn.Parameter(torch.ones(3,  seq_length, hidden_size))
        self.attn_scale = nn.Parameter(torch.ones(1))
        self.attention_weights.requires_grad = True
        self.cat_weights.requires_grad = True 
        self.attn_scale.requires_grad = True
        self.device = device

        self.fc1 = nn.Linear(in_features=seq_length * hidden_size, out_features=hidden_size)
        self.do = nn.Dropout(0.3)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, src, debug=False):
        attention_outputs = [
            self.attention(src)[0], \
            self.scaled_dot_product_attention(src, src, src, scale=self.attn_scale)[0], \
            self.additive_attention(src, src, src)[0] \
        ]

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
        # cat = line1 * cat_weights[0] + line3 * cat_weights[1]
        cat = cat.transpose(1, 2)
        cat = self.bn2(cat).transpose(1, 2)

        # cat = self.dropout_final(torch.cat(pooled, dim = 1))
        # src = self.dropout_final(cat[:, -1, :])
        # output = self.fc(src)

        out = torch.reshape(cat, (-1, self.seq_length * self.hidden_size))
        out = self.fc1(out)
        out = F.relu(self.do(out))
        output = self.fc2(out)

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

    def scaled_dot_product_attention(self, query, key, value, scale=None):
        """
        Compute scaled dot product attention given a query, key, and value.
        Args:
            query: tensor of shape (batch_size, query_length, embed_dim)
            key: tensor of shape (batch_size, key_length, embed_dim)
            value: tensor of shape (batch_size, value_length, embed_dim)
            scale: scale factor for the attention scores
        Returns:
            attention: tensor of shape (batch_size, query_length, embed_dim)
            weights: tensor of shape (batch_size, query_length, key_length)
        """
        # Define linear layers for query, key, and value
        k_dim = key.shape[-1]
        query_layer = nn.Linear(query.shape[-1], query.shape[-1], bias=False, device=self.device)
        key_layer = nn.Linear(key.shape[-1], key.shape[-1], bias=False, device=self.device)
        value_layer = nn.Linear(value.shape[-1], value.shape[-1], bias=False, device=self.device)
        
        # Apply linear layers to query, key, and value
        query = query_layer(query)
        key = key_layer(key)
        value = value_layer(value)
        
        # Transpose key and value to shape (batch_size, key_length, embed_dim)
        key = key.transpose(1, 2)
        
        # Compute dot product attention
        scores = torch.matmul(query, key) / math.sqrt(k_dim) # shape (batch_size, query_length, key_length)
        
        # Scale the scores if specified
        if scale is not None:
            scores = scores * scale

        mask = torch.tril(torch.ones_like(scores))
        scores = scores.masked_fill(mask == 0, -1e9)

        # Normalize the scores
        attention_scores = F.softmax(scores, dim=-1)  # shape (batch_size, query_length, key_length)

        # Compute the weighted sum of the values
        weighted_sum = torch.matmul(attention_scores, value)  # shape (batch_size, query_length, embed_dim)

        return weighted_sum, attention_scores

    def additive_attention(self, query, key, value):
        "Calculate Attention"
        # query: [batch_size, Tq, dim]
        # key: [batch_size, Tv, dim]
        # value: [batch_size, Tv, dim]
        query = query.unsqueeze(2)  # [batch_size, Tq, 1, dim]
        key = key.unsqueeze(1)  # [batch_size, 1, Tv, dim]
        scores = torch.sum(torch.tanh(query + key), dim=-1)  # [batch_size, Tq, Tv]
        mask = torch.tril(torch.ones_like(scores))
        scores = scores.masked_fill(mask == 0, -1e9)
        attention_scores = F.softmax(scores, dim=-1)  # [batch_size, Tq, Tv]
        weighted_sum = torch.matmul(attention_scores, value)
        return weighted_sum, attention_scores  # [batch_size, Tq, dim], [batch_size, Tq, Tv]







