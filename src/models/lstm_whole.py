import torch
from torch import nn
import torch.nn.functional as F
import math
# import gensim

class LSTMSeriesClassifier(nn.Module):
    def __init__(self, seq_length, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTMSeriesClassifier, self).__init__()

        self.seq_length = seq_length
        self.hidden_size_1 = hidden_size
        self.hidden_size_2 = hidden_size // 4
        self.num_layers = num_layers

        self.bn11 = torch.nn.BatchNorm1d(num_features=input_size)
        self.bn12 = torch.nn.BatchNorm1d(num_features=input_size)
        self.bn13 = torch.nn.BatchNorm1d(num_features=input_size)

        self.lstm11 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.lstm12 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.lstm13 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)

        self.bn21 = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.bn22 = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.bn23 = torch.nn.BatchNorm1d(num_features=hidden_size)

        self.lstm21 = nn.LSTM(hidden_size, hidden_size // 4, num_layers, batch_first=True, dropout=0.5)
        self.lstm22 = nn.LSTM(hidden_size, hidden_size // 4, num_layers, batch_first=True, dropout=0.5)
        self.lstm23 = nn.LSTM(hidden_size, hidden_size // 4, num_layers, batch_first=True, dropout=0.5)

        self.bn31 = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.bn32 = torch.nn.BatchNorm1d(num_features=hidden_size // 4)
        self.bn33 = torch.nn.BatchNorm1d(num_features=hidden_size // 4)
        self.bn34 = torch.nn.BatchNorm1d(num_features=hidden_size // 4)
        self.bn35 = torch.nn.BatchNorm1d(num_features=hidden_size // 4)
        self.bn36 = torch.nn.BatchNorm1d(num_features=hidden_size // 4)
        self.bn37 = torch.nn.BatchNorm1d(num_features=hidden_size // 4)
        self.bn38 = torch.nn.BatchNorm1d(num_features=hidden_size // 4)

        # self.fc = nn.Linear(hidden_size, num_classes)
        # self.dropout_final = nn.Dropout(0.3)

        self.attention_weights_1 = nn.Parameter(torch.ones(3, seq_length, input_size))
        self.cat_weights_1 = nn.Parameter(torch.ones(3,  seq_length, hidden_size))
        self.attn_scale_1 = nn.Parameter(torch.ones(1))
        self.attn_lin_1 = [nn.Linear(in_features=input_size, out_features=input_size, device=device) for _ in range(3)]

        self.attention_weights_2 = nn.Parameter(torch.ones(3, seq_length, hidden_size))
        self.cat_weights_2 = nn.Parameter(torch.ones(3,  seq_length, hidden_size // 4))
        self.attn_scale_2 = nn.Parameter(torch.ones(1))
        self.attn_lin_2 = [nn.Linear(in_features=hidden_size, out_features=hidden_size, device=device) for _ in range(3)]
        
        self.attention_weights_1.requires_grad = True
        self.cat_weights_1.requires_grad = True 
        self.attn_scale_1.requires_grad = True
        self.attention_weights_2.requires_grad = True
        self.cat_weights_2.requires_grad = True 
        self.attn_scale_2.requires_grad = True


        # attention weights
        # self.lstm3 = nn.LSTM(hidden_size // 4, hidden_size // 4, 1, batch_first=True, dropout=0.5)
        # self.lstm4 = nn.LSTM(hidden_size // 4, hidden_size // 4, 1, batch_first=True, dropout=0.5)
        # self.lstm5 = nn.LSTM(hidden_size // 4, hidden_size // 4, 1, batch_first=True, dropout=0.5)
        # self.lstm4 = nn.LSTM(hidden_size // 4, hidden_size // 4, 1, batch_first=True, dropout=0.5)
        self.attention_weights_3 = nn.Parameter(torch.ones(3, seq_length, hidden_size // 4))
        self.attn_scale_3 = nn.Parameter(torch.ones(1))
        self.attn_lin_3 = [nn.Linear(in_features=hidden_size // 4, out_features=hidden_size // 4, device=device) for _ in range(3)]
        # self.attn_enc_3 = nn.Linear(in_features=hidden_size // 4, out_features=hidden_size // 8, device=device)
        # self.attn_do_31 = nn.Dropout(0.3)
        # self.attn_dec_3 = nn.Linear(in_features=hidden_size // 8, out_features=hidden_size // 4, device=device)
        # self.attn_do_32 = nn.Dropout(0.3)

        self.attention_weights_4 = nn.Parameter(torch.ones(3, seq_length, hidden_size // 4))
        self.attn_scale_4 = nn.Parameter(torch.ones(1))
        self.attn_lin_4 = [nn.Linear(in_features=hidden_size // 4, out_features=hidden_size // 4, device=device) for _ in range(3)]
        # self.attn_enc_4 = nn.Linear(in_features=hidden_size // 4, out_features=hidden_size // 8, device=device)
        # self.attn_do_41 = nn.Dropout(0.3)
        # self.attn_dec_4 = nn.Linear(in_features=hidden_size // 8, out_features=hidden_size // 4, device=device)
        # self.attn_do_42 = nn.Dropout(0.3)

        self.attention_weights_5 = nn.Parameter(torch.ones(3, seq_length, hidden_size // 4))
        self.attn_scale_5 = nn.Parameter(torch.ones(1))
        self.attn_lin_5 = [nn.Linear(in_features=hidden_size // 4, out_features=hidden_size // 4, device=device) for _ in range(3)]
        # self.attn_enc_5 = nn.Linear(in_features=hidden_size // 4, out_features=hidden_size // 8, device=device)
        # self.attn_do_51 = nn.Dropout(0.3)
        # self.attn_dec_5 = nn.Linear(in_features=hidden_size // 8, out_features=hidden_size // 4, device=device)
        # self.attn_do_52 = nn.Dropout(0.3)

        self.attention_weights_6 = nn.Parameter(torch.ones(3, seq_length, hidden_size // 4))
        self.attn_scale_6 = nn.Parameter(torch.ones(1))
        self.attn_lin_6 = [nn.Linear(in_features=hidden_size // 4, out_features=hidden_size // 4, device=device) for _ in range(3)]
        # self.attn_enc_6 = nn.Linear(in_features=hidden_size // 4, out_features=hidden_size // 8, device=device)
        # self.attn_do_61 = nn.Dropout(0.3)
        # self.attn_dec_6 = nn.Linear(in_features=hidden_size // 8, out_features=hidden_size // 4, device=device)
        # self.attn_do_62 = nn.Dropout(0.3)

        self.attention_weights_7 = nn.Parameter(torch.ones(3, seq_length, hidden_size // 4))
        self.attn_scale_7 = nn.Parameter(torch.ones(1))
        self.attn_lin_7 = [nn.Linear(in_features=hidden_size // 4, out_features=hidden_size // 4, device=device) for _ in range(3)]
        # self.attn_enc_7 = nn.Linear(in_features=hidden_size // 4, out_features=hidden_size // 8, device=device)
        # self.attn_do_71 = nn.Dropout(0.3)
        # self.attn_dec_7 = nn.Linear(in_features=hidden_size // 8, out_features=hidden_size // 4, device=device)
        # self.attn_do_72 = nn.Dropout(0.3)

        self.attention_weights_8 = nn.Parameter(torch.ones(3, seq_length, hidden_size // 4))
        self.attn_scale_8 = nn.Parameter(torch.ones(1))
        self.attn_lin_8 = [nn.Linear(in_features=hidden_size // 4, out_features=hidden_size // 4, device=device) for _ in range(3)]
        # self.attn_enc_8 = nn.Linear(in_features=hidden_size // 4, out_features=hidden_size // 8, device=device)
        # self.attn_do_81 = nn.Dropout(0.3)
        # self.attn_dec_8 = nn.Linear(in_features=hidden_size // 8, out_features=hidden_size // 4, device=device)
        # self.attn_do_82 = nn.Dropout(0.3)

        self.attn_scale_3.requires_grad = True
        self.attn_scale_4.requires_grad = True
        self.attn_scale_5.requires_grad = True
        self.attn_scale_6.requires_grad = True
        self.attn_scale_7.requires_grad = True
        self.attn_scale_8.requires_grad = True

        self.device = device

        self.fc1 = nn.Linear(in_features=seq_length * hidden_size // 4 , out_features=hidden_size // 4)
        self.do = nn.Dropout(0.4)
        self.fc2 = nn.Linear(in_features=hidden_size // 4, out_features=num_classes)

        # self.fc = nn.Linear(hidden_size // 4, num_classes)
        # self.dropout_final = nn.Dropout(0.3)

    def forward(self, src, debug=False):
        attention_outputs = [
            self.attention(src)[0], \
            self.scaled_dot_product_attention(src, src, src, scale=self.attn_scale_1)[0], \
            self.additive_attention(src, src, src)[0] \
        ]

        attention_weights = self.attention_weights_1 / self.attention_weights_1.sum(dim=0)
        attention_weights = [attention_weight.to(self.device) for attention_weight in attention_weights]
        attention_weights = [attn_lin(attention_weight) for attn_lin, attention_weight in zip(self.attn_lin_1, attention_weights)]
        attention_weights = [F.relu(attention_weight) for attention_weight in attention_weights]
        weighted_attention_outputs = [attention_outputs[i] * attention_weights[i] for i in range(3)]
        attention_output = sum(weighted_attention_outputs)

        line1 = src.transpose(1, 2)
        line2 = torch.add(src, attention_output).transpose(1, 2)
        line3 = attention_output.transpose(1, 2)

        line1 = self.bn11(line1).transpose(1, 2)
        line2 = self.bn12(line2).transpose(1, 2)
        line3 = self.bn13(line3).transpose(1, 2)

        h1 = torch.zeros(self.num_layers, src.size(0), self.hidden_size_1).to(self.device)
        c1 = torch.zeros(self.num_layers, src.size(0), self.hidden_size_1).to(self.device)

        h2 = torch.zeros(self.num_layers, src.size(0), self.hidden_size_1).to(self.device)
        c2 = torch.zeros(self.num_layers, src.size(0), self.hidden_size_1).to(self.device)

        h3 = torch.zeros(self.num_layers, src.size(0), self.hidden_size_1).to(self.device)
        c3 = torch.zeros(self.num_layers, src.size(0), self.hidden_size_1).to(self.device)
    
        line1, _ = self.lstm11(line1, (h1, c1))     
        line2, _ = self.lstm12(line2, (h2, c2)) 
        line3, _ = self.lstm13(line3, (h3, c3))  

        cat_weights = self.cat_weights_1 / self.cat_weights_1.sum(dim=0)
        cat = line1 * cat_weights[0] + line2 * cat_weights[1] + line3 * cat_weights[2]
        cat = cat.transpose(1, 2)
        cat = self.bn31(cat).transpose(1, 2)

        # print(cat.shape)

        # Round 2

        attention_outputs = [
            self.attention(cat)[0], \
            self.scaled_dot_product_attention(cat, cat, cat, scale=self.attn_scale_2)[0], \
            self.additive_attention(cat, cat, cat)[0] \
        ]

        attention_weights = self.attention_weights_2 / self.attention_weights_2.sum(dim=0)
        attention_weights = [attn_lin(attention_weight) for attn_lin, attention_weight in zip(self.attn_lin_2, attention_weights)]
        attention_weights = [F.relu(attention_weight) for attention_weight in attention_weights]
        weighted_attention_outputs = [attention_outputs[i] * attention_weights[i] for i in range(3)]
        attention_output = sum(weighted_attention_outputs)
        # print(attention_output.shape)

        line1 = cat.transpose(1, 2)
        line2 = torch.add(cat, attention_output).transpose(1, 2)
        line3 = attention_output.transpose(1, 2)

        line1 = self.bn21(line1).transpose(1, 2)
        line2 = self.bn22(line2).transpose(1, 2)
        line3 = self.bn23(line3).transpose(1, 2)

        h1 = torch.zeros(self.num_layers, src.size(0), self.hidden_size_2).to(self.device)
        c1 = torch.zeros(self.num_layers, src.size(0), self.hidden_size_2).to(self.device)

        h2 = torch.zeros(self.num_layers, src.size(0), self.hidden_size_2).to(self.device)
        c2 = torch.zeros(self.num_layers, src.size(0), self.hidden_size_2).to(self.device)

        h3 = torch.zeros(self.num_layers, src.size(0), self.hidden_size_2).to(self.device)
        c3 = torch.zeros(self.num_layers, src.size(0), self.hidden_size_2).to(self.device)
    
        line1, _ = self.lstm21(line1, (h1, c1))     
        line2, _ = self.lstm22(line2, (h2, c2)) 
        line3, _ = self.lstm23(line3, (h3, c3))  

        cat_weights = self.cat_weights_2 / self.cat_weights_2.sum(dim=0)
        cat = line1 * cat_weights[0] + line2 * cat_weights[1] + line3 * cat_weights[2]
        # cat = line1 * cat_weights[0] + line3 * cat_weights[1]
        cat = cat.transpose(1, 2)
        cat = self.bn32(cat).transpose(1, 2)

        # cat = self.dropout_final(torch.cat(pooled, dim = 1))
        # src = self.dropout_final(cat[:, -1, :])
        # output = self.fc(src)

        # apply multiple attention layers

        src = cat

        attention_outputs = [
            self.attention(cat)[0], \
            self.scaled_dot_product_attention(cat, cat, cat, scale=self.attn_scale_3)[0], \
            self.additive_attention(cat, cat, cat)[0] \
        ]

        attention_weights = self.attention_weights_3 / self.attention_weights_3.sum(dim=0)
        attention_weights = [attn_lin(attention_weight) for attn_lin, attention_weight in zip(self.attn_lin_3, attention_weights)]
        attention_weights = [F.relu(attention_weight) for attention_weight in attention_weights]
        weighted_attention_outputs = [attention_outputs[i] * attention_weights[i] for i in range(3)]
        attention_output = sum(weighted_attention_outputs)

        cat = attention_output
        cat = cat + src
        cat = cat.transpose(1, 2)
        cat = self.bn33(cat).transpose(1, 2)
        src = cat

        attention_outputs = [
            self.attention(cat)[0], \
            self.scaled_dot_product_attention(cat, cat, cat, scale=self.attn_scale_4)[0], \
            self.additive_attention(cat, cat, cat)[0] \
        ]

        attention_weights = self.attention_weights_4 / self.attention_weights_4.sum(dim=0)
        attention_weights = [attn_lin(attention_weight) for attn_lin, attention_weight in zip(self.attn_lin_4, attention_weights)]
        attention_weights = [F.relu(attention_weight) for attention_weight in attention_weights]
        weighted_attention_outputs = [attention_outputs[i] * attention_weights[i] for i in range(3)]
        attention_output = sum(weighted_attention_outputs)

        cat = attention_output
        cat = cat + src
        cat = cat.transpose(1, 2)
        cat = self.bn34(cat).transpose(1, 2)
        src = cat

        # cat = cat + src
        # src = cat

        attention_outputs = [
            self.attention(cat)[0], \
            self.scaled_dot_product_attention(cat, cat, cat, scale=self.attn_scale_5)[0], \
            self.additive_attention(cat, cat, cat)[0] \
        ]

        attention_weights = self.attention_weights_5 / self.attention_weights_5.sum(dim=0)
        attention_weights = [attn_lin(attention_weight) for attn_lin, attention_weight in zip(self.attn_lin_5, attention_weights)]
        attention_weights = [F.relu(attention_weight) for attention_weight in attention_weights]
        weighted_attention_outputs = [attention_outputs[i] * attention_weights[i] for i in range(3)]
        attention_output = sum(weighted_attention_outputs)

        cat = attention_output
        cat = cat + src
        cat = cat.transpose(1, 2)
        cat = self.bn35(cat).transpose(1, 2)
        src = cat

        attention_outputs = [
            self.attention(cat)[0], \
            self.scaled_dot_product_attention(cat, cat, cat, scale=self.attn_scale_6)[0], \
            self.additive_attention(cat, cat, cat)[0] \
        ]

        attention_weights = self.attention_weights_6 / self.attention_weights_6.sum(dim=0)
        attention_weights = [attn_lin(attention_weight) for attn_lin, attention_weight in zip(self.attn_lin_6, attention_weights)]
        attention_weights = [F.relu(attention_weight) for attention_weight in attention_weights]
        weighted_attention_outputs = [attention_outputs[i] * attention_weights[i] for i in range(3)]
        attention_output = sum(weighted_attention_outputs)

        cat = attention_output
        cat = cat + src
        cat = cat.transpose(1, 2)
        cat = self.bn36(cat).transpose(1, 2)
        src = cat

        # cat = cat + src
        # src = cat

        attention_outputs = [
            self.attention(cat)[0], \
            self.scaled_dot_product_attention(cat, cat, cat, scale=self.attn_scale_6)[0], \
            self.additive_attention(cat, cat, cat)[0] \
        ]

        attention_weights = self.attention_weights_7 / self.attention_weights_7.sum(dim=0)
        attention_weights = [attn_lin(attention_weight) for attn_lin, attention_weight in zip(self.attn_lin_7, attention_weights)]
        attention_weights = [F.relu(attention_weight) for attention_weight in attention_weights]
        weighted_attention_outputs = [attention_outputs[i] * attention_weights[i] for i in range(3)]
        attention_output = sum(weighted_attention_outputs)

        cat = attention_output
        cat = cat + src
        cat = cat.transpose(1, 2)
        cat = self.bn37(cat).transpose(1, 2)
        src = cat

        attention_outputs = [
            self.attention(cat)[0], \
            self.scaled_dot_product_attention(cat, cat, cat, scale=self.attn_scale_6)[0], \
            self.additive_attention(cat, cat, cat)[0] \
        ]

        attention_weights = self.attention_weights_8 / self.attention_weights_8.sum(dim=0)
        attention_weights = [attn_lin(attention_weight) for attn_lin, attention_weight in zip(self.attn_lin_8, attention_weights)]
        attention_weights = [F.relu(attention_weight) for attention_weight in attention_weights]
        weighted_attention_outputs = [attention_outputs[i] * attention_weights[i] for i in range(3)]
        attention_output = sum(weighted_attention_outputs)

        cat = attention_output
        cat = cat + src
        cat = cat.transpose(1, 2)
        cat = self.bn38(cat).transpose(1, 2)
        src = cat

        # cat = src + cat

        out = torch.reshape(cat, (-1, self.seq_length * self.hidden_size_2))
        out = self.fc1(out)
        out = self.do(out)
        out = F.relu(out)
        output = self.fc2(out)

        # src = self.dropout_final(cat[:, -1, :])
        # output = self.fc(src)

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







