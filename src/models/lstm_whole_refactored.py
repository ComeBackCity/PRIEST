import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from typing import List
from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef

class CustomCategoricalCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CustomCategoricalCrossEntropyLoss, self).__init__()
        self.loss1 = nn.CrossEntropyLoss()
        self.loss2 = nn.MSELoss()

    def forward(self, inputs, targets):
        predictions = inputs.argmax(dim=1)
        loss1 = self.loss1(inputs, targets)
        loss2 = self.loss2(predictions, targets.float())
        
        loss = 0.5 * loss1 + 0.5 * loss2

        return loss

class MultiAttention(nn.Module):
    def __init__(self, 
        seq_length, 
        input_size, 
        device
    ) -> None:
        super(MultiAttention, self).__init__()

        self.attention_weights = nn.Parameter(torch.ones(3))
        # self.attention_weights = nn.Parameter(torch.ones(3, seq_length, input_size))
        self.attention_scale = nn.Parameter(torch.ones(1))
        self.attention_linear_layers = [nn.Linear(in_features=input_size, out_features=input_size, device=device) for _ in range(3)]

        # for scaled dot product
        self.query_layer = nn.Linear(in_features=input_size, out_features=input_size, bias=False, device=device)
        self.key_layer = nn.Linear(in_features=input_size, out_features=input_size, device=device)
        self.value_layer = nn.Linear(in_features=input_size, out_features=input_size, bias=False, device=device)

        self.attention_weights.requires_grad = True
        self.attention_scale.requires_grad = True

        self.device = device

    def forward(self, x:Tensor) -> Tensor:
        attention_outputs = [
            self.attention(x)[0], \
            self.scaled_dot_product_attention(x, x, x, scale=self.attention_scale)[0], \
            self.additive_attention(x, x, x)[0] \
        ]

        # attention_weights = self.attention_weights / self.attention_weights.sum(dim=0)
        attention_weights = self.attention_weights / self.attention_weights.sum()
        attention_weights = [attention_weight.to(self.device) for attention_weight in attention_weights]
        attention_weights = [attn_lin(attention_weight) for attn_lin, attention_weight in zip(self.attention_linear_layers, attention_weights)]
        attention_weights = [F.relu(attention_weight) for attention_weight in attention_weights]
        weighted_attention_outputs = [attention_outputs[i] * attention_weights[i] for i in range(3)]
        attention_output = sum(weighted_attention_outputs)

        return attention_output

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
        # query_layer = nn.Linear(query.shape[-1], query.shape[-1], bias=False, device=self.device)
        # key_layer = nn.Linear(key.shape[-1], key.shape[-1], bias=False, device=self.device)
        # value_layer = nn.Linear(value.shape[-1], value.shape[-1], bias=False, device=self.device)
        
        # Apply linear layers to query, key, and value
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)
        
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


class AttentionEncoderLayer(nn.Module):
    def __init__(self,  
        seq_length, 
        input_size, 
        device, 
        ff_dropout: float = 0.3,
        ff_enabled: bool = False, 
        non_linearity: str = 'relu',
    ) -> None:
        super(AttentionEncoderLayer, self).__init__()

        self.attention_layer = MultiAttention(
            seq_length=seq_length,
            input_size=input_size,
            device=device
        )

        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)

        self.ff_enabled = ff_enabled
        modules = [
            nn.Linear(in_features=input_size, out_features=input_size // 2),
            nn.Dropout(ff_dropout),
            nn.GELU() if non_linearity == 'gelu' else nn.ReLU(),
            nn.Linear(in_features=input_size // 2, out_features=input_size),
            nn.Dropout(ff_dropout),
            nn.GELU() if non_linearity == 'gelu' else nn.ReLU()
        ]

        if ff_enabled:
            self.ff = nn.Sequential(*modules)

    def forward(self, x : Tensor) -> Tensor:

        x = self.norm1(x + self.attention_layer(x))
        if self.ff_enabled:
            x = self.norm2(x + self.ff(x))

        return x


class MultiChannelMultiAttention(nn.Module):
    def __init__(self,
        seq_length, 
        input_size, 
        hidden_size, 
        num_layers, 
        dropout,
        device
    ) -> None:
        super(MultiChannelMultiAttention, self).__init__()

        self.multi_attention_block = MultiAttention(
            seq_length=seq_length,
            input_size=input_size,
            device=device
        )

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.bn11 = torch.nn.BatchNorm1d(num_features=input_size)
        self.bn12 = torch.nn.BatchNorm1d(num_features=input_size)
        self.bn13 = torch.nn.BatchNorm1d(num_features=input_size)

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.lstm3 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.layernorm = nn.LayerNorm(hidden_size)

        # self.sum_weight = nn.Parameter(torch.ones(3, seq_length, hidden_size))
        self.sum_weight = nn.Parameter(torch.ones(3))
        self.sum_weight.requires_grad = True

    def forward(self, x : Tensor) -> Tensor:
        attention_output = self.multi_attention_block(x)

        line1 = x.transpose(1, 2) 
        line2 = torch.add(x, attention_output).transpose(1, 2)
        line3 = attention_output.transpose(1, 2)

        line1 = self.bn11(line1).transpose(1, 2)
        line2 = self.bn12(line2).transpose(1, 2)
        line3 = self.bn13(line3).transpose(1, 2)


        h1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        h2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        h3 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c3 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
    
        line1, _ = self.lstm1(line1, (h1, c1))     
        line2, _ = self.lstm2(line2, (h2, c2)) 
        line3, _ = self.lstm3(line3, (h3, c3))  

        # cat_weights = self.sum_weight / self.sum_weight.sum(dim=0)
        cat_weights = self.sum_weight / self.sum_weight.sum()
        cat = line1 * cat_weights[0] + line2 * cat_weights[1] + line3 * cat_weights[2]
        cat = self.layernorm(cat)
        
        return cat

class AttentionEncoder(nn.Module):
    def __init__(self, 
        seq_length, 
        input_size, 
        device, 
        num_layers, 
        ff_dropout: float = 0.3,
        ff_enabled: bool = False, 
        non_linearity: str = 'relu',
    ) -> None:
        super(AttentionEncoder, self).__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(
                AttentionEncoderLayer(
                    seq_length=seq_length, 
                    input_size=input_size, 
                    device=device,
                    ff_dropout = ff_dropout,
                    ff_enabled = ff_enabled, 
                    non_linearity = non_linearity,
                )
            )
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

class PositionalEncoder(nn.Module):
    """
    The authors of the original transformer paper describe very succinctly what 
    the positional encoding layer does and why it is needed:
    
    "Since our model contains no recurrence and no convolution, in order for the 
    model to make use of the order of the sequence, we must inject some 
    information about the relative or absolute position of the tokens in the 
    sequence." (Vaswani et al, 2017)
    Adapted from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self, 
        device,
        dropout: float=0.5, 
        max_seq_len: int=9, 
        embed_dim: int=1,
        ):

        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """

        super().__init__()

        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2,  dtype=torch.float32) * (-math.log(10000.0) / embed_dim))
        
        self.positional_encoding = torch.zeros(max_seq_len, embed_dim, device=device)
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)
        
        
    def forward(self, x: Tensor, debug:bool=False) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """
        x = x + self.positional_encoding
        return self.dropout(x)

class LSTMSeriesClassifier(nn.Module):
    def __init__(self, 
        seq_length, 
        input_size, 
        hidden_size, 
        num_layers, 
        num_classes, 
        device
    ):
        super(LSTMSeriesClassifier, self).__init__()

        self.seq_length = seq_length
        self.hidden_size_1 = hidden_size
        self.hidden_size_2 = hidden_size // 4
        self.num_layers = num_layers

        self.positional_encoder_1 = PositionalEncoder(
            device=device,
            dropout=0.2,
            max_seq_len=seq_length,
            embed_dim=input_size
        )

        self.positional_encoder_2 = PositionalEncoder(
            device=device,
            dropout=0.2,
            max_seq_len=seq_length,
            embed_dim=hidden_size
        )

        # self.positional_encoder_3 = PositionalEncoder(
        #     device=device,
        #     dropout=0.2,
        #     max_seq_len=seq_length,
        #     embed_dim=hidden_size // 2
        # )

        # self.positional_encoder_4 = PositionalEncoder(
        #     device=device,
        #     dropout=0.2,
        #     max_seq_len=seq_length,
        #     embed_dim=hidden_size // 4
        # )

        self.mcma1 = MultiChannelMultiAttention(
            seq_length=seq_length,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.4,
            device=device
        )

        # self.mcma2 = MultiChannelMultiAttention(
        #     seq_length=seq_length,
        #     input_size=hidden_size,
        #     hidden_size=hidden_size // 2,
        #     num_layers=num_layers,
        #     dropout=0.4,
        #     device=device
        # )

        # self.mcma3 = MultiChannelMultiAttention(
        #     seq_length=seq_length,
        #     input_size=hidden_size // 2,
        #     hidden_size=hidden_size // 4,
        #     num_layers=num_layers,
        #     dropout=0.4,
        #     device=device
        # )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size // 4, 
            nhead=8,
            dim_feedforward=512 ,
            # dropout=0.2,
            batch_first=True,
            activation='relu',
            layer_norm_eps=1e-4,
            norm_first=False
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=4, 
            norm=None
        )

        self.device = device

        self.length = seq_length * hidden_size 

        self.fc1 = nn.Linear(in_features=self.length , out_features=self.length // 4)
        self.do1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=self.length // 4, out_features=self.length // 16)
        self.do2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(in_features=self.length // 16, out_features=self.length // 32)
        self.do3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(in_features=self.length // 32, out_features=num_classes)

    def forward(self, src, debug=False):

        src = self.positional_encoder_1(src)
        src = self.mcma1(src)

        src =self.positional_encoder_2(src)
        # src = self.mcma2(src)

        # src = self.positional_encoder_3(src)
        # src = self.mcma3(src)

        # src = self.positional_encoder_4(src)
        # src = self.mcma3(src)

        mask = torch.tril(torch.ones((self.seq_length, self.seq_length), device=self.device))
        src = self.encoder(
            src,
            mask
        )
        
        out = torch.reshape(src, (-1, self.seq_length * self.hidden_size_2))
        out = self.fc1(out)
        out = self.do1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.do2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = self.do3(out)
        out = F.relu(out)
        output = self.fc4(out)

        return output, 0

