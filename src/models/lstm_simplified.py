import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from typing import List
# from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef

class MultiAttention(nn.Module):
    def __init__(self, 
        seq_length, 
        input_size, 
        device
    ) -> None:
        super(MultiAttention, self).__init__()

        self.attention_weights = nn.Parameter(torch.ones(3))
        self.attention_scale = nn.Parameter(torch.ones(1))
        # self.attention_linear_layers = [nn.Linear(in_features=input_size, out_features=input_size, device=device) for _ in range(3)]
        # self.attention_linear_layer = nn.Linear(in_features=input_size, out_features=input_size, device=device)

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

        attention_weights = self.attention_weights / self.attention_weights.sum()
        # attention_weights = [attention_weight.to(self.device) for attention_weight in attention_weights]
        # attention_weights = [attn_lin(attention_weight) for attn_lin, attention_weight in zip(self.attention_linear_layers, attention_weights)]
        # attention_weights = [F.relu(attention_weight) for attention_weight in attention_weights]
        weighted_attention_outputs = [attention_outputs[i] * attention_weights[i] for i in range(3)]
        attention_output = sum(weighted_attention_outputs)
        # attention_output = self.attention_linear_layer(attention_output)
        # attention_output = F.relu(attention_output)

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
            nn.Linear(in_features=input_size, out_features=512),
            nn.Dropout(ff_dropout),
            nn.GELU() if non_linearity == 'gelu' else nn.ReLU(),
            nn.Linear(in_features=512, out_features=input_size),
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

class MultiChannelMultiAttentionCNN(nn.Module):
    def __init__(self,
        seq_length, 
        input_size, 
        hidden_size, 
        num_layers, 
        dropout,
        device
    ) -> None:
        super(MultiChannelMultiAttentionCNN, self).__init__()

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

        self.inc_blk1 = InceptionBlock(seq_length, dropout, input_size)
        self.inc_blk2 = InceptionBlock(seq_length, dropout, input_size)
        self.inc_blk3 = InceptionBlock(seq_length, dropout, input_size)

        self.layernorm = nn.LayerNorm(3 * input_size - 8)

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
 
        line1 = self.inc_blk1(line1)
        line2 = self.inc_blk2(line2)
        line3 = self.inc_blk3(line3)

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
        
        # print(x.shape)
        # print(self.positional_encoding.shape)
        x = x + self.positional_encoding
        x = F.layer_norm(x, normalized_shape=x.shape)
        x = self.dropout(x)
        return x
    

class InceptionBlock(nn.Module):
    def __init__(self, seq_length, dropout, embed_dim) -> None:
        super(InceptionBlock, self).__init__()

        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

        self.conv11 = nn.Conv1d(
            in_channels=self.seq_length, 
            out_channels=self.seq_length, 
            kernel_size=1
        )
        self.bn11 = nn.BatchNorm1d(
            num_features=self.embed_dim 
        )
        

        self.conv21 = nn.Conv1d(
            in_channels=self.seq_length, 
            out_channels=self.seq_length, 
            kernel_size=1
        )
        self.bn21 = nn.BatchNorm1d(
            num_features=self.embed_dim 
        )

        self.conv22 = nn.Conv1d(
            in_channels=self.seq_length, 
            out_channels=self.seq_length, 
            kernel_size=3
        )

        self.bn22 = nn.BatchNorm1d(
            num_features=self.embed_dim - 3 +1
        )

        self.conv31 = nn.Conv1d(
            in_channels=self.seq_length, 
            out_channels=self.seq_length, 
            kernel_size=1
        )

        self.bn31 = nn.BatchNorm1d(
            num_features=self.embed_dim
        )

        self.conv32 = nn.Conv1d(
            in_channels=self.seq_length, 
            out_channels=self.seq_length, 
            kernel_size=3
        )

        self.bn32 = nn.BatchNorm1d(
            num_features=self.embed_dim - 3 + 1
        )

        self.conv33 = nn.Conv1d(
            in_channels=self.seq_length, 
            out_channels=self.seq_length, 
            kernel_size=5
        )

        self.bn33 = nn.BatchNorm1d(
            num_features=self.embed_dim - 8 + 2
        )

    def forward(self, src):
        lin1 = F.relu(self.conv11(src))
        lin1 = self.dropout(lin1)
        lin1 = torch.transpose(lin1, 1, 2)
        lin1 = self.bn11(lin1).transpose(1, 2)

        lin2 = F.relu(self.conv21(src))
        lin2 = self.dropout(lin2)
        lin2 = torch.transpose(lin2, 1, 2)
        lin2 = self.bn21(lin2).transpose(1, 2)
        lin2 = F.relu(self.conv22(lin2))
        lin2 = self.dropout(lin2)
        lin2 = torch.transpose(lin2, 1, 2)
        lin2 = self.bn22(lin2).transpose(1, 2)

        lin3 = F.relu(self.conv31(src))
        lin3 = self.dropout(lin3)
        lin3 = torch.transpose(lin3, 1, 2)
        lin3 = self.bn31(lin3).transpose(1, 2)
        lin3 = F.relu(self.conv32(lin3))
        lin3 = self.dropout(lin3)
        lin3 = torch.transpose(lin3, 1, 2)
        lin3 = self.bn32(lin3).transpose(1, 2)
        lin3 = F.relu(self.conv33(lin3))
        lin3 = self.dropout(lin3)
        lin3 = torch.transpose(lin3, 1, 2)
        lin3 = self.bn33(lin3).transpose(1, 2)

        output = torch.cat([lin1, lin2, lin3], dim=2)

        return output


class InceptionBlockAttn(nn.Module):
    def __init__(self, seq_length, dropout, pos_dr, embed_dim, device) -> None:
        super(InceptionBlockAttn, self).__init__()

        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

        self.conv11 = nn.Conv1d(
            in_channels=self.seq_length, 
            out_channels=self.seq_length, 
            kernel_size=1,
            # padding=1
        )
        self.bn11 = nn.BatchNorm1d(
            num_features=self.embed_dim 
        )
        

        self.conv21 = nn.Conv1d(
            in_channels=self.seq_length, 
            out_channels=self.seq_length, 
            kernel_size=1,
            # padding=1
        )
        self.bn21 = nn.BatchNorm1d(
            num_features=self.embed_dim 
        )

        self.conv22 = nn.Conv1d(
            in_channels=self.seq_length, 
            out_channels=self.seq_length, 
            kernel_size=3,
            padding=1
        )

        self.bn22 = nn.BatchNorm1d(
            num_features=self.embed_dim
        )

        self.conv31 = nn.Conv1d(
            in_channels=self.seq_length, 
            out_channels=self.seq_length, 
            kernel_size=1,
            # padding=1
        )

        self.bn31 = nn.BatchNorm1d(
            num_features=self.embed_dim
        )

        self.conv32 = nn.Conv1d(
            in_channels=self.seq_length, 
            out_channels=self.seq_length, 
            kernel_size=3,
            padding=1
        )

        self.bn32 = nn.BatchNorm1d(
            num_features=self.embed_dim
        )

        self.conv33 = nn.Conv1d(
            in_channels=self.seq_length, 
            out_channels=self.seq_length, 
            kernel_size=5,
            padding=2
        )

        self.bn33 = nn.BatchNorm1d(
            num_features=self.embed_dim
        )

        self.scale1 = nn.Parameter(torch.ones(1))
        self.scale2 = nn.Parameter(torch.ones(1))
        self.scale3 = nn.Parameter(torch.ones(1))

        self.scale1.requires_grad = True
        self.scale2.requires_grad = True
        self.scale3.requires_grad = True

        # self.attn1 = MultiAttention(seq_length=seq_length, input_size=embed_dim, device=device)
        # self.attn2 = MultiAttention(seq_length=seq_length, input_size=embed_dim -3 + 1, device=device)
        # self.attn3 = MultiAttention(seq_length=seq_length, input_size=embed_dim -8 + 2, device=device)

        self.transformed_input_size = embed_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )
        
        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        encoder_layer3 = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=2,
        )

        self.encoder2 = nn.TransformerEncoder(
            encoder_layer=encoder_layer2,
            num_layers=2,
        )

        self.encoder3 = nn.TransformerEncoder(
            encoder_layer=encoder_layer3,
            num_layers=2,
        )

        self.pe1 = PositionalEncoder(device=device, dropout=pos_dr, max_seq_len=seq_length, embed_dim=embed_dim)
        self.pe2 = PositionalEncoder(device=device, dropout=pos_dr, max_seq_len=seq_length, embed_dim=embed_dim)
        self.pe3 = PositionalEncoder(device=device, dropout=pos_dr, max_seq_len=seq_length, embed_dim=embed_dim)

    def forward(self, src):
        lin1 = F.relu(self.conv11(src))
        lin1 = self.dropout(lin1)
        lin1 = torch.transpose(lin1, 1, 2)
        lin1 = self.bn11(lin1).transpose(1, 2)

        lin2 = F.relu(self.conv21(src))
        lin2 = self.dropout(lin2)
        lin2 = torch.transpose(lin2, 1, 2)
        lin2 = self.bn21(lin2).transpose(1, 2)
        lin2 = F.relu(self.conv22(lin2))
        lin2 = self.dropout(lin2)
        lin2 = torch.transpose(lin2, 1, 2)
        lin2 = self.bn22(lin2).transpose(1, 2)

        lin3 = F.relu(self.conv31(src))
        lin3 = self.dropout(lin3)
        lin3 = torch.transpose(lin3, 1, 2)
        lin3 = self.bn31(lin3).transpose(1, 2)
        lin3 = F.relu(self.conv32(lin3))
        lin3 = self.dropout(lin3)
        lin3 = torch.transpose(lin3, 1, 2)
        lin3 = self.bn32(lin3).transpose(1, 2)
        lin3 = F.relu(self.conv33(lin3))
        lin3 = self.dropout(lin3)
        lin3 = torch.transpose(lin3, 1, 2)
        lin3 = self.bn33(lin3).transpose(1, 2)

        lin1 = self.pe1(lin1)
        lin1 = self.encoder(lin1) * self.scale1 + lin1 * (1-self.scale1)
        lin2 = self.pe2(lin2)
        lin2 = self.encoder2(lin2) * self.scale2 + lin2 * (1-self.scale2)
        lin3 = self.pe3(lin3)
        lin3 = self.encoder3(lin3) * self.scale3 + lin3 * (1-self.scale3)

        output = torch.cat([lin1, lin2, lin3], dim=2)

        return output


class LSTMSeriesClassifier(nn.Module):
    def __init__(self, 
        seq_length, 
        input_size, 
        hidden_size, 
        num_layers, 
        num_classes, 
        dropout,
        positional_dropout,
        device
    ):
        super(LSTMSeriesClassifier, self).__init__()

        self.seq_length = seq_length
        self.hidden_size_1 = hidden_size
        self.hidden_size_2 = hidden_size // 4
        self.num_layers = num_layers

        self.positional_encoder_1 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=input_size
        )

        self.positional_encoder_2 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=hidden_size
        )

        self.positional_encoder_3 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=hidden_size
        )

        self.mcma1 = MultiChannelMultiAttentionCNN(
            seq_length=seq_length,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=device
        )

        self.lin = nn.Linear(
            in_features=3 * input_size - 8,
            out_features=hidden_size
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=16,
            # dim_feedforward=2,
            batch_first=True,
            activation='relu',
            # layer_norm_eps=1e-4,
            # norm_first=False
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=2, 
            norm=None
        )

        self.num_filters = 4
        self.kernel_size = 8

        # self.conv1d = nn.Conv1d(
        #     in_channels=self.seq_length, 
        #     out_channels=self.seq_length, 
        #     kernel_size=self.kernel_size
        # )

        self.inc_blk = InceptionBlock(
            seq_length=self.seq_length,
            dropout=dropout,
            embed_dim=self.hidden_size_1
        )

        self.fc1 = nn.Linear(
            in_features=self.seq_length * (3 * self.hidden_size_1 - 8),
            out_features=1024
        )

        self.fc2 = nn.Linear(
            in_features=1024,
            out_features=512
        )

        self.fc3 = nn.Linear(
            in_features=512,
            out_features=num_classes
        )

        self.device = device
        self.length = seq_length * hidden_size 
        self.do = nn.Dropout(dropout)
        self.dropout_final = nn.Dropout(dropout)

    def forward(self, src, debug=False):

        src = self.positional_encoder_1(src)
        src = self.mcma1(src)
        src = self.lin(src)

        src =self.positional_encoder_2(src)
        src = self.do(src)

        mask = torch.tril(torch.ones((self.seq_length, self.seq_length), device=self.device))
        src = self.encoder(
            src,
            mask
        )

        src =self.positional_encoder_3(src)
        x = src
        x = self.inc_blk(x)
        # x = F.relu(x)
        x = self.dropout_final(x)
        x = x.view(x.shape[0], -1)

        # cat = self.dropout_final(torch.cat(x, dim = 1))
        x = self.fc1(x)
        x = self.dropout_final(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout_final(x)
        x = F.relu(x)
        output = self.fc3(x)

        return output, 0

class LSTMSeriesClassifier2(nn.Module):
    def __init__(self, 
        seq_length, 
        input_size, 
        hidden_size, 
        num_layers, 
        num_classes, 
        dropout,
        positional_dropout,
        device
    ):
        super(LSTMSeriesClassifier2, self).__init__()

        self.conv11 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.conv21 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.conv22 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.transformed_input_size = input_size 

        self.pe = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.pe2 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.pe3 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.bn1 = nn.BatchNorm1d(
            num_features=input_size
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=2,
        )

        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder2 = nn.TransformerEncoder(
            encoder_layer=encoder_layer2,
            num_layers=2,
        )


        encoder_layer3 = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder3 = nn.TransformerEncoder(
            encoder_layer=encoder_layer3,
            num_layers=2,
        )
        
        self.fc3 = nn.Linear(
            in_features=self.transformed_input_size,
            out_features=num_classes
        )

        self.dropout = nn.Dropout(dropout)

        # self.weights = nn.Parameter(torch.ones(3))
        self.inp_weight = nn.Parameter(torch.ones(1))
        self.conv1_weight = nn.Parameter(torch.ones(1))
        # self.conv2_weight = nn.Parameter(torch.ones(1))
        self.final_weight = nn.Parameter(torch.ones(2))
        # self.final_weight = self.final_weight / self.final_weight.sum()
        # self.weights.requires_grad = True
        self.inp_weight.requires_grad = True
        self.conv1_weight.requires_grad = True
        # self.conv1_weight.requires_grad = True
        self.final_weight.requires_grad = True
        # # self.dropout2 = nn.Dropout(dropout / 2)


    def forward(self, src, debug=False):
        
        inp = src
        inp_pos = self.pe(inp)
        # inp_pos = self.dropout(inp_pos)
        inp = self.encoder(inp_pos) * self.inp_weight + inp * (1 - self.inp_weight)

        conved1 = self.conv11(src)
        conved1 = F.relu(conved1)
        conved1 = self.dropout(conved1)
        conved1 = torch.transpose(conved1, 1, 2)
        conved1 = self.bn1(conved1).transpose(1, 2)
        conved1_pos = self.pe2(conved1)
        # conved1_pos = self.dropout(conved1_pos)
        conved1 = self.encoder2(conved1_pos) * self.conv1_weight + conved1 * (1 - self.conv1_weight)

        # conved2 = self.conv21(src)
        # conved2 = self.conv22(src)
        # conved2_pos = self.pe2(conved2)
        # conved2_pos = self.dropout(conved2_pos)
        # conved2 = self.encoder2(conved2_pos) * self.conv2_weight + conved2 * (1 - self.conv2_weight)


        weights = self.final_weight / self.final_weight.sum()
        src = inp * weights[0] + \
              conved1 * weights[1] 
            #   conved2 * weights[2]
        # weights = self.weights / self.weights.sum()
        # src = inp * weights[0] + conved * weights[1] + src + weights[2]
        # src = inp * (1-self.final_weight) + conved * self.final_weight
        # src = src.view(src.shape[0], -1)
        # src = F.relu(self.fc2(src))
        # src = self.dropout(src)
        # output = self.fc3(src)

        # src = self.inc_blk(src)
        # src = self.dropout(src)
        # src = self.conv11(src)
        # src = self.dropout(src)
        # src = self.inc_blk_2(src)
        # src = self.dropout(src)
        # src = self.conv12(src)


        output = self.fc3(src[:, -1, :])

        return output, 0


class LSTMSeriesClassifier3(nn.Module):
    def __init__(self, 
        seq_length, 
        input_size, 
        hidden_size, 
        num_layers, 
        num_classes, 
        dropout,
        positional_dropout,
        device
    ):
        super(LSTMSeriesClassifier3, self).__init__()

        self.conv11 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.conv21 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=1, 
            padding=0
        )

        self.conv22 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.transformed_input_size = input_size 

        self.pe = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.pe2 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.pe3 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.bn1 = nn.BatchNorm1d(
            num_features=input_size
        )

        self.bn2 = nn.BatchNorm1d(
            num_features=input_size
        )

        self.bn3 = nn.BatchNorm1d(
            num_features=input_size
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=2,
        )

        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder2 = nn.TransformerEncoder(
            encoder_layer=encoder_layer2,
            num_layers=2,
        )


        encoder_layer3 = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder3 = nn.TransformerEncoder(
            encoder_layer=encoder_layer3,
            num_layers=2,
        )
        
        self.fc3 = nn.Linear(
            in_features=self.transformed_input_size,
            out_features=num_classes
        )

        self.dropout = nn.Dropout(dropout)

        # self.weights = nn.Parameter(torch.ones(3))
        self.inp_weight = nn.Parameter(torch.ones(1))
        self.conv1_weight = nn.Parameter(torch.ones(1))
        self.conv2_weight = nn.Parameter(torch.ones(1))
        self.final_weight = nn.Parameter(torch.ones(3))
        # self.final_weight = self.final_weight / self.final_weight.sum()
        # self.weights.requires_grad = True
        self.inp_weight.requires_grad = True
        self.conv1_weight.requires_grad = True
        self.conv2_weight.requires_grad = True
        self.final_weight.requires_grad = True
        # # self.dropout2 = nn.Dropout(dropout / 2)


    def forward(self, src, debug=False):
        
        inp = src
        inp_pos = self.pe(inp)
        inp = self.encoder(inp_pos) * self.inp_weight + inp * (1 - self.inp_weight)

        conved1 = self.conv11(src)
        conved1 = F.relu(conved1)
        conved1 = self.dropout(conved1)
        conved1 = torch.transpose(conved1, 1, 2)
        conved1 = self.bn1(conved1).transpose(1, 2)
        conved1_pos = self.pe2(conved1)
        conved1 = self.encoder2(conved1_pos) * self.conv1_weight + conved1 * (1 - self.conv1_weight)

        conved2 = self.conv21(src)
        conved2 = F.relu(conved2)
        conved2 = self.dropout(conved2)
        conved2 = torch.transpose(conved2, 1, 2)
        conved2 = self.bn2(conved2).transpose(1, 2)
        conved2_pos = self.pe2(conved2)
        conved2_pos = self.dropout(conved2_pos)
        conved2 = self.encoder2(conved2_pos) * self.conv2_weight + conved2 * (1 - self.conv2_weight)


        weights = self.final_weight / self.final_weight.sum()
        src = inp * weights[0] + \
              conved1 * weights[1] + \
              conved2 * weights[2]


        output = self.fc3(src[:, -1, :])

        return output, 0

class LSTMSeriesClassifier4(nn.Module):
    def __init__(self, 
        seq_length, 
        input_size, 
        hidden_size, 
        num_layers, 
        num_classes, 
        dropout,
        positional_dropout,
        device
    ):
        super(LSTMSeriesClassifier4, self).__init__()

        self.conv11 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.conv21 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=1, 
            padding=0
        )

        self.conv22 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.transformed_input_size = input_size 

        self.pe = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.pe2 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.pe3 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.bn1 = nn.BatchNorm1d(num_features=input_size)
        self.bn2 = nn.BatchNorm1d(num_features=input_size)
        self.bn3 = nn.BatchNorm1d(num_features=input_size)
        self.bn4 = nn.BatchNorm1d(num_features=input_size)
        self.bn5 = nn.BatchNorm1d(num_features=input_size)
        self.bn6 = nn.BatchNorm1d(num_features=input_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=2,
        )

        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder2 = nn.TransformerEncoder(
            encoder_layer=encoder_layer2,
            num_layers=2,
        )


        encoder_layer3 = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder3 = nn.TransformerEncoder(
            encoder_layer=encoder_layer3,
            num_layers=2,
        )
        
        self.fc3 = nn.Linear(
            in_features=self.transformed_input_size,
            out_features=num_classes
        )

        self.dropout = nn.Dropout(dropout)

        # self.weights = nn.Parameter(torch.ones(3))
        self.inp_weight = nn.Parameter(torch.ones(1))
        self.conv1_weight = nn.Parameter(torch.ones(1))
        self.conv2_weight = nn.Parameter(torch.ones(1))
        self.final_weight = nn.Parameter(torch.ones(3))
        # self.final_weight = self.final_weight / self.final_weight.sum()
        # self.weights.requires_grad = True
        self.inp_weight.requires_grad = True
        self.conv1_weight.requires_grad = True
        self.conv2_weight.requires_grad = True
        self.final_weight.requires_grad = True
        # # self.dropout2 = nn.Dropout(dropout / 2)


    def forward(self, src, debug=False):
        
        inp = src
        inp_pos = self.pe(inp)
        inp = self.encoder(inp_pos) * self.inp_weight + inp * (1 - self.inp_weight)

        conved1 = self.conv11(src)
        conved1 = F.relu(conved1)
        conved1 = self.dropout(conved1)
        conved1 = torch.transpose(conved1, 1, 2)
        conved1 = self.bn1(conved1).transpose(1, 2)
        conved1 = conved1 + src
        conved1 = F.relu(conved1)
        conved1 = torch.transpose(conved1, 1, 2)
        conved1 = self.bn3(conved1).transpose(1, 2)
        conved1_pos = self.pe2(conved1)
        conved1 = self.encoder2(conved1_pos) * self.conv1_weight + conved1 * (1 - self.conv1_weight)

        conved2 = self.conv21(src)
        conved2 = F.relu(conved2)
        conved2 = self.dropout(conved2)
        conved2 = torch.transpose(conved2, 1, 2)
        conved2 = self.bn2(conved2).transpose(1, 2)
        conved2 = self.conv22(conved2)
        conved2 = F.relu(conved2)
        conved2 = self.dropout(conved2)
        conved2 = torch.transpose(conved2, 1, 2)
        conved2 = self.bn5(conved2).transpose(1, 2)
        conved2 = conved2 + src
        conved2 = F.relu(conved2)
        conved2 = torch.transpose(conved2, 1, 2)
        conved2 = self.bn4(conved2).transpose(1, 2)
        conved2_pos = self.pe2(conved2)
        conved2_pos = self.dropout(conved2_pos)
        conved2 = self.encoder2(conved2_pos) * self.conv2_weight + conved2 * (1 - self.conv2_weight)


        weights = self.final_weight / self.final_weight.sum()
        src = inp * weights[0] + \
              conved1 * weights[1] + \
              conved2 * weights[2]


        output = self.fc3(src[:, -1, :])

        return output, 0


# class LSTMSeriesClassifier5(nn.Module):
#     def __init__(self, 
#         seq_length, 
#         input_size, 
#         hidden_size, 
#         num_layers, 
#         num_classes, 
#         dropout,
#         positional_dropout,
#         device
#     ):
#         super(LSTMSeriesClassifier5, self).__init__()

#         self.seq_length = seq_length
#         self.device = device
#         self.conv11 = nn.Conv1d(
#             in_channels=input_size, 
#             out_channels=input_size, 
#             kernel_size=3, 
#             padding=1
#         )

#         self.conv21 = nn.Conv1d(
#             in_channels=input_size, 
#             out_channels=input_size, 
#             kernel_size=1, 
#             padding=0
#         )

#         self.conv22 = nn.Conv1d(
#             in_channels=input_size, 
#             out_channels=input_size, 
#             kernel_size=3, 
#             padding=1
#         )

#         self.conv31 = nn.Conv1d(
#             in_channels=input_size, 
#             out_channels=input_size, 
#             kernel_size=1, 
#             padding=0
#         )

#         self.conv32 = nn.Conv1d(
#             in_channels=input_size, 
#             out_channels=input_size, 
#             kernel_size=3, 
#             padding=1
#         )

#         self.conv33 = nn.Conv1d(
#             in_channels=input_size, 
#             out_channels=input_size, 
#             kernel_size=3, 
#             padding=1
#         )

#         self.conv34 = nn.Conv1d(
#             in_channels=input_size, 
#             out_channels=input_size, 
#             kernel_size=3, 
#             padding=1
#         )
        

#         self.transformed_input_size = input_size 

#         self.pe = PositionalEncoder(
#             device=device,
#             dropout=positional_dropout,
#             max_seq_len=seq_length,
#             embed_dim=self.transformed_input_size
#         )

#         self.pe2 = PositionalEncoder(
#             device=device,
#             dropout=positional_dropout,
#             max_seq_len=seq_length,
#             embed_dim=self.transformed_input_size
#         )

#         self.pe3 = PositionalEncoder(
#             device=device,
#             dropout=positional_dropout,
#             max_seq_len=seq_length,
#             embed_dim=self.transformed_input_size
#         )

#         self.pe4 = PositionalEncoder(
#             device=device,
#             dropout=positional_dropout,
#             max_seq_len=seq_length,
#             embed_dim=self.transformed_input_size
#         )

#         self.bn1 = nn.BatchNorm1d(num_features=input_size)
#         self.bn2 = nn.BatchNorm1d(num_features=input_size)
#         self.bn3 = nn.BatchNorm1d(num_features=input_size)
#         self.bn4 = nn.BatchNorm1d(num_features=input_size)
#         self.bn5 = nn.BatchNorm1d(num_features=input_size)
#         self.bn6 = nn.BatchNorm1d(num_features=input_size)
#         self.bn7 = nn.BatchNorm1d(num_features=input_size)
#         self.bn8 = nn.BatchNorm1d(num_features=input_size)
#         self.bn9 = nn.BatchNorm1d(num_features=input_size)
#         self.bn10 = nn.BatchNorm1d(num_features=input_size)

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.transformed_input_size, 
#             nhead=5,
#             batch_first=True
#         )

#         self.encoder = nn.TransformerEncoder(
#             encoder_layer=encoder_layer,
#             num_layers=2,
#         )

#         encoder_layer2 = nn.TransformerEncoderLayer(
#             d_model=self.transformed_input_size, 
#             nhead=5,
#             batch_first=True
#         )

#         self.encoder2 = nn.TransformerEncoder(
#             encoder_layer=encoder_layer2,
#             num_layers=2,
#         )


#         encoder_layer3 = nn.TransformerEncoderLayer(
#             d_model=self.transformed_input_size, 
#             nhead=5,
#             batch_first=True
#         )

#         self.encoder3 = nn.TransformerEncoder(
#             encoder_layer=encoder_layer3,
#             num_layers=2,
#         )

#         encoder_layer4 = nn.TransformerEncoderLayer(
#             d_model=self.transformed_input_size, 
#             nhead=5,
#             batch_first=True
#         )

#         self.encoder4 = nn.TransformerEncoder(
#             encoder_layer=encoder_layer4,
#             num_layers=2,
#         )
        
#         self.fc3 = nn.Linear(
#             in_features=self.transformed_input_size,
#             out_features=num_classes
#         )

#         self.dropout = nn.Dropout(dropout)

#         # self.weights = nn.Parameter(torch.ones(3))
#         self.inp_weight = nn.Parameter(torch.ones(2))
#         self.conv1_weight = nn.Parameter(torch.ones(2))
#         self.conv2_weight = nn.Parameter(torch.ones(2))
#         self.final_weight = nn.Parameter(torch.ones(4))
#         self.conv3_weight = nn.Parameter(torch.ones(2))
#         # self.final_weight = self.final_weight / self.final_weight.sum()
#         # self.weights.requires_grad = True
#         self.inp_weight.requires_grad = True
#         self.conv1_weight.requires_grad = True
#         self.conv2_weight.requires_grad = True
#         self.final_weight.requires_grad = True
#         self.conv3_weight.requires_grad = True
#         # # self.dropout2 = nn.Dropout(dropout / 2)


#     def forward(self, src, debug=False, save_path = None):
        
#         seq_len = self.seq_length
#         mask = torch.tril(torch.ones((self.seq_length, self.seq_length), device=self.device))
#         # mask = mask == 1
#         inp = src
#         inp_pos = self.pe(inp)
#         if not save_path:
#             inp_weight = self.inp_weight / torch.sum(self.inp_weight)
#             inp = self.encoder(inp_pos) * inp_weight[0] + inp * inp_weight[1]
#         else:
#             transformer_1_attn_maps = []
#             norm_first = False
#             batch_size = src.shape[0]
#             src_mask = torch.zeros((seq_len, seq_len)).bool().to(self.device)
#             src_key_padding_mask = torch.zeros((batch_size, seq_len)).bool().to(self.device)

#             for i in range(2):
#                 # compute attention of layer i
#                 h = inp_pos.clone()
#                 if norm_first:
#                     h = self.encoder.layers[i].norm1(h)
#                 attn = self.encoder.layers[i].self_attn(h, h, h,attn_mask=src_mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
#                 transformer_1_attn_maps.append(attn)
#                 # forward of layer i
#                 # inp_pos = self.encoder.layers[i](inp_pos,src_mask=mask,src_key_padding_mask=None)
#                 inp_pos = self.encoder.layers[i](inp_pos,src_key_padding_mask=None)

#             # inp = inp_pos * self.inp_weight + inp * (1 - self.inp_weight)
#             inp_weight = self.inp_weight / torch.sum(self.inp_weight)
#             inp = inp_pos * inp_weight[0] + inp * inp_weight[1]

#         # conv 1
#         src = src.permute((0, 2, 1))
#         conved1 = self.conv11(src)
#         conved1 = F.relu(conved1)
#         conved1 = self.dropout(conved1)
#         # conved1 = torch.transpose(conved1, 1, 2)
#         # conved1 = self.bn1(conved1).transpose(1, 2)
#         conved1 = self.bn1(conved1)
        
#         conved1 = conved1 + src
#         conved1 = F.relu(conved1)
#         # conved1 = torch.transpose(conved1, 1, 2)
#         conved1 = self.bn2(conved1).transpose(1, 2)
#         conved1_pos = self.pe2(conved1)
#         # print(conved1_pos)
#         if not save_path:
#             conv1_weight = self.conv1_weight / torch.sum(self.conv1_weight)
#             conved1 = self.encoder2(conved1_pos) * conv1_weight[0] + conved1 * self.conv1_weight[1]
#         else:
#             transformer_2_attn_maps = []
#             norm_first = False
#             batch_size = src.shape[0]
#             src_mask = torch.zeros((seq_len, seq_len)).bool().to(self.device)
#             src_key_padding_mask = torch.zeros((batch_size, seq_len)).bool().to(self.device)

#             for i in range(2):
#                 # compute attention of layer i
#                 h = conved1_pos.clone()
#                 # print(h)
#                 if norm_first:
#                     h = self.encoder2.layers[i].norm1(h)
#                 # attn = self.encoder2.layers[i].self_attn(h, h, h,attn_mask=mask,key_padding_mask=None,need_weights=True)[1]
#                 attn = self.encoder2.layers[i].self_attn(h, h, h,key_padding_mask=None,need_weights=True)[1]
#                 transformer_2_attn_maps.append(attn)
#                 # forward of layer i
#                 # conved1_pos = self.encoder2.layers[i](conved1_pos,src_mask=mask,src_key_padding_mask=None)
#                 conved1_pos = self.encoder2.layers[i](conved1_pos,src_key_padding_mask=None)

#             # conved1 = conved1_pos * self.conv1_weight + conved1 * (1 - self.conv1_weight)
#             conv1_weight = self.conv1_weight / torch.sum(self.conv1_weight)
#             conved1 = conved1_pos * conv1_weight[0] + conved1 * self.conv1_weight[1]

#         # conv 1 -> conv 3
#         conved2 = self.conv21(src)
#         conved2 = F.relu(conved2)
#         conved2 = self.dropout(conved2)
#         # conved2 = torch.transpose(conved2, 1, 2)
#         # conved2 = self.bn3(conved2).transpose(1, 2)
#         conved2 = self.bn3(conved2)

#         conved2 = self.conv22(conved2)
#         conved2 = F.relu(conved2)
#         conved2 = self.dropout(conved2)
#         # conved2 = torch.transpose(conved2, 1, 2)
#         # conved2 = self.bn4(conved2).transpose(1, 2)
#         conved2 = self.bn4(conved2)

#         conved2 = conved2 + src
#         conved2 = F.relu(conved2)
#         # conved2 = torch.transpose(conved2, 1, 2)
#         conved2 = self.bn5(conved2).transpose(1, 2)
#         conved2_pos = self.pe3(conved2)
#         conved2_pos = self.dropout(conved2_pos)
#         if not save_path:
#             conv2_weight = self.conv2_weight / torch.sum(self.conv2_weight)
#             conved2 = self.encoder3(conved2_pos) * conv2_weight[0] + conved2 * self.conv2_weight[1]
#         else:
#             transformer_3_attn_maps = []
#             norm_first = False
#             batch_size = src.shape[0]
#             src_mask = torch.zeros((seq_len, seq_len)).bool().to(self.device)
#             src_key_padding_mask = torch.zeros((batch_size, seq_len)).bool().to(self.device)

#             for i in range(2):
#                 # compute attention of layer i
#                 h = conved2_pos.clone()
#                 if norm_first:
#                     h = self.encoder3.layers[i].norm1(h)
#                 # attn = self.encoder3.layers[i].self_attn(h, h, h,attn_mask=src_mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
#                 attn = self.encoder3.layers[i].self_attn(h, h, h,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
#                 transformer_3_attn_maps.append(attn)
#                 # forward of layer i
#                 # conved2_pos = self.encoder3.layers[i](conved2_pos,src_mask=mask,src_key_padding_mask=None)
#                 conved2_pos = self.encoder3.layers[i](conved2_pos,src_key_padding_mask=None)
        
#             # conved2 = conved2_pos * self.conv2_weight + conved2 * (1 - self.conv2_weight)
#             conv2_weight = self.conv2_weight / torch.sum(self.conv2_weight)
#             conved2 = conved2_pos * conv2_weight[0] + conved2 * self.conv2_weight[1]

#         # conv 1 -> conv 3 -> conv 3 -> conv 3
#         conved3 = self.conv31(src)
#         conved3 = F.relu(conved3)
#         conved3 = self.dropout(conved3)
#         # conved3 = torch.transpose(conved3, 1, 2)
#         # conved3 = self.bn6(conved3).transpose(1, 2)
#         conved3 = self.bn6(conved3)

#         conved3 = self.conv32(conved3)
#         conved3 = F.relu(conved3)
#         conved3 = self.dropout(conved3)
#         # conved3 = torch.transpose(conved3, 1, 2)
#         # conved3 = self.bn7(conved3).transpose(1, 2)
#         conved3 = self.bn7(conved3)

#         conved3 = self.conv33(conved3)
#         conved3 = F.relu(conved3)
#         conved3 = self.dropout(conved3)
#         # conved3 = torch.transpose(conved3, 1, 2)
#         # conved3 = self.bn8(conved3).transpose(1, 2)
#         conved3 = self.bn8(conved3)

#         conved3 = self.conv34(conved3)
#         conved3 = F.relu(conved3)
#         conved3 = self.dropout(conved3)
#         # conved3 = torch.transpose(conved3, 1, 2)
#         # conved3 = self.bn9(conved3).transpose(1, 2)
#         conved3 = self.bn9(conved3)

#         conved3 = conved3 + src
#         conved3 = F.relu(conved3)
#         # conved3 = torch.transpose(conved3, 1, 2)
#         conved3 = self.bn10(conved3).transpose(1, 2)
#         conved3_pos = self.pe4(conved3)
#         conved3_pos = self.dropout(conved3_pos)
#         if not save_path:
#             conv3_weight = self.conv3_weight / torch.sum(self.conv3_weight)
#             conved3 = self.encoder4(conved3_pos) * conv3_weight[0] + conved3 * conv3_weight[1]
#         else:
#             transformer_4_attn_maps = []
#             norm_first = False
#             batch_size = src.shape[0]
#             src_mask = torch.zeros((seq_len, seq_len)).bool().to(self.device)
#             src_key_padding_mask = torch.zeros((batch_size, seq_len)).bool().to(self.device)
            
#             for i in range(2):
#                 # compute attention of layer i
#                 h = conved3_pos.clone()
#                 if norm_first:
#                     h = self.encoder4.layers[i].norm1(h)
#                 # attn = self.encoder4.layers[i].self_attn(h, h, h,attn_mask=src_mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
#                 attn = self.encoder4.layers[i].self_attn(h, h, h,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
#                 transformer_4_attn_maps.append(attn)
#                 # forward of layer i
#                 # conved3_pos = self.encoder4.layers[i](conved3_pos,src_mask=mask,src_key_padding_mask=None)
#                 conved3_pos = self.encoder4.layers[i](conved3_pos,src_key_padding_mask=None)
        
#             # conved3 = conved3_pos * self.conv3_weight + conved3 * (1 - self.conv3_weight)
#             conv3_weight = self.conv3_weight / torch.sum(self.conv3_weight)
#             conved3 = conved3_pos * conv3_weight[0] + conved3 * conv3_weight[1]

#         weights = self.final_weight / self.final_weight.sum()
#         src = inp * weights[0] + \
#               conved1 * weights[1] + \
#               conved2 * weights[2] + \
#               conved3 * weights[3]

#         if save_path:
#             all_weights = {
#                 "channel_1_attn_map" : transformer_1_attn_maps,
#                 "channel_2_attn_map" : transformer_2_attn_maps,
#                 "channel_3_attn_map" : transformer_3_attn_maps,
#                 "channel_4_attn_map" : transformer_4_attn_maps,
#                 "channel_1_weights" : self.inp_weight,
#                 "channel_2_weights" : self.conv1_weight,
#                 "channel_3_weights" : self.conv2_weight,
#                 "channel_4_weights" : self.conv3_weight,
#                 "final_weights" : self.final_weight
#             }

#             torch.save(all_weights, save_path)

#         output = self.fc3(src[:, -1, :])
        
#         return output, 0

class LSTMSeriesClassifier5(nn.Module):
    def __init__(self, 
        seq_length, 
        input_size, 
        hidden_size, 
        num_layers, 
        num_classes, 
        dropout,
        positional_dropout,
        device
    ):
        super(LSTMSeriesClassifier5, self).__init__()

        self.seq_length = seq_length
        self.device = device
        self.conv11 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.conv21 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=1, 
            padding=0
        )

        self.conv22 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.conv31 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=1, 
            padding=0
        )

        self.conv32 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.conv33 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.conv34 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.transformed_input_size = input_size 

        self.pe = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.pe2 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.pe3 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.pe4 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.bn1 = nn.BatchNorm1d(num_features=input_size)
        self.bn2 = nn.BatchNorm1d(num_features=input_size)
        self.bn3 = nn.BatchNorm1d(num_features=input_size)
        self.bn4 = nn.BatchNorm1d(num_features=input_size)
        self.bn5 = nn.BatchNorm1d(num_features=input_size)
        self.bn6 = nn.BatchNorm1d(num_features=input_size)
        self.bn7 = nn.BatchNorm1d(num_features=input_size)
        self.bn8 = nn.BatchNorm1d(num_features=input_size)
        self.bn9 = nn.BatchNorm1d(num_features=input_size)
        self.bn10 = nn.BatchNorm1d(num_features=input_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=2,
        )

        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder2 = nn.TransformerEncoder(
            encoder_layer=encoder_layer2,
            num_layers=2,
        )


        encoder_layer3 = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder3 = nn.TransformerEncoder(
            encoder_layer=encoder_layer3,
            num_layers=2,
        )

        encoder_layer4 = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder4 = nn.TransformerEncoder(
            encoder_layer=encoder_layer4,
            num_layers=2,
        )
        
        self.fc3 = nn.Linear(
            in_features=self.transformed_input_size,
            out_features=num_classes
        )

        self.dropout = nn.Dropout(dropout)

        # self.weights = nn.Parameter(torch.ones(3))
        self.inp_weight = nn.Parameter(torch.rand(2))
        self.conv1_weight = nn.Parameter(torch.rand(2))
        self.conv2_weight = nn.Parameter(torch.rand(3))
        self.final_weight = nn.Parameter(torch.rand(4))
        self.conv3_weight = nn.Parameter(torch.rand(2))
        # self.final_weight = self.final_weight / self.final_weight.sum()
        # self.weights.requires_grad = True
        self.inp_weight.requires_grad = True
        self.conv1_weight.requires_grad = True
        self.conv2_weight.requires_grad = True
        self.final_weight.requires_grad = True
        self.conv3_weight.requires_grad = True
        # # self.dropout2 = nn.Dropout(dropout / 2)


    def forward(self, src, debug=False, save_path = None):
        
        seq_len = self.seq_length
        mask = torch.tril(torch.ones((self.seq_length, self.seq_length), device=self.device))
        # mask = mask == 1
        inp = src
        inp_pos = self.pe(inp)
        if not save_path:
            inp_weight = self.inp_weight / torch.sum(self.inp_weight)
            inp = self.encoder(inp_pos) * inp_weight[0] + inp * inp_weight[1]
        else:
            transformer_1_attn_maps = []
            norm_first = False
            batch_size = src.shape[0]
            src_mask = torch.zeros((seq_len, seq_len)).bool().to(self.device)
            src_key_padding_mask = torch.zeros((batch_size, seq_len)).bool().to(self.device)

            for i in range(2):
                # compute attention of layer i
                h = inp_pos.clone()
                if norm_first:
                    h = self.encoder.layers[i].norm1(h)
                attn = self.encoder.layers[i].self_attn(h, h, h,attn_mask=src_mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
                transformer_1_attn_maps.append(attn)
                # forward of layer i
                inp_pos = self.encoder.layers[i](inp_pos,src_mask=mask,src_key_padding_mask=None)
                # inp_pos = self.encoder.layers[i](inp_pos,src_key_padding_mask=None)

            # inp = inp_pos * self.inp_weight + inp * (1 - self.inp_weight)
            # inp_weight = self.inp_weight / torch.sum(self.inp_weight)
            # inp = inp_pos * inp_weight[0] + inp * inp_weight[1]

        # conv 1
        # src = src.permute((0, 2, 1))
        conved1 = self.conv11(src)
        conved1 = F.relu(conved1)
        conved1 = self.dropout(conved1)
        conved1 = torch.transpose(conved1, 1, 2)
        conved1 = self.bn1(conved1).transpose(1, 2)
        # conved1 = self.bn1(conved1)
        
        conved1 = conved1 + src
        conved1 = F.relu(conved1)
        conved1 = torch.transpose(conved1, 1, 2)
        conved1 = self.bn2(conved1).transpose(1, 2)
        conved1_pos = self.pe2(conved1)
        # print(conved1_pos)
        if not save_path:
            conv1_weight = self.conv1_weight / torch.sum(self.conv1_weight)
            conved1 = self.encoder2(conved1_pos) * conv1_weight[0] + conved1 * self.conv1_weight[1]
        else:
            transformer_2_attn_maps = []
            norm_first = False
            batch_size = src.shape[0]
            src_mask = torch.zeros((seq_len, seq_len)).bool().to(self.device)
            src_key_padding_mask = torch.zeros((batch_size, seq_len)).bool().to(self.device)

            for i in range(2):
                # compute attention of layer i
                h = conved1_pos.clone()
                # print(h)
                if norm_first:
                    h = self.encoder2.layers[i].norm1(h)
                # attn = self.encoder2.layers[i].self_attn(h, h, h,attn_mask=mask,key_padding_mask=None,need_weights=True)[1]
                attn = self.encoder2.layers[i].self_attn(h, h, h,key_padding_mask=None,need_weights=True)[1]
                transformer_2_attn_maps.append(attn)
                # forward of layer i
                # conved1_pos = self.encoder2.layers[i](conved1_pos,src_mask=mask,src_key_padding_mask=None)
                conved1_pos = self.encoder2.layers[i](conved1_pos,src_key_padding_mask=None)

            # conved1 = conved1_pos * self.conv1_weight + conved1 * (1 - self.conv1_weight)
            conv1_weight = self.conv1_weight / torch.sum(self.conv1_weight)
            conved1 = conved1_pos * conv1_weight[0] + conved1 * self.conv1_weight[1]

        # conv 1 -> conv 3
        conved2 = self.conv21(src)
        conved2 = F.relu(conved2)
        conved2 = self.dropout(conved2)
        conved2 = torch.transpose(conved2, 1, 2)
        conved2 = self.bn3(conved2).transpose(1, 2)
        # conved2 = self.bn3(conved2)

        conved2 = self.conv22(conved2)
        conved2 = F.relu(conved2)
        conved2 = self.dropout(conved2)
        conved2 = torch.transpose(conved2, 1, 2)
        conved2 = self.bn4(conved2).transpose(1, 2)
        # conved2 = self.bn4(conved2)

        conved2 = conved2 + src
        conved2 = F.relu(conved2)
        conved2 = torch.transpose(conved2, 1, 2)
        conved2 = self.bn5(conved2).transpose(1, 2)
        conved2_pos = self.pe3(conved2)
        conved2_pos = self.dropout(conved2_pos)
        if not save_path:
            conv2_weight = self.conv2_weight / torch.sum(self.conv2_weight)
            conved2 = self.encoder3(conved2_pos) * conv2_weight[0] + conved2 * self.conv2_weight[1]
        else:
            transformer_3_attn_maps = []
            norm_first = False
            batch_size = src.shape[0]
            src_mask = torch.zeros((seq_len, seq_len)).bool().to(self.device)
            src_key_padding_mask = torch.zeros((batch_size, seq_len)).bool().to(self.device)

            for i in range(2):
                # compute attention of layer i
                h = conved2_pos.clone()
                if norm_first:
                    h = self.encoder3.layers[i].norm1(h)
                # attn = self.encoder3.layers[i].self_attn(h, h, h,attn_mask=src_mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
                attn = self.encoder3.layers[i].self_attn(h, h, h,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
                transformer_3_attn_maps.append(attn)
                # forward of layer i
                # conved2_pos = self.encoder3.layers[i](conved2_pos,src_mask=mask,src_key_padding_mask=None)
                conved2_pos = self.encoder3.layers[i](conved2_pos,src_key_padding_mask=None)
        
            # conved2 = conved2_pos * self.conv2_weight + conved2 * (1 - self.conv2_weight)
            conv2_weight = self.conv2_weight / torch.sum(self.conv2_weight)
            conved2 = conved2_pos * conv2_weight[0] + conved2 * self.conv2_weight[1]

        # conv 1 -> conv 3 -> conv 3 -> conv 3
        conved3 = self.conv31(src)
        conved3 = F.relu(conved3)
        conved3 = self.dropout(conved3)
        conved3 = torch.transpose(conved3, 1, 2)
        conved3 = self.bn6(conved3).transpose(1, 2)
        # conved3 = self.bn6(conved3)

        conved3 = self.conv32(conved3)
        conved3 = F.relu(conved3)
        conved3 = self.dropout(conved3)
        conved3 = torch.transpose(conved3, 1, 2)
        conved3 = self.bn7(conved3).transpose(1, 2)
        # conved3 = self.bn7(conved3)

        conved3 = self.conv33(conved3)
        conved3 = F.relu(conved3)
        conved3 = self.dropout(conved3)
        conved3 = torch.transpose(conved3, 1, 2)
        conved3 = self.bn8(conved3).transpose(1, 2)
        # conved3 = self.bn8(conved3)

        conved3 = self.conv34(conved3)
        conved3 = F.relu(conved3)
        conved3 = self.dropout(conved3)
        conved3 = torch.transpose(conved3, 1, 2)
        conved3 = self.bn9(conved3).transpose(1, 2)
        # conved3 = self.bn9(conved3)

        conved3 = conved3 + src
        conved3 = F.relu(conved3)
        conved3 = torch.transpose(conved3, 1, 2)
        conved3 = self.bn10(conved3).transpose(1, 2)
        conved3_pos = self.pe4(conved3)
        conved3_pos = self.dropout(conved3_pos)
        if not save_path:
            conv3_weight = self.conv3_weight / torch.sum(self.conv3_weight)
            conved3 = self.encoder4(conved3_pos) * conv3_weight[0] + conved3 * conv3_weight[1]
        else:
            transformer_4_attn_maps = []
            norm_first = False
            batch_size = src.shape[0]
            src_mask = torch.zeros((seq_len, seq_len)).bool().to(self.device)
            src_key_padding_mask = torch.zeros((batch_size, seq_len)).bool().to(self.device)
            
            for i in range(2):
                # compute attention of layer i
                h = conved3_pos.clone()
                if norm_first:
                    h = self.encoder4.layers[i].norm1(h)
                # attn = self.encoder4.layers[i].self_attn(h, h, h,attn_mask=src_mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
                attn = self.encoder4.layers[i].self_attn(h, h, h,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
                transformer_4_attn_maps.append(attn)
                # forward of layer i
                # conved3_pos = self.encoder4.layers[i](conved3_pos,src_mask=mask,src_key_padding_mask=None)
                conved3_pos = self.encoder4.layers[i](conved3_pos,src_key_padding_mask=None)
        
            # conved3 = conved3_pos * self.conv3_weight + conved3 * (1 - self.conv3_weight)
            conv3_weight = self.conv3_weight / torch.sum(self.conv3_weight)
            conved3 = conved3_pos * conv3_weight[0] + conved3 * conv3_weight[1]

        weights = self.final_weight / self.final_weight.sum()
        src = inp * weights[0] + \
              conved1 * weights[1] + \
              conved2 * weights[2] + \
              conved3 * weights[3]

        if save_path:
            all_weights = {
                "channel_1_attn_map" : transformer_1_attn_maps,
                "channel_2_attn_map" : transformer_2_attn_maps,
                "channel_3_attn_map" : transformer_3_attn_maps,
                "channel_4_attn_map" : transformer_4_attn_maps,
                "channel_1_weights" : self.inp_weight,
                "channel_2_weights" : self.conv1_weight,
                "channel_3_weights" : self.conv2_weight,
                "channel_4_weights" : self.conv3_weight,
                "final_weights" : self.final_weight
            }

            torch.save(all_weights, save_path)

        output = self.fc3(src[:, -1, :])
        
        return output, 0


class LSTMSeriesClassifier4(nn.Module):
    def __init__(self, 
        seq_length, 
        input_size, 
        hidden_size, 
        num_layers, 
        num_classes, 
        dropout,
        positional_dropout,
        device
    ):
        super(LSTMSeriesClassifier4, self).__init__()

        self.conv11 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.conv21 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=1, 
            padding=0
        )

        self.conv22 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.transformed_input_size = input_size 

        self.pe = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.pe2 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.pe3 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.bn1 = nn.BatchNorm1d(num_features=input_size)
        self.bn2 = nn.BatchNorm1d(num_features=input_size)
        self.bn3 = nn.BatchNorm1d(num_features=input_size)
        self.bn4 = nn.BatchNorm1d(num_features=input_size)
        self.bn5 = nn.BatchNorm1d(num_features=input_size)
        self.bn6 = nn.BatchNorm1d(num_features=input_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=2,
        )

        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder2 = nn.TransformerEncoder(
            encoder_layer=encoder_layer2,
            num_layers=2,
        )


        encoder_layer3 = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder3 = nn.TransformerEncoder(
            encoder_layer=encoder_layer3,
            num_layers=2,
        )
        
        self.fc3 = nn.Linear(
            in_features=self.transformed_input_size,
            out_features=num_classes
        )

        self.dropout = nn.Dropout(dropout)

        # self.weights = nn.Parameter(torch.ones(3))
        self.inp_weight = nn.Parameter(torch.ones(1))
        self.conv1_weight = nn.Parameter(torch.ones(1))
        self.conv2_weight = nn.Parameter(torch.ones(1))
        self.final_weight = nn.Parameter(torch.ones(3))
        # self.final_weight = self.final_weight / self.final_weight.sum()
        # self.weights.requires_grad = True
        self.inp_weight.requires_grad = True
        self.conv1_weight.requires_grad = True
        self.conv2_weight.requires_grad = True
        self.final_weight.requires_grad = True
        # # self.dropout2 = nn.Dropout(dropout / 2)


    def forward(self, src, debug=False):
        
        inp = src
        inp_pos = self.pe(inp)
        inp = self.encoder(inp_pos) * self.inp_weight + inp * (1 - self.inp_weight)

        conved1 = self.conv11(src)
        conved1 = F.relu(conved1)
        conved1 = self.dropout(conved1)
        conved1 = torch.transpose(conved1, 1, 2)
        conved1 = self.bn1(conved1).transpose(1, 2)
        conved1 = conved1 + src
        conved1 = F.relu(conved1)
        conved1 = torch.transpose(conved1, 1, 2)
        conved1 = self.bn3(conved1).transpose(1, 2)
        conved1_pos = self.pe2(conved1)
        conved1 = self.encoder2(conved1_pos) * self.conv1_weight + conved1 * (1 - self.conv1_weight)

        conved2 = self.conv21(src)
        conved2 = F.relu(conved2)
        conved2 = self.dropout(conved2)
        conved2 = torch.transpose(conved2, 1, 2)
        conved2 = self.bn2(conved2).transpose(1, 2)
        conved2 = self.conv22(conved2)
        conved2 = F.relu(conved2)
        conved2 = self.dropout(conved2)
        conved2 = torch.transpose(conved2, 1, 2)
        conved2 = self.bn5(conved2).transpose(1, 2)
        conved2 = conved2 + src
        conved2 = F.relu(conved2)
        conved2 = torch.transpose(conved2, 1, 2)
        conved2 = self.bn4(conved2).transpose(1, 2)
        conved2_pos = self.pe2(conved2)
        conved2_pos = self.dropout(conved2_pos)
        conved2 = self.encoder2(conved2_pos) * self.conv2_weight + conved2 * (1 - self.conv2_weight)


        weights = self.final_weight / self.final_weight.sum()
        src = inp * weights[0] + \
              conved1 * weights[1] + \
              conved2 * weights[2]


        output = self.fc3(src[:, -1, :])

        return output, 0


class LSTMSeriesClassifier6(nn.Module):
    def __init__(self, 
        seq_length, 
        input_size, 
        hidden_size, 
        num_layers, 
        num_classes, 
        dropout,
        positional_dropout,
        device
    ):
        super(LSTMSeriesClassifier6, self).__init__()

        self.seq_length = seq_length
        self.device = device
        self.conv11 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.conv21 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=1, 
            padding=0
        )

        self.conv22 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.conv31 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=1, 
            padding=0
        )

        self.conv32 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.conv33 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.conv34 = nn.Conv1d(
            in_channels=seq_length, 
            out_channels=seq_length, 
            kernel_size=3, 
            padding=1
        )

        self.transformed_input_size = input_size 

        self.pe = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.pe2 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.pe3 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.pe4 = PositionalEncoder(
            device=device,
            dropout=positional_dropout,
            max_seq_len=seq_length,
            embed_dim=self.transformed_input_size
        )

        self.bn1 = nn.BatchNorm1d(num_features=input_size)
        self.bn2 = nn.BatchNorm1d(num_features=input_size)
        self.bn3 = nn.BatchNorm1d(num_features=input_size)
        self.bn4 = nn.BatchNorm1d(num_features=input_size)
        self.bn5 = nn.BatchNorm1d(num_features=input_size)
        self.bn6 = nn.BatchNorm1d(num_features=input_size)
        self.bn7 = nn.BatchNorm1d(num_features=input_size)
        self.bn8 = nn.BatchNorm1d(num_features=input_size)
        self.bn9 = nn.BatchNorm1d(num_features=input_size)
        self.bn10 = nn.BatchNorm1d(num_features=input_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=2,
        )

        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder2 = nn.TransformerEncoder(
            encoder_layer=encoder_layer2,
            num_layers=2,
        )


        encoder_layer3 = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder3 = nn.TransformerEncoder(
            encoder_layer=encoder_layer3,
            num_layers=2,
        )

        encoder_layer4 = nn.TransformerEncoderLayer(
            d_model=self.transformed_input_size, 
            nhead=5,
            batch_first=True
        )

        self.encoder4 = nn.TransformerEncoder(
            encoder_layer=encoder_layer4,
            num_layers=2,
        )
        
        self.fc3 = nn.Linear(
            in_features=self.transformed_input_size,
            out_features=num_classes
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.dropout6 = nn.Dropout(dropout)
        self.dropout7 = nn.Dropout(dropout)
        self.dropout8 = nn.Dropout(dropout)
        self.dropout9 = nn.Dropout(dropout)
        self.dropout10 = nn.Dropout(dropout)
        self.dropout11 = nn.Dropout(dropout)

        # self.weights = nn.Parameter(torch.ones(3))
        self.inp_weight = nn.Parameter(torch.ones(1))
        self.conv1_weight = nn.Parameter(torch.ones(1))
        self.conv2_weight = nn.Parameter(torch.ones(1))
        self.final_weight = nn.Parameter(torch.ones(4))
        self.conv3_weight = nn.Parameter(torch.ones(1))
        # self.final_weight = self.final_weight / self.final_weight.sum()
        # self.weights.requires_grad = True
        self.inp_weight.requires_grad = True
        self.conv1_weight.requires_grad = True
        self.conv2_weight.requires_grad = True
        self.final_weight.requires_grad = True
        self.conv3_weight.requires_grad = True
        # # self.dropout2 = nn.Dropout(dropout / 2)


    def forward(self, src, debug=False, save_path = None):
        
        seq_len = self.seq_length
        mask = torch.tril(torch.ones((self.seq_length, self.seq_length), device=self.device))
        # print(mask)
        inp = src
        inp_pos = self.pe(inp)
        # inp_pos = self.dropout11(inp_pos)
        if not save_path:
            inp = self.encoder(inp_pos, mask) * self.inp_weight + inp * (1 - self.inp_weight)
        else:
            transformer_1_attn_maps = []
            norm_first = False
            batch_size = src.shape[0]
            src_mask = torch.zeros((seq_len, seq_len)).bool().to(self.device)
            src_key_padding_mask = torch.zeros((batch_size, seq_len)).bool().to(self.device)

            for i in range(2):
                # compute attention of layer i
                h = inp_pos.clone()
                if norm_first:
                    h = self.encoder.layers[i].norm1(h)
                attn = self.encoder.layers[i].self_attn(h, h, h,attn_mask=src_mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
                transformer_1_attn_maps.append(attn)
                # forward of layer i
                inp_pos = self.encoder.layers[i](inp_pos,src_mask=mask,src_key_padding_mask=None)

            inp = inp_pos * self.inp_weight + inp * (1 - self.inp_weight)

        # conv 1
        conved1 = self.conv11(src)
        conved1 = F.relu(conved1)
        conved1 = self.dropout1(conved1)
        conved1 = torch.transpose(conved1, 1, 2)
        conved1 = self.bn1(conved1).transpose(1, 2)

        conved1 = conved1 + src
        conved1 = F.relu(conved1)
        conved1 = torch.transpose(conved1, 1, 2)
        conved1 = self.bn2(conved1).transpose(1, 2)
        conved1_pos = self.pe2(conved1)
        # conved1_pos = self.dropout10(conved1_pos)
        if not save_path:
            conved1 = self.encoder2(conved1_pos, mask) * self.conv1_weight + conved1 * (1 - self.conv1_weight)
        else:
            transformer_2_attn_maps = []
            norm_first = False
            batch_size = src.shape[0]
            src_mask = torch.zeros((seq_len, seq_len)).bool().to(self.device)
            src_key_padding_mask = torch.zeros((batch_size, seq_len)).bool().to(self.device)

            for i in range(2):
                # compute attention of layer i
                h = conved1_pos.clone()
                if norm_first:
                    h = self.encoder2.layers[i].norm1(h)
                attn = self.encoder2.layers[i].self_attn(h, h, h,attn_mask=src_mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
                transformer_2_attn_maps.append(attn)
                # forward of layer i
                conved1_pos = self.encoder2.layers[i](conved1_pos,src_mask=mask,src_key_padding_mask=None)

            conved1 = conved1_pos * self.conv1_weight + conved1 * (1 - self.conv1_weight)

        # conv 1 -> conv 3
        conved2 = self.conv21(src)
        conved2 = F.relu(conved2)
        conved2 = self.dropout2(conved2)
        conved2 = torch.transpose(conved2, 1, 2)
        conved2 = self.bn3(conved2).transpose(1, 2)

        conved2 = self.conv22(conved2)
        conved2 = F.relu(conved2)
        conved2 = self.dropout3(conved2)
        conved2 = torch.transpose(conved2, 1, 2)
        conved2 = self.bn4(conved2).transpose(1, 2)

        conved2 = conved2 + src
        conved2 = F.relu(conved2)
        conved2 = torch.transpose(conved2, 1, 2)
        conved2 = self.bn5(conved2).transpose(1, 2)
        conved2_pos = self.pe3(conved2)
        # conved2_pos = self.dropout9(conved2_pos)
        if not save_path:
            conved2 = self.encoder3(conved2_pos, mask) * self.conv2_weight + conved2 * (1 - self.conv2_weight)
        else:
            transformer_3_attn_maps = []
            norm_first = False
            batch_size = src.shape[0]
            src_mask = torch.zeros((seq_len, seq_len)).bool().to(self.device)
            src_key_padding_mask = torch.zeros((batch_size, seq_len)).bool().to(self.device)

            for i in range(2):
                # compute attention of layer i
                h = conved2_pos.clone()
                if norm_first:
                    h = self.encoder3.layers[i].norm1(h)
                attn = self.encoder3.layers[i].self_attn(h, h, h,attn_mask=src_mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
                transformer_3_attn_maps.append(attn)
                # forward of layer i
                conved2_pos = self.encoder3.layers[i](conved2_pos,src_mask=mask,src_key_padding_mask=None)
        
            conved2 = conved2_pos * self.conv2_weight + conved2 * (1 - self.conv2_weight)

        # conv 1 -> conv 3 -> conv 3 -> conv 3
        conved3 = self.conv31(src)
        conved3 = F.relu(conved3)
        conved3 = self.dropout4(conved3)
        conved3 = torch.transpose(conved3, 1, 2)
        conved3 = self.bn6(conved3).transpose(1, 2)

        conved3 = self.conv32(conved3)
        conved3 = F.relu(conved3)
        conved3 = self.dropout5(conved3)
        conved3 = torch.transpose(conved3, 1, 2)
        conved3 = self.bn7(conved3).transpose(1, 2)

        conved3 = self.conv33(conved3)
        conved3 = F.relu(conved3)
        conved3 = self.dropout6(conved3)
        conved3 = torch.transpose(conved3, 1, 2)
        conved3 = self.bn8(conved3).transpose(1, 2)

        conved3 = self.conv34(conved3)
        conved3 = F.relu(conved3)
        conved3 = self.dropout7(conved3)
        conved3 = torch.transpose(conved3, 1, 2)
        conved3 = self.bn9(conved3).transpose(1, 2)

        conved3 = conved3 + src
        conved3 = F.relu(conved3)
        conved3 = torch.transpose(conved3, 1, 2)
        conved3 = self.bn10(conved3).transpose(1, 2)
        conved3_pos = self.pe4(conved3)
        # conved3_pos = self.dropout8(conved3_pos)
        if not save_path:
            conved3 = self.encoder4(conved3_pos, mask) * self.conv3_weight + conved3 * (1 - self.conv3_weight)
        else:
            transformer_4_attn_maps = []
            norm_first = False
            batch_size = src.shape[0]
            src_mask = torch.zeros((seq_len, seq_len)).bool().to(self.device)
            src_key_padding_mask = torch.zeros((batch_size, seq_len)).bool().to(self.device)
            
            for i in range(2):
                # compute attention of layer i
                h = conved3_pos.clone()
                if norm_first:
                    h = self.encoder4.layers[i].norm1(h)
                attn = self.encoder4.layers[i].self_attn(h, h, h,attn_mask=src_mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
                transformer_4_attn_maps.append(attn)
                # forward of layer i
                conved3_pos = self.encoder4.layers[i](conved3_pos,src_mask=mask,src_key_padding_mask=None)
        
            conved3 = conved3_pos * self.conv3_weight + conved3 * (1 - self.conv3_weight)

        weights = self.final_weight / self.final_weight.sum()
        src = inp * weights[0] + \
              conved1 * weights[1] + \
              conved2 * weights[2] + \
              conved3 * weights[3]

        if save_path:
            all_weights = {
                "channel_1_attn_map" : transformer_1_attn_maps,
                "channel_2_attn_map" : transformer_2_attn_maps,
                "channel_3_attn_map" : transformer_3_attn_maps,
                "channel_4_attn_map" : transformer_4_attn_maps,
                "channel_1_weights" : self.inp_weight,
                "channel_2_weights" : self.conv1_weight,
                "channel_3_weights" : self.conv2_weight,
                "channel_4_weights" : self.conv3_weight,
                "final_weights" : self.final_weight
            }

            torch.save(all_weights, save_path)

        output = self.fc3(src[:, -1, :])

        return output, 0
