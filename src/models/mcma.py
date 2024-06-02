import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import init
import math

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
        x = F.layer_norm(x, normalized_shape=x.shape)
        x = self.dropout(x)
        return x

class MultiChannelMultiAttention(nn.Module):
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
        super(MultiChannelMultiAttention, self).__init__()

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

        # for name, param in self.encoder.named_parameters():
        #     if 'weight' in name and len(param.size()) > 1:
        #         init.xavier_normal_(param.data, mode='fan_in', nonlinearity='relu')
        
        # for name, param in self.encoder2.named_parameters():
        #     if 'weight' in name and len(param.size()) > 1:
        #         init.xavier_normal_(param.data, mode='fan_in', nonlinearity='relu')

        # for name, param in self.encoder3.named_parameters():
        #     if 'weight' in name and len(param.size()) > 1:
        #         init.xavier_normal_(param.data, mode='fan_in', nonlinearity='relu')

        # for name, param in self.encoder4.named_parameters():
        #     if 'weight' in name and len(param.size()) > 1:
        #         init.xavier_normal_(param.data, mode='fan_in', nonlinearity='relu')

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.dropout6 = nn.Dropout(dropout)
        self.dropout7 = nn.Dropout(dropout)
        # self.dropout8 = nn.Dropout(dropout)
        # self.dropout9 = nn.Dropout(dropout)
        # self.dropout10 = nn.Dropout(dropout)

        # self.weights = nn.Parameter(torch.ones(3))
        self.inp_weight = nn.Parameter(torch.rand(1))
        self.conv1_weight = nn.Parameter(torch.rand(1))
        self.conv2_weight = nn.Parameter(torch.rand(1))
        self.final_weight = nn.Parameter(torch.rand(4))
        self.conv3_weight = nn.Parameter(torch.rand(1))
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
        inp = src
        inp_pos = self.pe(inp)
        # print(inp_pos)
        if not save_path:
            inp = self.encoder(inp_pos, mask) * self.inp_weight + inp * (1 - self.inp_weight)
            # print(inp)
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
        # conved1 = F.relu(conved1)
        # conved1 = torch.transpose(conved1, 1, 2)
        # conved1 = self.bn2(conved1).transpose(1, 2)
        conved1_pos = self.pe2(conved1)
        # print(conved1_pos)
        if not save_path:
            conved1 = self.encoder2(conved1_pos, mask) * self.conv1_weight + conved1 * (1 - self.conv1_weight)
            # print(conved1)
            # exit()
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
        # conved2 = F.relu(conved2)
        # conved2 = torch.transpose(conved2, 1, 2)
        # conved2 = self.bn5(conved2).transpose(1, 2)
        conved2_pos = self.pe3(conved2)
        # conved2_pos = self.dropout(conved2_pos)
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
        # conved3 = F.relu(conved3)
        # conved3 = torch.transpose(conved3, 1, 2)
        # conved3 = self.bn10(conved3).transpose(1, 2)
        conved3_pos = self.pe4(conved3)
        # conved3_pos = self.dropout(conved3_pos)
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