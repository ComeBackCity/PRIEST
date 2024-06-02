from mimetypes import init
import torch
import torch.nn as nn 
import math
from torch import nn, Tensor
from typing import List
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int):
        super(MLP, self).__init__()
        layers = []

        for idx, layer in enumerate(hidden_layers):
            if idx == 0:
                nn_layer = nn.Linear(input_dim, layer)
            else:
                nn_layer = nn.Linear(hidden_layers[idx-1], layer)

            # torch.nn.init.xavier_uniform_(nn_layer.weight)
            # torch.nn.init.zeros_(nn_layer.bias)
            layers.append(nn_layer)

        layers.append(nn.ReLU())
        final_layer = nn.Linear(hidden_layers[-1], output_dim)
        # torch.nn.init.xavier_uniform_(final_layer.weight)
        # torch.nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        self.mlp = nn.Sequential(*layers)


    def forward(self, x):
        return self.mlp(x)

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
        dropout: float=0.5, 
        max_seq_len: int=9, 
        d_model: int=1,
        batch_first: bool=False
        ):

        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """

        super().__init__()
        # print('In PE init')

        self.d_model = d_model
        # print(self.d_model)
        
        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first
        # print(f'In pe {self.batch_first}')

        self.x_dim = 1 if batch_first else 0

        position = torch.arange(max_seq_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_seq_len, 1, d_model)
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor, debug:bool=False) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """
        a  = self.pe[:x.size(self.x_dim)]
        a = a.reshape((-1, self.d_model))
        
        x = x + a

        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):

    """
    This class implements a transformer model that can be used for times series
    forecasting. This time series transformer model is based on the paper by
    Wu et al (2020) [1]. The paper will be referred to as "the paper".
    A detailed description of the code can be found in my article here:
    https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
    In cases where the paper does not specify what value was used for a specific
    configuration/hyperparameter, this class uses the values from Vaswani et al
    (2017) [2] or from PyTorch source code.
    Unlike the paper, this class assumes that input layers, positional encoding 
    layers and linear mapping layers are separate from the encoder and decoder, 
    i.e. the encoder and decoder only do what is depicted as their sub-layers 
    in the paper. For practical purposes, this assumption does not make a 
    difference - it merely means that the linear and positional encoding layers
    are implemented inside the present class and not inside the 
    Encoder() and Decoder() classes.
    [1] Wu, N., Green, B., Ben, X., O'banion, S. (2020). 
    'Deep Transformer Models for Time Series Forecasting: 
    The Influenza Prevalence Case'. 
    arXiv:2001.08317 [cs, stat] [Preprint]. 
    Available at: http://arxiv.org/abs/2001.08317 (Accessed: 9 March 2022).
    [2] Vaswani, A. et al. (2017) 
    'Attention Is All You Need'.
    arXiv:1706.03762 [cs] [Preprint]. 
    Available at: http://arxiv.org/abs/1706.03762 (Accessed: 9 March 2022).
    """

    def __init__(self, 
        input_size: int,
        enc_seq_len: int,
        window_size: List[int],
        # dec_seq_len: int,
        batch_first: bool = True,
        out_seq_len: int=58,
        dim_val: int=128,  
        n_encoder_layers: int=8,
        n_decoder_layers: int=4,
        n_heads: int=16,
        dropout_encoder: float=0.1, 
        dropout_decoder: float=0.2,
        dropout_pos_enc: float=0.2,
        dim_feedforward_encoder: int=512,
        dim_feedforward_decoder: int=2048,
        num_predicted_features: int=1,
        output_dim: int=2
        ): 

        """
        Args:
            input_size: int, number of input variables. 1 if univariate.
            dec_seq_len: int, the length of the input sequence fed to the decoder
            dim_val: int, aka d_model. All sub-layers in the model produce 
                     outputs of dimension dim_val
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_decoder_layers: int, number of stacked encoder layers in the decoder
            n_heads: int, the number of attention heads (aka parallel attention layers)
            dropout_encoder: float, the dropout rate of the encoder
            dropout_decoder: float, the dropout rate of the decoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            dim_feedforward_encoder: int, number of neurons in the linear layer 
                                     of the encoder
            dim_feedforward_decoder: int, number of neurons in the linear layer 
                                     of the decoder
            num_predicted_features: int, the number of features you want to predict.
                                    Most of the time, this will be 1 because we're
                                    only forecasting FCR-N prices in DK2, but in
                                    we wanted to also predict FCR-D with the same
                                    model, num_predicted_features should be 2.
        """

        super().__init__() 

        # self.dec_seq_len = dec_seq_len

        #print("input_size is: {}".format(input_size))
        #print("dim_val is: {}".format(dim_val))

        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(
            in_features=input_size, 
            out_features=dim_val 
        )

        # self.decoder_input_layer = nn.Linear(
        #     in_features=num_predicted_features,
        #     out_features=dim_val
        #     )  
        
        self.linear_mapping = nn.Linear(
            in_features=dim_val, 
            out_features=num_predicted_features
            )

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            max_seq_len=enc_seq_len,
            dropout=dropout_pos_enc,
            batch_first = batch_first
        )

        # The encoder layer used in the paper is identical to the one used by
        # Vaswani et al (2017) on which the PyTorch module is based.
        # if debug:
        #   print(f'In transformer {batch_first}')
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first,
            activation='gelu',
            layer_norm_eps=1e-4,
            norm_first=False
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers, 
            norm=None
        )

        self.out = nn.Linear(
            in_features=dim_val * enc_seq_len,
            out_features=output_dim
        )

        self.out_fnn = MLP(
          input_dim = dim_val * enc_seq_len,
          hidden_layers = [ 512, 256, 32, 8],
          output_dim = 2
        )

        n_filters = 10
        vector_size = dim_val
        filter_sizes = window_size

        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, 
                                  out_channels = n_filters, 
                                  kernel_size = (fs, vector_size)) 
                                    for fs in filter_sizes])

        self.linear_final = nn.Linear(len(filter_sizes) \
                      * n_filters, output_dim)

        self.dropout_final = nn.Dropout(0.2)

        self.attention_weights = nn.Parameter(torch.ones(3))
        self.attention_weights.requires_grad = True


        # self.convs = nn.ModuleList([
        #     nn.Conv2d(1, NUM_FILTERS, [window_size, EMBEDDING_SIZE], padding=(window_size - 1, 0))
        #     for window_size in window_sizes
        # ])

        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=dim_val,
        #     nhead=n_heads,
        #     dim_feedforward=dim_feedforward_decoder,
        #     dropout=dropout_decoder,
        #     batch_first=batch_first
        #     )

        # Stack the decoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerDecoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        # self.decoder = nn.TransformerDecoder(
        #     decoder_layer=decoder_layer,
        #     num_layers=n_decoder_layers, 
        #     norm=None
        #     )

    def forward(self, src: Tensor, tgt: Tensor=None, src_mask: Tensor=None, 
                tgt_mask: Tensor=None, debug:bool = False) -> Tensor:
        """
        Returns a tensor of shape:
        [target_sequence_length, batch_size, num_predicted_features]
        
        Args:
            src: the encoder's output sequence. Shape: (S,E) for unbatched input, 
                 (S, N, E) if batch_first=False or (N, S, E) if 
                 batch_first=True, where S is the source sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)
            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input, 
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if 
                 batch_first=True, where T is the target sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)
            src_mask: the mask for the src sequence to prevent the model from 
                      using data points from the target sequence
            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence
        """

        if debug:
          print("From model.forward(): Size of src as given to forward(): {}".format(src.size()))
        # print("From model.forward(): tgt size = {}".format(tgt.size()))

        # Pass throguh the input layer right before the encoder
        if debug:
          print(src.shape)
        src = self.encoder_input_layer(src) # src shape: [batch_size, src length, dim_val] regardless of number of input features
        # print(self.encoder_input_layer.state_dict()['weight'].shape)
        # print(self.encoder_input_layer.state_dict()['bias'].shape)
        if debug:
          print("From model.forward(): Size of src after input layer: {}".format(src.size()))
        # exit()
        # Pass through the positional encoding layer
        src = self.positional_encoding_layer(src, debug) # src shape: [batch_size, src length, dim_val] regardless of number of input features
        if debug:
          print("From model.forward(): Size of src after pos_enc layer: {}".format(src.size()))
        # exit()

        # Pass through all the stacked encoder layers in the encoder
        # Masking is only needed in the encoder if input sequences are padded
        # which they are not in this time series use case, because all my
        # input sequences are naturally of the same length. 
        # (https://github.com/huggingface/transformers/issues/4083)

        # for layer in self.encoder.layers:
        #     print('Before ')

        # for idx, layer in enumerate(self.encoder.layers):
        #     print(f'\nBefore layer {idx}')
        #     print(src.shape)
        #     src = layer(src)
        #     # print(layer.state_dict().keys())
        #     print(layer.state_dict()['self_attn.in_proj_weight'].shape)
        #     print(layer.state_dict()['self_attn.out_proj.weight'].shape)
        #     print(f'after layer {idx}')
        #     print(src.shape)

        src = self.encoder( # src shape: [batch_size, enc_seq_len, dim_val]
            src=src
        )

        src = src.unsqueeze(1)

        attention_outputs = [self.attention(src)[0], self.additive_attention(src, src, src)[0], \
            self.scaled_dot_product_attention(src, src, src)[0]]

        attention_weights = self.attention_weights / self.attention_weights.sum()
        weighted_attention_outputs = [attention_outputs[i] * attention_weights[i] for i in range(3)]
        attention_output = sum(weighted_attention_outputs)

        src = src + attention_output
    
        # src = self.attention(src)[0]

        conved = [F.relu(conv(src)).squeeze(3) 
                  for conv in self.convs]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) 
                  for conv in conved]

        cat = self.dropout_final(torch.cat(pooled, dim = 1))
        output = self.linear_final(cat)


        # src_shape = list(src.shape)
        # if debug:
        #   print(src_shape)
        # exit()

        # flattened_tensor = src.reshape(src_shape[0], -1)
        # if debug:
        #   print(flattened_tensor.shape)
        # output = self.out(flattened_tensor)
        # output = self.out_fnn(flattened_tensor)
        # output = torch.softmax(output, dim=1)


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


    def extract_self_attn_maps(self, src: Tensor, device):

        src = self.encoder_input_layer(src) # src shape: [batch_size, src length, dim_val] regardless of number of input features
        src = self.positional_encoding_layer(src) # src shape: [batch_size, src length, dim_val] regardless of number of input features

        transformer_encoder = self.encoder
        x = src

        attention_maps = []
        num_layers = transformer_encoder.num_layers
        num_heads = transformer_encoder.layers[0].self_attn.num_heads
        norm_first = transformer_encoder.layers[0].norm_first

        seq_len = src.shape[1]
        batch_size = src.shape[0]
        src_mask = torch.zeros((seq_len, seq_len)).bool().to(device)
        src_key_padding_mask = torch.zeros((batch_size, seq_len)).bool().to(device)


        with torch.no_grad():
            for i in range(num_layers):
                # compute attention of layer i
                h = x.clone()
                if norm_first:
                    h = transformer_encoder.layers[i].norm1(h)
                attn = transformer_encoder.layers[i].self_attn(h, h, h,attn_mask=src_mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
                print(f'here {i}')
                print(transformer_encoder.layers[i].self_attn._qkv_same_embed_dim)
                print(transformer_encoder.layers[i].self_attn.q_proj_weight)
                print(transformer_encoder.layers[i].self_attn.k_proj_weight)
                print(transformer_encoder.layers[i].self_attn.v_proj_weight)
                attention_maps.append(attn)
                # forward of layer i
                x = transformer_encoder.layers[i](x,src_mask=src_mask,src_key_padding_mask=src_key_padding_mask)
        return attention_maps