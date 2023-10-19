import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Union, List

class jf_ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class jf_MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = jf_ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class jf_PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class jf_EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(jf_EncoderLayer, self).__init__()
        self.slf_attn = jf_MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = jf_PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class jf_SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(jf_SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        # pdb.set_trace()
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class model_jf_JointTransformer(nn.Module):
    """
    The joint transformer model that predicts 3D poses from 2D poses.
    Arguments
    ---------
    num_joints_in: int
        Number of joints per pose. Default=16 for human36m.
    n_features: Tuple[int]
        Number of features per joint. Tuple for input and output features. Default=(2, 3) for 2D->3D poses.
    n_layer: int
        Number of transformer encoder layers. Default=4.
    d_model: int
        Size of the hidden dimension per transformer encoder. Default=128.
    d_inner: int
        Size of the hidden dimension within the feed forward module inside the encoder transformer layers. Default=512.
    n_head: int
        Number of multi-head attention head. Default=8.
    d_k: int
        Size of the keys within the multi-head attention modules. Default=64.
    d_v: int
        Size of the values within the multi-head attention modules. Default=64.
    encoder_dropout: float
        Dropout probability within the transformer encoder layers. Default=0.0.
    pred_dropout: float
        Dropout probability for the prediction layers. Default=0.2.
    intermediate: bool
        Set to True for intermediate supervision. In this case the output pose is predicted
        after each transformer encoder layer and returned in a list. Default=False.
    spatial_encoding: bool
        Set to True for spatial encoding of the input poses instead of the default positional encoding. Default=False.
    embedding_type: str
        Type of layer to use to embed the input coordinates to the hidden dimension. Default='conv'.
    adj: torch.Tensor
        Adjacency matrix for the skeleton. Only needed if embedding_type='graph'. Default=None.
    error_prediction: bool
        Set to True to predict the error in the output prediction for each joint. Default=True.
    Methods
    -------
    forward(torch.Tensor, torch.Tensor, bool): Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor, Union[List[torch.Tensor], torch.Tensor], List[torch.Tensor]]
        Forward pass through the module. Given an input pose and optional image features, will predict the output pose.
        Returns the predicted pose, the last hidden state of the transformer encoders and the predicted error.
    """

    def __init__(self, num_joints_in: int = 133, n_features: Tuple[int] = (2, 3), n_layers: int = 4, d_model: int = 128,
                 d_inner: int = 512, n_head: int = 8, d_k: int = 64, d_v: int = 64, encoder_dropout: float = 0.0,
                 pred_dropout: float = 0.2,
                 intermediate: bool = False, spatial_encoding: bool = False, embedding_type: str = 'conv',
                 adj: torch.Tensor = None,
                 error_prediction: bool = True) -> None:

        """
        Initialize the network.
        """

        super(model_jf_JointTransformer, self).__init__()

        self.intermediate = intermediate
        self.error_prediction = error_prediction
        self.spatial_encoding = spatial_encoding
        assert embedding_type in ['conv', 'linear',
                                  'graph'], 'The chosen embedding type \'{}\' is not supported.'.format(embedding_type)
        self.embedding_type = embedding_type

        # Expand from x,y input coordinates to the size of the hidden dimension.
        if embedding_type == 'conv':
            self.embedding = nn.Conv1d(n_features[0], d_model, 1)
            self.embedding_forward = lambda x: self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif embedding_type == 'linear':
            self.embedding = nn.Linear(n_features[0] * num_joints_in, d_model * num_joints_in)
            self.embedding_forward = lambda x: self.embedding(x.reshape(x.size(0), -1)).view(x.size(0), x.size(1), -1)
        elif embedding_type == 'graph':
            self.embedding = jf_SemGraphConv(n_features[0], d_model, adj)
            self.embedding_forward = lambda x: self.embedding(x)
        else:
            raise NotImplementedError

        # Positional encoding.
        if self.spatial_encoding:
            self.position_enc = nn.Parameter(torch.zeros(1, num_joints_in, d_model))
        self.dropout = nn.Dropout(p=encoder_dropout)

        # Stacked encoders.
        self.layer_stack = nn.ModuleList([
            jf_EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=encoder_dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # Output prediction layers.
        if self.intermediate:
            self.intermediate_pred = nn.ModuleList([nn.Sequential(nn.LayerNorm(num_joints_in * d_model),
                                                                  nn.Dropout(p=pred_dropout),
                                                                  nn.Linear(num_joints_in * d_model,
                                                                            num_joints_in * n_features[1])) for _ in
                                                    range(n_layers)])
            self.intermediate_enc = nn.ModuleList(
                [nn.Linear(num_joints_in * n_features[1], num_joints_in * d_model) for _ in range(n_layers)])
            if self.error_prediction:
                self.intermediate_error = nn.ModuleList([nn.Sequential(nn.LayerNorm(num_joints_in * d_model),
                                                                       nn.Dropout(p=pred_dropout),
                                                                       nn.Linear(num_joints_in * d_model,
                                                                                 num_joints_in * n_features[1])) for _
                                                         in range(n_layers)])
        else:
            self.decoder = nn.Sequential(
                nn.LayerNorm(num_joints_in * d_model),
                nn.Dropout(p=pred_dropout),
                nn.Linear(num_joints_in * d_model, num_joints_in * n_features[1])
            )
            if self.error_prediction:
                self.error_decoder = nn.Sequential(
                    nn.LayerNorm(num_joints_in * d_model),
                    nn.Dropout(p=pred_dropout),
                    nn.Linear(num_joints_in * d_model, num_joints_in * n_features[1])
                )

        # Initialize layers with xavier.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, image: torch.Tensor = None, return_attns: bool = False) -> Tuple[
        Union[List[torch.Tensor], torch.Tensor], torch.Tensor, Union[List[torch.Tensor], torch.Tensor], List[
            torch.Tensor]]:
        """
        Forward pass through the network. Input is a sequence of 2D joints belonging to one pose.

        Parameters
        ----------
        src: torch.Tensor
            Tensor of 2D joints. [B, J, 2], where J is the number of joints.
        image: torch.Tensor
            Tensor of cropped image features around each joint. [B, J, H, W].
            This tensor is not currently used.
        return_attns: bool
            Set to True if the self attention tensors should be returned.
        Returns
        -------
        out: Union[List[torch.Tensor], torch.Tensor]
            The predicted 3D pose for each joint. If intermediate is true, this is a list of predicted 3D poses for each encoder layer.
            The shape is [B, J, 3]. In case a list is returned, they are ordered from first encoder layer to last encoder layer.
        enc_output: torch.Tensor
            The final hidden state that is the output of the last encoder layer. These features can be further used for sequence prediction.
            The shape is [B, J, d_model].
        error: Union[List[torch.Tensor], torch.Tensor]
            The predicted error of the 3D pose for each joint. If intermediate is true, this is a list of predicted errors for each encoder layer.
            The shape is [B, J, 3]. In case a list is returned, they are ordered from first encoder layer to last encoder layer.
        return_attns: List[torch.Tensor]
            Optional attention maps for every transformer encoder in the stack.
        """

        b, j, _ = src.shape  # Batch, Number of joints, Number of features per joint
        intermediate_list = []
        error_list = []
        enc_slf_attn_list = []

        # Expand dimensions.
        src = self.embedding_forward(src)

        # Positional encoding.
        if self.spatial_encoding:
            src += self.position_enc

        enc_output = self.dropout(src)
        enc_output = self.layer_norm(enc_output)

        # Stack of encoders.
        for i, enc_layer in enumerate(self.layer_stack):
            residual = enc_output
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=None)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

            if self.intermediate:
                pred = self.intermediate_pred[i](enc_output.clone().view(b, -1))
                res = self.intermediate_enc[i](pred).view(b, j, -1)
                if self.error_prediction:
                    error = self.intermediate_error[i](enc_output.clone().view(b, -1))
                    error_list += [error]

                enc_output += res
                intermediate_list += [pred]

            enc_output += residual
            enc_output = F.relu(enc_output)

        # Output either predictions for each encoder, or one prediction at the end. Always also output the last encoder output.
        if self.intermediate:
            out = [out.view(b, j, -1) for out in intermediate_list]
            error = [e.view(b, j, -1) for e in error_list]
        else:
            out = self.decoder(enc_output.view(b, -1)).view(b, j, -1)
            error = self.error_decoder(enc_output.view(b, -1)).view(b, j, -1) if self.error_prediction else None

        if return_attns:
            return out, enc_output, error, enc_slf_attn_list
        else:
            return out, enc_output,

class model_jf_ErrorRefinement(nn.Module):
    """
    The error refinement network that refines a predicted 3D pose using the initial pose and the predicted error.
    Arguments
    ---------
    num_joints_in: int
        Number of joints per pose. Default=16 for human36m.
    n_features: Tuple[int]
        Number of features per joint. Tuple for input and output features. Default=(2, 3) for 2D->3D poses.
    n_layer: int
        Number of transformer encoder layers. Default=4.
    d_model: int
        Size of the hidden dimension per transformer encoder. Default=128.
    d_inner: int
        Size of the hidden dimension within the feed forward module inside the encoder transformer layers. Default=512.
    n_head: int
        Number of multi-head attention head. Default=8.
    d_k: int
        Size of the keys within the multi-head attention modules. Default=64.
    d_v: int
        Size of the values within the multi-head attention modules. Default=64.
    encoder_dropout: float
        Dropout probability within the transformer encoder layers. Default=0.0.
    pred_dropout: float
        Dropout probability for the prediction layers. Default=0.2.
    intermediate: bool
        Set to True for intermediate supervision. In this case the output pose is predicted
        after each transformer encoder layer and returned in a list. Default=False.
    spatial_encoding: bool
        Set to True for spatial encoding of the input poses instead of the default positional encoding. Default=False.
    Methods
    -------
    forward(torch.Tensor): Union[List[torch.Tensor], torch.Tensor]
        Forward pass through the module. Given an input pose, predicted pose and predicted error will refine the predicted pose.
    """

    def __init__(self, num_joints_in: int = 133, n_features: Tuple[int] = (2, 3), n_layers: int = 4, d_model: int = 128,
                 d_inner: int = 512, n_head: int = 8, d_k: int = 64, d_v: int = 64, encoder_dropout: float = 0.0,
                 pred_dropout: float = 0.2,
                 intermediate: bool = False, spatial_encoding: bool = False) -> None:

        """
        Initialize the network.
        """

        super(model_jf_ErrorRefinement, self).__init__()

        self.spatial_encoding = spatial_encoding
        self.intermediate = intermediate

        self.embedding = nn.Conv1d(n_features[0] + 2 * n_features[1], d_model,
                                   1)  # Input features + predicted features + predicted error -> hidden dimension.
        self.embedding_forward = lambda x: self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Positional encoding.
        if self.spatial_encoding:
            self.position_enc = nn.Parameter(torch.zeros(1, num_joints_in, d_model))
        self.dropout = nn.Dropout(p=encoder_dropout)

        # Stacked encoders.
        self.layer_stack = nn.ModuleList([
            jf_EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=encoder_dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # Output prediction.
        if self.intermediate:
            self.intermediate_pred = nn.ModuleList([nn.Sequential(nn.LayerNorm(num_joints_in * d_model),
                                                                  nn.Dropout(p=pred_dropout),
                                                                  nn.Linear(num_joints_in * d_model,
                                                                            num_joints_in * n_features[1])) for _ in
                                                    range(n_layers)])
            self.intermediate_enc = nn.ModuleList(
                [nn.Linear(num_joints_in * n_features[1], num_joints_in * d_model) for _ in range(n_layers)])
        else:
            self.decoder = nn.Sequential(
                nn.LayerNorm(num_joints_in * d_model),
                nn.Dropout(p=pred_dropout),
                nn.Linear(num_joints_in * d_model, num_joints_in * n_features[1])
            )

        # Initialize layers with xavier.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through the network. Input is a sequence of 2D joints, predicted 3D joints and predicted error belonging to one pose.

        Parameters
        ----------
        src: torch.Tensor
            Tensor of 2D joints, predicted 3D joints and predicted error. [B, J, 2+3+3], where J is the number of joints.
        Returns
        -------
        out: Union[List[torch.Tensor], torch.Tensor]
            The predicted 3D pose for each joint. If intermediate is true, this is a list of predicted 3D poses for each encoder layer.
            The shape is [B, J, 3]. In case a list is returned, they are ordered from first encoder layer to last encoder layer.
        """

        b, j, _ = src.shape  # Batch, Number of joints, Number of features per joint
        intermediate_list = []

        # Expand dimensions.
        src = self.embedding_forward(src)

        # Positional encoding.
        if self.spatial_encoding:
            src += self.position_enc

        enc_output = self.dropout(src)
        enc_output = self.layer_norm(enc_output)

        # Stack of encoders.
        for i, enc_layer in enumerate(self.layer_stack):
            residual = enc_output
            enc_output, _ = enc_layer(enc_output, slf_attn_mask=None)

            if self.intermediate:
                pred = self.intermediate_pred[i](enc_output.clone().view(b, -1))
                res = self.intermediate_enc[i](pred).view(b, j, -1)

                enc_output += res
                intermediate_list += [pred]

            enc_output += residual
            enc_output = F.relu(enc_output)

        # Final prediction.
        if self.intermediate:
            out = [out.view(b, j, -1) for out in intermediate_list]
        else:
            out = self.decoder(enc_output.view(b, -1)).view(b, j, -1)

        return out



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_coarse = model_jf_JointTransformer(intermediate=True).to(device)
    net_fine = model_jf_ErrorRefinement(intermediate=True).to(device)
    if os.path.exists('./net_coarse.pth'):
        net_coarse.load_state_dict(torch.load('./net_coarse.pth', map_location=device))
        print('load pretrained jointformer coarse')
    if os.path.exists('./net_fine.pth'):
        net_fine.load_state_dict(torch.load('./net_fine.pth', map_location=device))
        print('load pretrained jointformer fine')

    batch_size = 2
    input_2d = torch.rand(batch_size, 133, 2).to(device)  # should be in meter. if you use our h3wb dataset, check if they are currently in mm (>1000) or m (<10)
    confidence = (torch.sum(torch.square(input_2d), dim=-1, keepdim=True) > 0.001) * 1.0  # 0.0 = missing joint

    output_3d, _, pred_error, _ = net_coarse(input_2d, return_attns=True)
    new_input = torch.cat((input_2d, output_3d[-1], pred_error[-1]), dim=-1)
    output_3d = net_fine(new_input)
    output_3d = output_3d[-1]  # also in meter. if you want to evaluate on h3wb, please x1000
    print(output_3d.shape)
