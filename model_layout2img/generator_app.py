import torch
import torch.nn as nn
import torch.nn.functional as F
from models2.bilinear import crop_bbox_batch
from models2.transformer import transformer_encoder
import numpy as np


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image
    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j
    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    f_g = f_g.cuda()

    batch_size = f_g.size(0)

    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=-1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return(embedding)


def box_attention(query, key, value, box_relation_embds_matrix, mask=None, dropout=None):
    '''
    Compute 'Scaled Dot Product Attention as in paper Relation Networks for Object Detection'.
    Follow the implementation in https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1026-L1055
    '''
    if mask is not None:
        mask = mask.unsqueeze(1).expand(mask.size(0), query.size(1), mask.size(1))

    N = value.size()[:2]
    dim_k = key.size(-1)
    dim_g = box_relation_embds_matrix.size()[-1]

    # print(mask.shape)

    w_q = query
    w_k = key.transpose(-2, -1)
    w_v = value
    # print(w_q.shape)
    # print(w_k.shape)
    # attention weights
    scaled_dot = torch.matmul(w_q, w_k)
    # print(scaled_dot.shape)
    scaled_dot = scaled_dot / np.sqrt(dim_k)
    if mask is not None:
        scaled_dot = scaled_dot.masked_fill(mask == 0, -1e9)

    #w_g = box_relation_embds_matrix.view(N,N)
    w_g = box_relation_embds_matrix.squeeze(1)
    w_g = w_g.masked_fill(mask == 0, 0)
    w_a = scaled_dot
    #w_a = scaled_dot.view(N,N)
    # print(w_g.shape)
    # print(w_a.shape)

    # multiplying log of geometric weights by feature weights
    # print(w_a)
    # print(w_g)
    w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
    #w_mn = w_a
    w_mn = torch.nn.Softmax(dim=-1)(w_mn)
    if dropout is not None:
        w_mn = dropout(w_mn)

    output = torch.matmul(w_mn, w_v)

    return output, w_mn


class BoxMultiHeadedAttention(nn.Module):
    '''
    Self-attention layer with relative position weights.
    Following the paper "Relation Networks for Object Detection" in https://arxiv.org/pdf/1711.11575.pdf
    '''

    def __init__(self, h, d_model, trignometric_embedding=True, legacy_extra_skip=False, dropout=0.1):
        "Take in model size and number of heads."
        super(BoxMultiHeadedAttention, self).__init__()

        assert d_model % h == 0
        self.trignometric_embedding = trignometric_embedding
        self.legacy_extra_skip = legacy_extra_skip

        # We assume d_v always equals d_k
        self.h = h
        self.d_k = d_model // h
        self.d_v = d_model // h
        if self.trignometric_embedding:
            self.dim_g = 64
        else:
            self.dim_g = 4
        geo_feature_dim = self.dim_g

        # matrices W_q, W_k, W_v, and one last projection layer
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.WGs = clones(nn.Linear(geo_feature_dim, 1, bias=True), h)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm0 = nn.LayerNorm(d_model)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_query, input_key, input_value, input_box, mask=None):
        "Implements Figure 2 of Relation Network for Object Detection"
        # if mask is not None:
        # Same mask applied to all h heads.
        #    mask = mask.unsqueeze(1)

        d_k, d_v, n_head = self.d_k, self.d_v, self.h
        sz_b0, len_q, _ = input_query.size()
        sz_b, len_k, _ = input_key.size()
        sz_b, len_v, _ = input_value.size()

        nbatches = input_query.size(0)

        residual = input_query

        # tensor with entries R_mn given by a hardcoded embedding of the relative position between bbox_m and bbox_n
        relative_geometry_embeddings = BoxRelationalEmbedding(input_box, trignometric_embedding=self.trignometric_embedding)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, self.dim_g)
        # print(flatten_relative_geometry_embeddings.shape)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k)
             for l, x in zip(self.linears, (input_query, input_key, input_value))]

        q = query.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = key.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = value.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.box_attn = box_attention(q, k, v, relative_geometry_weights, mask=mask,
                                         dropout=self.dropout)
        # print(x.shape)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        # An extra internal skip connection is added. This is only
        # kept here for compatibility with some legacy models. In
        # general, there is no advantage in using it, as there is
        # already an outer skip connection surrounding this layer.
        # if self.legacy_extra_skip:
        #    x = input_value + x
        # print(residual.shape)

        output = self.layer_norm0(x + residual)
        new_residual = output

        output = self.dropout(self.linears[-1](output))
        output = self.layer_norm(output + new_residual)

        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask):

        # if mask is not None:
        #    mask = mask.unsqueeze(1)

        attn = torch.bmm(q, k.transpose(1, 2))
        if mask is not None:
            mask = mask.unsqueeze(1).expand(mask.size(0), q.size(1), mask.size(1))
            # print(mask.shape)
            # print(attn.shape)
            attn = attn.masked_fill(mask == 0, -1e9)
        # print(attn.shape)
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm0 = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        # if mask is not None:
        #    mask = mask.unsqueeze(1).expand(mask.size(0),q.size(1),mask.size(1))
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b0, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b0, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b0, len_q, -1)  # b x lq x (n*dv)
        # print(output.shape)
        output = self.layer_norm0(output + residual)
        new_residual = output

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + new_residual)

        return output


def feature_vector_combine(z, mask):
    all_feat = []

    for i in range(z.size(0)):
        for j in range(z.size(1)):
            if mask[i, j] != 0:
                all_feat.append(z[i, j])

    return torch.stack(all_feat)


def feature_vector_split(z, obj_to_img):
    obj_num = []
    ref = 0
    count = 0
    for i in range(z.size(0)):
        if obj_to_img[i] == ref:
            count += 1
        else:
            obj_num.append(count)
            ref = obj_to_img[i]
            count = 1
    obj_num.append(count)
    L = max(obj_num)
    holder = torch.zeros(len(obj_num), L, z.size(1)).cuda()

    mask = torch.zeros(len(obj_num), L).cuda()
    start = 0
    for i in range(len(obj_num)):
        holder[i, :obj_num[i], :] = z[start:start + obj_num[i]]
        mask[i, :obj_num[i]] = 1
        start = start + obj_num[i]
    return holder, mask


def featuremap_composition(z, obj_to_img):
    obj_num = []
    ref = 0
    count = 0
    for i in range(z.size(0)):
        if obj_to_img[i] == ref:
            count += 1
        else:
            obj_num.append(count)
            ref = obj_to_img[i]
            count = 1
    obj_num.append(count)
    L = max(obj_num)
    holder = torch.zeros(len(obj_num), z.size(1), 8, 8).cuda()
    #mask = torch.zeros(len(obj_num),L)
    start = 0
    for i in range(len(obj_num)):
        holder[i, :, :, :] = z[start:start + obj_num[i]].sum(0)
        #mask[i,:obj_num[i]] = 1
        start = start + obj_num[i]
    return holder  # ,mask


def get_z_random(batch_size, z_dim, random_type='gauss'):
    if random_type == 'uni':
        z = torch.rand(batch_size, z_dim) * 2.0 - 1.0
    elif random_type == 'gauss':
        z = torch.randn(batch_size, z_dim)
    return z


def transform_z_flat(batch_size, time_step, z_flat, obj_to_img):
    # restore z to batch with padding
    z = torch.zeros(batch_size, time_step, z_flat.size(1)).to(z_flat.device)
    for i in range(batch_size):
        idx = (obj_to_img.data == i).nonzero()
        if idx.dim() == 0:
            continue
        idx = idx.view(-1)
        n = idx.size(0)
        z[i, :n] = z_flat[idx]
    return z


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, device):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        if isinstance(hidden_dim, list):
            num_layers = len(hidden_dim)
        elif isinstance(hidden_dim, int):
            num_layers = 1

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0), device=input_tensor.device)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class LayoutConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True, return_all_layers=False):
        super(LayoutConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        if isinstance(hidden_dim, list) or isinstance(hidden_dim, tuple):
            num_layers = len(hidden_dim)
        elif isinstance(hidden_dim, int):
            num_layers = 1

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size, input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, obj_tensor, obj_to_img, hidden_state=None):
        """

        Parameters
        ----------
        obj_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        # split input_tensor into list according to obj_to_img
        O = obj_tensor.size(0)
        previous_img_id = 0

        layouts_list = []
        temp = []
        for i in range(O):
            current_img_id = obj_to_img[i]
            if current_img_id == previous_img_id:
                temp.append(obj_tensor[i])
            else:
                temp = torch.stack(temp, dim=0)
                temp = torch.unsqueeze(temp, 0)
                layouts_list.append(temp)
                temp = [obj_tensor[i]]
                previous_img_id = current_img_id
        # append last one
        temp = torch.stack(temp, dim=0)
        temp = torch.unsqueeze(temp, 0)
        layouts_list.append(temp)

        N = len(layouts_list)
        # print(N)
        all_layer_output_list, all_last_state_list = [], []
        for i in range(N):
            obj_tensor = layouts_list[i]
            # print(obj_tensor.shape)
            hidden_state = self._init_hidden(batch_size=obj_tensor.size(0), device=obj_tensor.device)

            layer_output_list = []
            last_state_list = []

            seq_len = obj_tensor.size(1)
            cur_layer_input = obj_tensor

            for layer_idx in range(self.num_layers):

                h, c = hidden_state[layer_idx]
                output_inner = []
                for t in range(seq_len):
                    h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                     cur_state=[h, c])
                    output_inner.append(h)

                layer_output = torch.stack(output_inner, dim=1)
                cur_layer_input = layer_output

                layer_output_list.append(layer_output)
                last_state_list.append([h, c])

            if not self.return_all_layers:
                layer_output_list = layer_output_list[-1:]
                last_state_list = last_state_list[-1:]

            all_layer_output_list.append(layer_output_list)
            all_last_state_list.append(last_state_list)

        # concate last output to form a tensor
        batch_output = []
        for i in range(N):
            batch_output.append(all_last_state_list[i][0][0])
        batch_output = torch.cat(batch_output, dim=0)

        return batch_output

    def _init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class CropEncoder(nn.Module):
    def __init__(self, conv_dim=64, z_dim=8, class_num=10):
        # default: (3, 32, 32) -> (256, 8, 8)
        super(CropEncoder, self).__init__()
        self.activation = nn.ReLU(inplace=True)

        # (3, 32, 32) -> (64, 32, 32)
        self.c1 = nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(conv_dim) if class_num == 0 else ConditionalBatchNorm2d(conv_dim, class_num)
        # (64, 32, 32) -> (128, 16, 16)
        self.c2 = nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(conv_dim * 2) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 2, class_num)
        # (128, 16, 16) -> (256, 8, 8)
        self.c3 = nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv_dim * 4) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 4, class_num)
        # (256, 8, 8) -> (512, 4, 4)
        self.c4 = nn.Conv2d(conv_dim * 4, conv_dim * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(conv_dim * 8) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 8, class_num)
        # (512, 4, 4) -> (1024, 2, 2)
        self.conv5 = nn.Conv2d(conv_dim * 8, conv_dim * 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(conv_dim * 16) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 16, class_num)
        # pool
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 256 -> 8
        self.fc_mu = nn.Linear(conv_dim * 16, z_dim)
        self.fc_logvar = nn.Linear(conv_dim * 16, z_dim)

    def forward(self, imgs, objs=None):
        x = imgs
        x = self.c1(x)
        x = self.bn1(x) if objs is None else self.bn1(x, objs)
        x = self.activation(x)
        x = self.c2(x)
        x = self.bn2(x) if objs is None else self.bn2(x, objs)
        x = self.activation(x)
        x = self.c3(x)
        x = self.bn3(x) if objs is None else self.bn3(x, objs)
        x = self.activation(x)
        x = self.c4(x)
        x = self.bn4(x) if objs is None else self.bn4(x, objs)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.bn5(x) if objs is None else self.bn5(x, objs)
        x = self.activation(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        std = logvar.mul(0.5).exp_()
        eps = get_z_random(std.size(0), std.size(1)).to(imgs.device)
        z = eps.mul(std).add_(mu)

        return z, mu, logvar


class LayoutEncoder(nn.Module):
    def __init__(self, conv_dim=64, z_dim=8, embedding_dim=64, class_num=10, resi_num=6, clstm_layers=3):
        super(LayoutEncoder, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.embedding = nn.Embedding(class_num, embedding_dim)
        self.transformer_encoder = transformer_encoder(3)
        self.context = MultiHeadAttention(1, embedding_dim + z_dim, embedding_dim + z_dim, embedding_dim + z_dim, dropout=0.0)
        if clstm_layers == 1:
            self.clstm = LayoutConvLSTM(8, 512, [64], (5, 5))
        elif clstm_layers == 2:
            self.clstm = LayoutConvLSTM(8, 512, [128, 64], (5, 5))
        elif clstm_layers == 3:
            self.clstm = LayoutConvLSTM(8, 512, [128, 64, 64], (5, 5))

        layers = []
        # Bottleneck layers.
        for i in range(resi_num):
            layers.append(ResidualBlock(dim_in=64, dim_out=64))
        self.residual = nn.Sequential(*layers)

        # (emb+z, 64, 64) -> (64, 64, 64)
        self.c1 = nn.Conv2d(embedding_dim + z_dim, conv_dim, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv_dim) if class_num == 0 else ConditionalBatchNorm2d(conv_dim, class_num)
        # (64, 64, 64) -> (128, 32, 32)
        self.c2 = nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(conv_dim * 2) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 2, class_num)
        # (128, 32, 32) -> (256, 16, 16)
        self.c3 = nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv_dim * 4) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 4, class_num)
        # (256, 16, 16) -> (512, 8, 8)
        self.c4 = nn.Conv2d(conv_dim * 4, conv_dim * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(conv_dim * 8) if class_num == 0 else ConditionalBatchNorm2d(conv_dim * 8, class_num)
        self.c5 = nn.Conv2d(conv_dim * 8, conv_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(conv_dim)

    def forward(self, objs, masks, obj_to_img, z):
        # prepare mask fm
        embeddings = self.embedding(objs)

        embeddings_z = torch.cat((embeddings, z), dim=1)
        # print(embeddings_z.shape)
        embeddings_z, fill_mask = feature_vector_split(embeddings_z, obj_to_img)
        # print(embeddings_z.shape)
        embeddings_z = self.context(embeddings_z, embeddings_z, embeddings_z, fill_mask)
        # print(embeddings_z.shape)
        embeddings_z = feature_vector_combine(embeddings_z, fill_mask)
        # print(embeddings_z.shape)
        h = embeddings_z.view(embeddings_z.size(0), embeddings_z.size(1), 1, 1) * masks
        # print(h.shape)
        # downsample layout
        h = self.c1(h)
        h = self.bn1(h, objs)
        h = self.activation(h)
        h = self.c2(h)
        h = self.bn2(h, objs)
        h = self.activation(h)
        h = self.c3(h)
        h = self.bn3(h, objs)
        h = self.activation(h)
        h = self.c4(h)
        h = self.bn4(h, objs)
        # print(h.shape)
        h_sum = featuremap_composition(h, obj_to_img)
        # print(h_sum.shape)
        h_sum = h_sum.permute(0, 2, 3, 1)
        h_sum = h_sum.contiguous().view(-1, 64, 512)
        # print(h_sum.shape)
        h_sum_refine = self.transformer_encoder(h_sum)
        # print(h_sum_refine.shape)
        h_sum_refine = h_sum_refine.permute(0, 2, 1).contiguous().view(-1, 512, 8, 8)
        # print(h_sum_refine.shape)
        h_sum_refine = self.c5(h_sum_refine)
        h_sum_refine = self.bn5(h_sum_refine)
        h_sum_refine = self.activation(h_sum_refine)
        # print(h_sum_refine.shape)

        # clstm fusion (O, 512, 8, 8) -> (n, 64, 8, 8)
        # replace this lstm with transformer
        #h = self.clstm(h, obj_to_img)
        # print(h.shape)
        # residual block
        h = self.residual(h_sum_refine)
        # print(h.shape)

        return h


class Decoder(nn.Module):
    def __init__(self, conv_dim=64):
        super(Decoder, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        # (64, 8, 8) -> (256, 8, 8)
        self.c0 = nn.Conv2d(conv_dim, conv_dim * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(conv_dim * 4)
        # (256, 8, 8) -> (256, 16, 16)
        self.dc1 = nn.ConvTranspose2d(conv_dim * 4, conv_dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv_dim * 4)
        # (256, 16, 16) -> (128, 32, 32)
        self.dc2 = nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(conv_dim * 2)
        # (128, 32, 32) -> (64, 64, 64)
        self.dc3 = nn.ConvTranspose2d(conv_dim * 2, conv_dim * 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv_dim * 1)
        # (64, 64, 64) -> (3, 64, 64)
        self.c4 = nn.Conv2d(conv_dim * 1, 3, kernel_size=7, stride=1, padding=3, bias=True)

    def forward(self, hidden):
        h = hidden
        h = self.c0(h)
        h = self.bn0(h)
        h = self.activation(h)
        h = self.dc1(h)
        h = self.bn1(h)
        h = self.activation(h)
        h = self.dc2(h)
        h = self.bn2(h)
        h = self.activation(h)
        h = self.dc3(h)
        h = self.bn3(h)
        h = self.activation(h)
        h = self.c4(h)
        return h


class Generator(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=64, z_dim=8, obj_size=32, clstm_layers=3):
        super(Generator, self).__init__()
        self.obj_size = obj_size
        # (3, 32, 32) -> (256, 4, 4) -> 8
        self.crop_encoder = CropEncoder(z_dim=z_dim, class_num=num_embeddings)
        self.layout_encoder = LayoutEncoder(z_dim=z_dim, embedding_dim=embedding_dim, class_num=num_embeddings, clstm_layers=clstm_layers)
        self.decoder = Decoder()
        # self.apply(weights_init)

    def forward(self, imgs, objs, boxes, masks, obj_to_img, z_rand):
        crops_input = crop_bbox_batch(imgs, boxes, obj_to_img, self.obj_size)
        z_rec, mu, logvar = self.crop_encoder(crops_input, objs)
        # print(z_rec.shape)
        # print(z_rand.shape)

        # (n, clstm_dim*2, 8, 8)
        h_rec = self.layout_encoder(objs, masks, obj_to_img, z_rec)
        h_rand = self.layout_encoder(objs, masks, obj_to_img, z_rand)

        img_rec = self.decoder(h_rec)
        img_rand = self.decoder(h_rand)

        crops_rand = crop_bbox_batch(img_rand, boxes, obj_to_img, self.obj_size)
        _, z_rand_rec, _ = self.crop_encoder(crops_rand, objs)

        crops_input_rec = crop_bbox_batch(img_rec, boxes, obj_to_img, self.obj_size)

        return crops_input, crops_input_rec, crops_rand, img_rec, img_rand, mu, logvar, z_rand_rec


class Generator_context(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=64, z_dim=8, obj_size=32, clstm_layers=3):
        super(Generator_context, self).__init__()
        self.obj_size = obj_size
        # (3, 32, 32) -> (256, 4, 4) -> 8

        self.crop_encoder = CropEncoder(z_dim=z_dim, class_num=num_embeddings)
        self.layout_encoder = LayoutEncoder(z_dim=z_dim, embedding_dim=embedding_dim, class_num=num_embeddings, clstm_layers=clstm_layers)
        self.decoder = Decoder()
        # self.apply(weights_init)

    def forward(self, imgs, objs, boxes, masks, obj_to_img, z_rand):
        crops_input = crop_bbox_batch(imgs, boxes, obj_to_img, self.obj_size)
        z_rec, mu, logvar = self.crop_encoder(crops_input, objs)
        # print(z_rec.shape)
        # print(z_rand.shape)

        # (n, clstm_dim*2, 8, 8)
        h_rec = self.layout_encoder(objs, masks, obj_to_img, z_rec)
        h_rand = self.layout_encoder(objs, masks, obj_to_img, z_rand)

        img_rec = self.decoder(h_rec)
        img_rand = self.decoder(h_rand)

        crops_rand = crop_bbox_batch(img_rand, boxes, obj_to_img, self.obj_size)
        _, z_rand_rec, _ = self.crop_encoder(crops_rand, objs)

        crops_input_rec = crop_bbox_batch(img_rec, boxes, obj_to_img, self.obj_size)

        return crops_input, crops_input_rec, crops_rand, img_rec, img_rand, mu, logvar, z_rand_rec


if __name__ == '__main__':
    from data.vg_custom_mask import get_dataloader

    device = torch.device('cuda:0')
    z_dim = 8
    batch_size = 4

    train_loader, _ = get_dataloader(batch_size=batch_size)
    vocab_num = train_loader.dataset.num_objects

    # test CropEncoder
    # model = CropEncoder(class_num=vocab_num).to(device)
    #
    # for batch in train_loader:
    #     imgs, objs, boxes, masks, obj_to_img = batch
    #     imgs, objs, boxes, masks, obj_to_img = imgs.to(device), objs.to(device), boxes.to(device), masks.to(device), obj_to_img.to(device)
    #
    #     crops = crop_bbox_batch(imgs, boxes, obj_to_img, 32)
    #     outputs = model(crops, objs)
    #
    #     for output in outputs:
    #         print(output.shape)
    #
    #     break

    # test MaskEncoder
    # model = LayoutEncoder(class_num=vocab_num, z_dim=z_dim).to(device)
    #
    # for batch in train_loader:
    #     imgs, objs, boxes, masks, obj_to_img = batch
    #     z = torch.randn(objs.size(0), z_dim).to(device)
    #     imgs, objs, boxes, masks, obj_to_img = imgs.to(device), objs.to(device), boxes.to(device), masks.to(device), obj_to_img.to(device)
    #
    #     crops = crop_bbox_batch(imgs, boxes, obj_to_img, 32)
    #     outputs = model(objs, masks, obj_to_img, z)
    #
    #     for output in outputs:
    #         print(output.shape)
    #
    #     break

    # test Generator
    model = Generator(num_embeddings=vocab_num, z_dim=z_dim).to(device)

    for i, batch in enumerate(train_loader):
        print(i)
        imgs, objs, boxes, masks, obj_to_img = batch
        z = torch.randn(objs.size(0), z_dim).to(device)
        imgs, objs, boxes, masks, obj_to_img = imgs.to(device), objs.to(device), boxes.to(device), masks.to(device), obj_to_img.to(device)

        outputs = model(imgs, objs, boxes, masks, obj_to_img, z)

        for output in outputs:
            print(output.shape)

        if i == 10:
            break
