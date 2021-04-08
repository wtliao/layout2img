import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm_module import *
from .mask_regression import *
from .sync_batchnorm import SynchronizedBatchNorm2d
import copy

BatchNorm = SynchronizedBatchNorm2d


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
    w_a = scaled_dot
    #w_a = scaled_dot.view(N,N)
    # print(w_g.shape)
    # print(w_a.shape)

    # multiplying log of geometric weights by feature weights
    w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
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


class ResnetGenerator128(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
        super(ResnetGenerator128, self).__init__()
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 180)

        num_w = 128 + 180
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4 * 4 * 16 * ch))

        self.res1 = ResBlock(ch * 16, ch * 16, upsample=True, num_w=num_w)
        self.res2 = ResBlock(ch * 16, ch * 8, upsample=True, num_w=num_w)
        self.res3 = ResBlock(ch * 8, ch * 4, upsample=True, num_w=num_w)
        self.res4 = ResBlock(ch * 4, ch * 2, upsample=True, num_w=num_w, psp_module=True)
        self.res5 = ResBlock(ch * 2, ch * 1, upsample=True, num_w=num_w, predict_mask=False)
        self.final = nn.Sequential(BatchNorm(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.alpha1 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha2 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha3 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha4 = nn.Parameter(torch.zeros(1, 184, 1))

        self.sigmoid = nn.Sigmoid()

        self.mask_regress = MaskRegressNetv2(num_w)
        self.init_parameter()

    def forward(self, z, bbox, z_im=None, y=None):
        b, o = z.size(0), z.size(1)
        label_embedding = self.label_embedding(y)

        z = z.view(b * o, -1)
        label_embedding = label_embedding.view(b * o, -1)

        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)

        w = self.mapping(latent_vector.view(b * o, -1))
        # preprocess bbox
        bmask = self.mask_regress(w, bbox)

        if z_im is None:
            z_im = torch.randn((b, 128), device=z.device)

        bbox_mask_ = bbox_mask(z, bbox, 64, 64)

        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        # 8x8
        x, stage_mask = self.res1(x, w, bmask)

        # 16x16
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
        x, stage_mask = self.res2(x, w, stage_bbox)

        # 32x32
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2
        x, stage_mask = self.res3(x, w, stage_bbox)

        # 64x64
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3
        x, stage_mask = self.res4(x, w, stage_bbox)

        # 128x128
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
        x, _ = self.res5(x, w, stage_bbox)

        # to RGB
        x = self.final(x)
        return x

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)


class ResnetGenerator128_context(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
        super(ResnetGenerator128_context, self).__init__()
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 180)

        num_w = 128 + 180
        self.context = BoxMultiHeadedAttention(1, num_w, dropout=0.0)
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4 * 4 * 16 * ch))

        self.res1 = ResBlock(ch * 16, ch * 16, upsample=True, num_w=num_w)
        self.res2 = ResBlock(ch * 16, ch * 8, upsample=True, num_w=num_w)
        self.res3 = ResBlock(ch * 8, ch * 4, upsample=True, num_w=num_w)
        self.res4 = ResBlock(ch * 4, ch * 2, upsample=True, num_w=num_w, psp_module=True)
        self.res5 = ResBlock(ch * 2, ch * 1, upsample=True, num_w=num_w, predict_mask=False)
        self.final = nn.Sequential(BatchNorm(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.alpha1 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha2 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha3 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha4 = nn.Parameter(torch.zeros(1, 184, 1))

        self.sigmoid = nn.Sigmoid()

        self.mask_regress = MaskRegressNetv2(num_w)
        self.init_parameter()

    def forward(self, z, bbox, z_im=None, y=None):
        b, o = z.size(0), z.size(1)
        label_embedding = self.label_embedding(y)

        z = z.view(b * o, -1)
        label_embedding = label_embedding.view(b * o, -1)

        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)

        w = self.mapping(latent_vector.view(b * o, -1)).view(b, o, -1)
        # print(w.shape)

        # context transformation
        w = self.context(w, w, w, bbox, y)
        # print(w.shape)
        w = w.view(b * o, -1)
        # print(w.shape)
        # preprocess bbox
        bmask = self.mask_regress(w, bbox)

        if z_im is None:
            z_im = torch.randn((b, 128), device=z.device)

        bbox_mask_ = bbox_mask(z, bbox, 64, 64)

        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        # 8x8
        x, stage_mask = self.res1(x, w, bmask)

        # 16x16
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
        x, stage_mask = self.res2(x, w, stage_bbox)

        # 32x32
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2
        x, stage_mask = self.res3(x, w, stage_bbox)

        # 64x64
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3
        x, stage_mask = self.res4(x, w, stage_bbox)

        # 128x128
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
        x, _ = self.res5(x, w, stage_bbox)

        # to RGB
        x = self.final(x)
        return x

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)


class ResnetGenerator256(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
        super(ResnetGenerator256, self).__init__()
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 180)

        num_w = 128 + 180
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4 * 4 * 16 * ch))

        self.res1 = ResBlock(ch * 16, ch * 16, upsample=True, num_w=num_w)
        self.res2 = ResBlock(ch * 16, ch * 8, upsample=True, num_w=num_w)
        self.res3 = ResBlock(ch * 8, ch * 8, upsample=True, num_w=num_w)
        self.res4 = ResBlock(ch * 8, ch * 4, upsample=True, num_w=num_w)
        self.res5 = ResBlock(ch * 4, ch * 2, upsample=True, num_w=num_w)
        self.res6 = ResBlock(ch * 2, ch * 1, upsample=True, num_w=num_w, predict_mask=False)
        self.final = nn.Sequential(BatchNorm(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.alpha1 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha2 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha3 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha4 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha5 = nn.Parameter(torch.zeros(1, 184, 1))
        self.sigmoid = nn.Sigmoid()

        self.mask_regress = MaskRegressNetv2(num_w)
        self.init_parameter()

    def forward(self, z, bbox, z_im=None, y=None, include_mask_loss=False):
        b, o = z.size(0), z.size(1)

        label_embedding = self.label_embedding(y)

        z = z.view(b * o, -1)
        label_embedding = label_embedding.view(b * o, -1)

        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)

        w = self.mapping(latent_vector.view(b * o, -1))

        # preprocess bbox
        bmask = self.mask_regress(w, bbox)

        if z_im is None:
            z_im = torch.randn((b, 128), device=z.device)

        bbox_mask_ = bbox_mask(z, bbox, 128, 128)

        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)
        w = self.mapping(latent_vector.view(b * o, -1))

        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        # 8x8
        # label mask
        x, stage_mask = self.res1(x, w, bmask)

        # 16x16
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
        x, stage_mask = self.res2(x, w, stage_bbox)

        # 32x32
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

        alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2
        x, stage_mask = self.res3(x, w, stage_bbox)

        # 64x64
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

        alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3
        x, stage_mask = self.res4(x, w, stage_bbox)

        # 128x128
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

        alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
        x, stage_mask = self.res5(x, w, stage_bbox)

        # 256x256
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

        alpha5 = torch.gather(self.sigmoid(self.alpha5).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha5) + seman_bbox * alpha5
        x, _ = self.res6(x, w, stage_bbox)
        # to RGB
        x = self.final(x)
        return x

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, upsample=False, num_w=128, predict_mask=True, psp_module=False):
        super(ResBlock, self).__init__()
        self.upsample = upsample
        self.h_ch = h_ch if h_ch else out_ch
        self.conv1 = conv2d(in_ch, self.h_ch, ksize, pad=pad)
        self.conv2 = conv2d(self.h_ch, out_ch, ksize, pad=pad)
        self.b1 = SpatialAdaptiveSynBatchNorm2d(in_ch, num_w=num_w, batchnorm_func=BatchNorm)
        self.b2 = SpatialAdaptiveSynBatchNorm2d(self.h_ch, num_w=num_w, batchnorm_func=BatchNorm)
        self.learnable_sc = in_ch != out_ch or upsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.activation = nn.ReLU()

        self.predict_mask = predict_mask
        if self.predict_mask:
            if psp_module:
                self.conv_mask = nn.Sequential(PSPModule(out_ch, 100),
                                               nn.Conv2d(100, 184, kernel_size=1))
            else:
                self.conv_mask = nn.Sequential(nn.Conv2d(out_ch, 100, 3, 1, 1),
                                               BatchNorm(100),
                                               nn.ReLU(),
                                               nn.Conv2d(100, 184, 1, 1, 0, bias=True))

    def residual(self, in_feat, w, bbox):
        x = in_feat
        x = self.b1(x, w, bbox)
        x = self.activation(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.b2(x, w, bbox)
        x = self.activation(x)
        x = self.conv2(x)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            if self.upsample:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.c_sc(x)
        return x

    def forward(self, in_feat, w, bbox):
        out_feat = self.residual(in_feat, w, bbox) + self.shortcut(in_feat)
        if self.predict_mask:
            mask = self.conv_mask(out_feat)
        else:
            mask = None
        return out_feat, mask


def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv


def batched_index_select(input, dim, index):
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def bbox_mask(x, bbox, H, W):
    bbox = bbox.to(x.device)
    b, o, _ = bbox.size()
    N = b * o

    bbox_1 = bbox.float().view(-1, 4)
    x0, y0 = bbox_1[:, 0], bbox_1[:, 1]
    ww, hh = bbox_1[:, 2], bbox_1[:, 3]

    x0 = x0.contiguous().view(N, 1).expand(N, H)
    ww = ww.contiguous().view(N, 1).expand(N, H)
    y0 = y0.contiguous().view(N, 1).expand(N, W)
    hh = hh.contiguous().view(N, 1).expand(N, W)

    X = torch.linspace(0, 1, steps=W).view(1, W).expand(N, W).to(x.device)  # cuda(device=x.device)
    Y = torch.linspace(0, 1, steps=H).view(1, H).expand(N, H).to(x.device)  # cuda(device=x.device)

    X = (X - x0) / ww
    Y = (Y - y0) / hh

    X_out_mask = ((X < 0) + (X > 1)).view(N, 1, W).expand(N, H, W)
    Y_out_mask = ((Y < 0) + (Y > 1)).view(N, H, 1).expand(N, H, W)

    out_mask = 1 - (X_out_mask + Y_out_mask).float().clamp(max=1)
    return out_mask.view(b, o, H, W)


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features)
        return nn.Sequential(prior, conv, bn, nn.ReLU())

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle
