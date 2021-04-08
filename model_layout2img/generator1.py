import torch
import torch.nn as nn
from models2.bilinear import crop_bbox_batch
from models2.transformer import transformer_encoder
from models2.transformer import transformer_decoder


class generate_latent(nn.Module):
    def __init__(self, embd_dim, pos_dim, num_class):
        super(generate_latent, self).__init__()
        self.pos_embd = nn.Linear(4, pos_dim)
        self.class_embd = nn.Embedding(num_class, embd_dim)

    def forward(self, cla, pos, z):
        pos_embd = self.pos_embd(pos)
        cla_embd = self.class_embd(cla)
        return torch.cat([cla_embd, z], dim=1), pos_embd


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
    holder = torch.zeros(len(obj_num), L, z.size(1)).cuda()
    mask = torch.zeros(len(obj_num), L)
    start = 0
    for i in range(len(obj_num)):
        holder[i, :obj_num[i], :] = z[start:start + obj_num[i], :]
        mask[i, :obj_num[i]] = 1
        start = start + obj_num[i]
    return holder, mask


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
        self.dim_in = dim_in
        self.dim_out = dim_out
        if dim_in != dim_out:
            self.conv = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True))
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        if self.dim_in == self.dim_out:

            return x + self.main(x)
        else:
            return self.conv(x) + self.main(x)


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
    def __init__(self, conv_dim=64, z_dim=8, embedding_dim=64, pos_dim=64, resi_num=6):
        super(LayoutEncoder, self).__init__()
        self.activation = nn.ReLU(inplace=True)

        layers = []
        # Bottleneck layers.
        in_dim = 64
        out_dim = 64
        for i in range(resi_num):
            if (i + 1) % 2 == 0:
                layers.append(ResidualBlock(dim_in=in_dim, dim_out=out_dim * 2))
                in_dim = in_dim * 2
                out_dim = out_dim * 2
            else:

                layers.append(ResidualBlock(dim_in=in_dim, dim_out=out_dim))
        self.residual = nn.Sequential(*layers)

        # (emb+z, 64, 64) -> (64, 64, 64)
        self.c1 = nn.Conv2d(embedding_dim + z_dim + pos_dim, conv_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(conv_dim)  # if class_num == 0 else ConditionalBatchNorm2d(conv_dim, class_num)

    def forward(self, z):
        # prepare mask fm
        #B = z.size(0)
        #objs = torch.arange(B).cuda()
        #embeddings = self.embedding(objs)
        #embeddings_z = torch.cat((embeddings, z), dim=1)
        #h = embeddings_z.view(embeddings_z.size(0), embeddings_z.size(1), 1, 1) * masks
        # print(h.shape)
        # downsample layout
        h = self.c1(z)
        h = self.bn1(h)
        h = self.activation(h)

        # print(h.shape)
        # clstm fusion (O, 512, 8, 8) -> (n, 64, 8, 8)
        #h = self.clstm(h, obj_to_img)
        # residual block
        h = self.residual(h)
        # print(h.shape)

        return h


class Decoder(nn.Module):
    def __init__(self, conv_dim=512):
        super(Decoder, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        # (64, 8, 8) -> (256, 8, 8)
        self.c0 = nn.Conv2d(conv_dim, conv_dim // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(conv_dim // 2)
        # (256, 8, 8) -> (256, 16, 16)
        self.dc1 = nn.ConvTranspose2d(conv_dim // 2, conv_dim // 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv_dim // 2)
        # (256, 16, 16) -> (128, 32, 32)
        self.dc2 = nn.ConvTranspose2d(conv_dim // 2, conv_dim // 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(conv_dim // 4)
        # (128, 32, 32) -> (64, 64, 64)
        self.dc3 = nn.ConvTranspose2d(conv_dim // 4, conv_dim // 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv_dim // 8)
        # (64, 64, 64) -> (3, 64, 64)
        self.c4 = nn.Conv2d(conv_dim // 8, 3, kernel_size=7, stride=1, padding=3, bias=True)

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
    def __init__(self, num_embeddings, embedding_dim=64, z_dim=8, pos_dim=64, obj_size=32):
        super(Generator, self).__init__()
        self.obj_size = obj_size
        # (3, 32, 32) -> (256, 4, 4) -> 8
        self.crop_encoder = CropEncoder(z_dim=z_dim, class_num=num_embeddings)
        self.latent_generator = generate_latent(embedding_dim, pos_dim, num_embeddings)
        #self.transformer_encoder = transformer_encoder(3)
        self.transformer_decoder = transformer_decoder(3)
        self.layout_encoder = LayoutEncoder(z_dim=z_dim, embedding_dim=embedding_dim, pos_dim=pos_dim)
        self.decoder = Decoder(conv_dim=512)
        # self.apply(weights_init)

    def forward(self, imgs, objs, boxes, masks, obj_to_img, z_rand):
        crops_input = crop_bbox_batch(imgs, boxes, obj_to_img, self.obj_size)
        z_rec, mu, logvar = self.crop_encoder(crops_input, objs)
        # print(z_rec.shape)

        # semantic and spatial information
        latent_z, pos_z = self.latent_generator(objs, boxes, z_rec)
        # print(latent_z.shape)
        latent_z_rand, pos_z_rand = self.latent_generator(objs, boxes, z_rand)
        composed_z, mask = featuremap_composition(latent_z, obj_to_img)
        print(composed_z.shape)
        print(mask.shape)
        composed_z_rand, mask_rand = featuremap_composition(latent_z_rand, obj_to_img)
        feat = self.transformer_decoder(composed_z, mask)
        feat_rand = self.transformer_decoder(composed_z_rand, mask)
        B = feat.size(0)
        C = feat.size(2)
        feat = feat.permute(0, 2, 1).contiguous().view(B, C, 8, 8)
        feat_rand = feat_rand.permute(0, 2, 1).contiguous().view(B, C, 8, 8)
        # print(feat.shape)
        h_rec = self.layout_encoder(feat)
        # print(h_rec.shape)
        h_rand = self.layout_encoder(feat_rand)

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
