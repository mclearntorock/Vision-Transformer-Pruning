import torch
# from terminaltables import AsciiTable
from copy import deepcopy
import numpy as np
# import torch.nn.functional as F
from torch.nn import Conv2d
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.utils import _pair
# from models.modeling import VisionTransformer, CONFIGS
from torch.nn import LayerNorm, Linear, Conv2d
from copy import deepcopy


def gather_ln_weights(module_list, prune_idx):
    size_list = [module_list[idx].weight.sum(axis=1).data.shape[0] for idx in prune_idx]

    ln_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        ln_weights[index:(index + size)] = module_list[idx].weight.sum(axis=1).data.abs().clone()
        index += size

    return ln_weights

def get_module_list(model, config):
    module_list = []
    prune_idx = []
    block_list = []
    for i in range((config.transformer["num_layers"])):
        prune_idx.append(i)
    for layer in model.transformer.encoder.layer:
        # module_list.append(layer.attn.query)
        # module_list.append(layer.attn.key)
        # module_list.append(layer.attn.value)
        module_list.append(layer.ffn.fc1)
        block_list.append(layer)
    # module_list.append(model.transformer.encoder.part_layer.attn.query)
    # module_list.append(model.transformer.encoder.part_layer.attn.query)
    # module_list.append(model.transformer.encoder.part_layer.attn.query)
    module_list.append(model.transformer.encoder.part_layer.ffn.fc1)
    block_list.append(model.transformer.encoder.part_layer)
    return module_list, prune_idx, block_list

def obtain_ln_mask(ln_module, thre):

    thre = thre.cuda()
    mask = ln_module.weight.sum(axis=1).data.abs().ge(thre).float()

    return mask

class LNOptimizer():

    @staticmethod
    def updateLN(sr_flag, module_list, s, prune_idx):
        if sr_flag:
            # s = s if global_step <= opt.epochs * 0.5 else s * 0.01
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                ln_module = module_list[idx]
                ln_module.weight.grad.data.add_(s * torch.sign(ln_module.weight.data))  # L1

def init_weights_from_loose_model(compact_model, loose_model, LNidx2mask, config):
    compact_LN, prune_idx, compact_block = get_module_list(compact_model, config)
    loose_LN, prune_idx, loose_block = get_module_list(loose_model, config)
    block_idx = prune_idx[0:config.transformer["num_layers"]]
    # for idx in prune_idx[1:10]:
    #     out_channel_idx = np.argwhere(LNidx2mask[idx-1])[:, 0].tolist()
    #     compact_block[idx].attention_norm.weight.data       = loose_block[idx].attention_norm.weight.data[out_channel_idx].clone()
    #     compact_block[idx].attention_norm.bias.data         = loose_block[idx].attention_norm.bias.data[out_channel_idx].clone()
    #     compact_block[idx].ffn_norm.weight.data             = loose_block[idx].ffn_norm.weight.data[out_channel_idx].clone()
    #     compact_block[idx].ffn_norm.bias.data               = loose_block[idx].ffn_norm.bias.data[out_channel_idx].clone()
    #
    #     # compact_LN[idx].running_mean.data = loose_LN[idx].running_mean.data[out_channel_idx].clone()
    #     # compact_LN[idx].running_var.data  = loose_LN[idx].running_var.data[out_channel_idx].clone()
    for idx in block_idx:
        # out_channel_idx = np.argwhere(LNidx2mask[(idx*4)])[:, 0].tolist()
        # out_channel_idx_1 = np.argwhere(LNidx2mask[(idx * 4)+1])[:, 0].tolist()
        # out_channel_idx_2 = np.argwhere(LNidx2mask[(idx * 4)+2])[:, 0].tolist()
        out_channel_idx_3 = np.argwhere(LNidx2mask[idx])[:, 0].tolist()
        # next_out_channel_idx = np.argwhere(LNidx2mask[idx])[:, 0].tolist()
        # next2_out_channel_idx = np.argwhere(LNidx2mask[(idx*2 + 2)])[:, 0].tolist()

        # compact_block[idx].trans1.weight.data     = loose_block[idx].trans1.weight.data[next_out_channel_idx, :][:, out_channel_idx].clone()
        # # compact_block[idx].trans1.bias.data       = loose_block[idx].trans1.bias.data[next_out_channel_idx].clone()
        # compact_block[idx].attn.query.weight.data = loose_block[idx].attn.query.weight.data[out_channel_idx, :].clone()
        # compact_block[idx].attn.query.bias.data   = loose_block[idx].attn.query.bias.data[out_channel_idx].clone()
        # compact_block[idx].attn.key.weight.data   = loose_block[idx].attn.key.weight.data[out_channel_idx_1, :].clone()
        # compact_block[idx].attn.key.bias.data     = loose_block[idx].attn.key.bias.data[out_channel_idx_1].clone()
        # compact_block[idx].attn.value.weight.data = loose_block[idx].attn.value.weight.data[out_channel_idx_2, :].clone()
        # compact_block[idx].attn.value.bias.data   = loose_block[idx].attn.value.bias.data[out_channel_idx_2].clone()
        # compact_block[idx].attn.out.weight.data   = loose_block[idx].attn.out.weight.data[:, out_channel_idx_2].clone()
        # compact_block[idx].attn.out.bias.data     = loose_block[idx].attn.out.bias.data.clone()
        compact_block[idx].ffn.fc1.weight.data    = loose_block[idx].ffn.fc1.weight.data[out_channel_idx_3, :].clone()
        compact_block[idx].ffn.fc1.bias.data      = loose_block[idx].ffn.fc1.bias.data[out_channel_idx_3].clone()
        compact_block[idx].ffn.fc2.weight.data    = loose_block[idx].ffn.fc2.weight.data[:, out_channel_idx_3].clone()
        compact_block[idx].ffn.fc2.bias.data      = loose_block[idx].ffn.fc2.bias.data.clone()
        # compact_block[idx].trans2.weight.data     = loose_block[idx].trans2.weight.data[next2_out_channel_idx, :][:, next_out_channel_idx].clone()
        # compact_block[idx].trans2.bias.data       = loose_block[idx].trans2.bias.data[next2_out_channel_idx].clone()
        # compact_LN[idx].weight.data               = loose_LN[idx].weight.data[:, out_channel_idx][next_out_channel_idx, :].clone()
        # compact_LN[idx].bias.data                 = loose_LN[idx].bias.data[next_out_channel_idx].clone()



    # trans_idx = np.argwhere(LNidx2mask[10])[:, 0].tolist()
    # compact_model.part_head.weight.data           = loose_model.part_head.weight.data[:, last_idx].clone()
    # compact_model.transformer.encoder.trans3.weight.data              = loose_model.transformer.encoder.trans3.weight.data[:, trans_idx].clone()
    # compact_model.transformer.encoder.trans3.bias.data                = loose_model.transformer.encoder.trans3.bias.data.clone()
    # compact_model.transformer.encoder.part_norm.weight.data           = loose_model.transformer.encoder.part_norm.weight.data[last_idx].clone()
    # compact_model.transformer.encoder.part_norm.bias.data = loose_model.transformer.encoder.part_norm.bias.data[last_idx].clone()

def prune_model_keep_size2(model, prune_idx, LNidx2mask, config):
    pruned_model = deepcopy(model)
    module_list =get_module_list(model ,config)[0]
    for i in prune_idx:
          mask = torch.from_numpy(LNidx2mask[i]).cuda()
          # mask = torch.from_numpy(LNidx2mask[i])
          mask1 = mask.repeat(768,1)
          mask1 =torch.t(mask1)
          ln_module = module_list[i]
          ln_module.weight.data.mul_(mask1)
          ln_module.bias.data.mul_(mask)
    return pruned_model

def rebuild_block(model, num_channel, config):
    remake_model = deepcopy(model)
    block_list = get_module_list(remake_model, config)[2]

    for i in range(len(block_list)):
        # print(block_list[i].attn.attention_head_size)

        # block_list[i].attn.attention_head_size = int(num_channel[(4*i)] / 12)
        # block_list[i].attn.all_head_size = 12 * (block_list[i].attn.attention_head_size)
        # # print(block_list[i].attn.attention_head_size)
        # block_list[i].attn.query = Linear(config.hidden_size, num_channel[(4*i)])
        # block_list[i].attn.key = Linear(config.hidden_size, num_channel[(4*i)+1])
        # block_list[i].attn.value = Linear(config.hidden_size, num_channel[(4*i)+2])
        # block_list[i].attn.out = Linear(num_channel[(4*i)+2], config.hidden_size)
        block_list[i].ffn.fc1 = Linear(config.hidden_size, num_channel[i])
        block_list[i].ffn.fc2 = Linear(num_channel[i], config.hidden_size)
    return remake_model

def rebuild_input(model, num_channel, config):
    remake_model = deepcopy(model)
    img_size = _pair(448)
    patch_size = _pair(config.patches["size"])
    if config.split == 'non-overlap':
         n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
         remake_model.transformer.embeddings.patch_embeddings = Conv2d(in_channels=3,
                                        out_channels=num_channel,
                                        kernel_size=patch_size,
                                        stride=patch_size)
    elif config.split == 'overlap':
         n_patches = ((img_size[0] - patch_size[0]) // config.slide_step + 1) * (
                 (img_size[1] - patch_size[1]) // config.slide_step + 1)
         remake_model.transformer.embeddings.patch_embeddings = Conv2d(in_channels=3,
                                        out_channels=num_channel,
                                        kernel_size=patch_size,
                                        stride=(config.slide_step, config.slide_step))
    remake_model.transformer.embeddings.position_embeddings = torch.nn.Parameter(torch.zeros(1, n_patches + 1, num_channel))
    remake_model.transformer.embeddings.cls_token = torch.nn.Parameter(torch.zeros(1, 1, num_channel))
    return remake_model


def obtain_channels_mask(module_list, thre, prune_idx):

    pruned = 0
    total = 0
    num_channels = []
    channels_mask= []

    for idx in prune_idx:
        ln_module = module_list[idx]
        mask = obtain_ln_mask(ln_module, thre).cpu().numpy()
        remain = int(mask.sum())
        pruned = pruned + mask.shape[0] - remain

        # if remain == 0:
        #         # print("Channels would be all pruned!")
        #          # raise Exception
        #          max_value = ln_module.weight.data.abs().max()
        #          mask = obtain_ln_mask(ln_module, max_value).cpu().numpy()
        #          remain = int(mask.sum())
        #          pruned = pruned + mask.shape[0] - remain

        print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}')


        total += mask.shape[0]
        num_channels.append(remain)
        channels_mask.append(mask.copy())

    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

    return num_channels, channels_mask
# def rebuild_input2(model, num_channel, config):
#     model.transformer.embeddings = Embeddings
#





# if __name__ == "__main__":
    # config = CONFIGS['ViT-B_16']
    # model = VisionTransformer(config, 448, zero_head=True, num_classes=17,
    #                           smoothing_value=0.0)
    # model.load_from(np.load('/data-tmp/TransFG/ViT-B_16.npz'))
    # # pretrained_model = torch.load("/data-tmp/TransFG/output/srcustom_checkpoint.bin")['model']
    # # model.load_state_dict(pretrained_model)
    #
    # module_list = []
    # prune_idx = []
    # for i in range(config.transformer.num_layers*2-1):
    #     prune_idx.append(i)
    # for layer in model.transformer.encoder.layer:
    #     module_list.append(layer.attention_norm)
    #     module_list.append(layer.ffn_norm)
    # module_list.append(model.transformer.encoder.part_norm)
    # print(module_list)
    # ln_weights1 = gather_ln_weights(module_list, prune_idx)
    # print(ln_weights1)
    # sr_flag = True
    # tb_writer = SummaryWriter()
    # LNOptimizer.updateLN(sr_flag, module_list, 0.001, prune_idx)
    #
    # ln_weights = gather_ln_weights(module_list , prune_idx)
    #
    # print(ln_weights)
