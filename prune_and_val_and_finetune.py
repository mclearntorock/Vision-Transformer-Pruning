import argparse
import logging
import time
from copy import deepcopy
import numpy as np

import torch
from terminaltables import AsciiTable
from tqdm import tqdm

from models.modeling import VisionTransformer, CONFIGS
from train import count_parameters, train, set_seed
from utils.data_utils import get_loader
from utils.prune_utils import obtain_ln_mask, get_module_list, gather_ln_weights, init_weights_from_loose_model, \
    prune_model_keep_size2
from utils.prune_utils import rebuild_input, rebuild_block


logger = logging.getLogger(__name__)
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    # dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def val(args, model, test_loader):
    # Validation!

    eval_losses = AverageMeter()
    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    accuracy = torch.tensor(accuracy).to(args.device)
    # dist.barrier()
    val_accuracy = reduce_mean(accuracy, args.nprocs)
    val_accuracy = val_accuracy.detach().cpu().numpy()
    return val_accuracy

def obtain_avg_forward_time(input, model, repeat=200):

    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(repeat):
             output = model(input)[0]
    avg_infer_time = (time.time() - start) / repeat

    return avg_infer_time, output

def prune_and_eval(args, model, sorted_ln, percent=.0):
    model_copy = deepcopy(model)
    thre_index = int(len(sorted_ln) * percent)
    thre = sorted_ln[thre_index]
    # for idx in range(len(sorted_weights)):
    #     # if (idx+1)%4 == 0:
    #     value = sorted_weights[idx][1200]
    #     # else :
    #     #     value = sorted_weights[idx][36]
    #     thre.append(value)


    module_list, prune_idx, _ = get_module_list(model_copy, config)
    print(thre)

    remain_num = 0
    for idx in prune_idx:
        bn_module = module_list[idx]

        mask = obtain_ln_mask(bn_module, thre)
        remain_num += int(mask.sum())
        mask1 = mask.repeat(768,1)
        mask1 = torch.t(mask1)

        bn_module.weight.data.mul_(mask1)
        bn_module.bias.data.mul_(mask)
    print("let's test the current model!")
    with torch.no_grad():
         if args.val:
            accuracy = val(args, model_copy, test_loader)
            print(f"mAP of the 'pruned' model is {accuracy:.4f}")
    print(f'Number of channels has been reduced  to {remain_num}')
    # print(f'Prune ratio: {1 - remain_num / len(sorted_ln):.3f}')


    return thre


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='finetune',
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017", "FGcsms"],
                        default="CUB_200_2011",
                        help="Which dataset.")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument('--data_root', type=str, default='E:\FGcsms')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument('--percent', type=float, default=0.84, help='channel prune percent')
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--pretrained_model", type=str, default='output/srwcustom_checkpoint.bin',
                        help="load pretrained model")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=8462, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--learning_rate", default=3e-4, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=200000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=20000, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--val", action='store_true',
                        help="val.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--finetune", default='store_true',
                        help="finetune.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                        help='train with channel sparsity regularization')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device ="cpu"
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.nprocs = torch.cuda.device_count()
    config = CONFIGS['ViT-B_16']
    model = VisionTransformer(config, 448, zero_head=True, num_classes=17,
                              smoothing_value=0.0)

    # %%
    # load sparse model
    pretrained_model = torch.load(args.pretrained_model)['model']
    model.load_state_dict(pretrained_model)

    model.to(args.device)

    origin_nparameters = count_parameters(model)

    # eval sparse model
    print("\nlet's test the original model first:")
    train_loader, test_loader = get_loader(args)
    if args.val:
        origin_model_metric = val(args, model, test_loader)

        print("Valid Accuracy: %2.5f" % origin_model_metric)
    # # get relative data
    module_list, prune_idx, _= get_module_list(model, config)
    # weights_per_idx = []

    # sorted_weights = []
    # for idx in prune_idx:
    ln_weights = gather_ln_weights(module_list, prune_idx)
    sorted_ln = torch.sort(ln_weights)[0]
        # weights_per_idx.append(ln_weights)
        # sorted_weights.append(sorted_ln)

    # # 避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
    highest_thre = []
    for idx in prune_idx:
        highest_thre.append(module_list[idx].weight.sum(axis=1).data.abs().max().item())
    highest_thre = min(highest_thre)

    # 找到highest_thre对应的下标对应的百分比
    percent_limit = (sorted_ln==highest_thre).nonzero().item() / len(ln_weights)

    print(f'Suggested Gamma threshold should be less than {highest_thre:.4f}.')
    print(f'The corresponding prune ratio is {percent_limit:.3f}, but you can set higher.')

    percent = args.percent
    print('the required prune percent is', percent)
    threshold = prune_and_eval(args, model, sorted_ln, percent)

    # %%
    # Prune model
    num_channels, channels_mask = obtain_channels_mask(module_list, threshold, prune_idx)
    LNidx2mask = {idx: mask.astype('float32') for idx, mask in zip(prune_idx, channels_mask)}
    pruned_model = prune_model_keep_size2(model, prune_idx, LNidx2mask, config)
    #
    # print(
    #     "\nnow prune the model but keep size,(actually add offset of BN beta to next layer), let's see how the mAP goes")
    # with torch.no_grad():
    #     if args.val:
    #        pruned_ac = val(args, pruned_model, test_loader)
    #        print("Valid Accuracy: %2.5f" % pruned_ac)

    # %%
    # Rebuild model with fixed channels
    # compact_model = rebuild_block(rebuild_input(pruned_model, num_channels[0], config), num_channels, config).to(device)
    compact_model = rebuild_block(pruned_model, num_channels, config).to(device)
    # compact_model.transformer.embeddings =
    compact_nparameters = count_parameters(compact_model)

    init_weights_from_loose_model(compact_model, pruned_model, LNidx2mask, config)

    # %%
    # Compare pruned_model and sparse_model
    random_input = torch.rand((1, 3, args.img_size,  args.img_size)).to(device)

    print('\ntesting avg forward time...')
    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
    compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)

    diff = (pruned_output - compact_output).abs().gt(0.001).sum().item()
    if diff > 0:
        print('Something wrong with the pruned model!')

    # %%
    # 在测试集上测试剪枝后的模型, 并统计模型的参数数量
    print('testing the mAP of final pruned model')
    with torch.no_grad():
        if args.val:
           compact_model_metric = val(args, compact_model, test_loader)
           print("Valid Accuracy: %2.5f" % compact_model_metric)
    # #%%
    # # 比较剪枝前后参数数量的变化、指标性能的变化
    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", f'{origin_model_metric:.6f}', f'{compact_model_metric:.6f}'],
        ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
        ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
    ]
    print(AsciiTable(metric_table).table)
    #
    # # %%
    # # Finetune model if necessary
    # if args.finetune:
    #     # Setup logging
    #     logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    #                         datefmt='%m/%d/%Y %H:%M:%S',
    #                         level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    #     logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
    #                    (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))
    #
    #     # Set seed
    #     set_seed(args)
    #
    #     train(args, compact_model)