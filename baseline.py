from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
from torch.utils.data import (SequentialSampler)
import numpy as np
import random
import os
from collections import OrderedDict
import pickle
import time
import argparse
from modules.tokenization import BertTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.until_module import PreTrainedModel, LayerNorm
from torch.nn import CrossEntropyLoss
from modules.modeling import UniAV, UniVL
from modules.module_baseline import BaseModel, BaseClassifyHead, CrossConfig
from modules.optimization import BertAdam
from dataloaders.dataloader_howto100m import Youtube_DataLoader
from dataloaders.dataloader_emotiondata import Base_DataLoader
from torch.utils.data import DataLoader
from util import get_logger
from torch import nn

torch.distributed.init_process_group(backend="nccl")

global logger


class BaseLineModel(PreTrainedModel, nn.Module):
    def __init__(self, base_config):
        self.cross = BaseModel(base_config)
        self.classifier = BaseClassifyHead(base_config)
        self.cross_entropy_loss = CrossEntropyLoss()

    @classmethod
    def from_pretrained(self, base_model_name,  cache_dir=None, type_vocab_size=2, *inputs, **kwargs):
        base_config = CrossConfig.get_config(base_model_name, cache_dir, type_vocab_size, state_dict=None)
        model = self(base_config)
        return model

    def forward(self, audio, video, emotion_label):
        cross_output, pooled_output = self.get_model_output(audio, video)
        audio_cross_ouput, video_cross_ouput = torch.split(cross_output, [audio.size[1], video.size[1]], dim=1)
        class_output = self.classifier(audio_cross_ouput, video_cross_ouput)

        loss = self.cross_entropy_loss(class_output, emotion_label)

        return loss



    def get_model_output(self, audio_feature, video_feature):
        concat_features = torch.cat((audio_feature, video_feature), dim=1)
        audio_type = torch.zeros(audio_feature.size(0), audio_feature.size(1))
        video_type = torch.ones(video_feature.size(0), video_feature.ize(1))
        concat_type = torch.cat((audio_type, video_type), dim=1)
        cross_layers,  pooled_output = self.cross(concat_features, concat_type, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output





def get_args(description='UniVL on Pretrain'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--cmumosei_csv', type=str, default='data/cmumosei.csv', help='cmumosei csv')
    parser.add_argument('--features_path', type=str, default='feature', help='feature path for 2D features')
    parser.add_argument('--data_path', type=str, default='data/data.pickle', help='data pickle file path')
    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--min_words', type=int, default=0, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--min_time', type=float, default=5.0, help='Gather small clips')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default="", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=False,
                        help="Bert pre-trained model")
    parser.add_argument("--audio_model", default="audio-base", type=str, help="Audio model")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true',
                        help="Whether use MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--audio_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=2, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=3, help="Layer NO. of decoder.")
    parser.add_argument('--num_hidden_layers', type=int, default=6, help="Layer NO. of encoder.")

    parser.add_argument('--stage_two', action='store_true', help="Whether training with decoder.")
    parser.add_argument('--pretrain_enhance_vmodal', action='store_true',
                        help="Enhance visual and other modalities when pretraining.")

    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_model", default="pytorch_base_model.bin.checkpoint", type=str, required=False,
                        help="Save the last model as a checkpoint.")
    
    #运行时参数
    parser.add_argument("--nproc_per_node",type=int, default=8)


    args = parser.parse_args()

    if args.sampled_use_mil:  # sample from each video, has a higher priority than use_mil.
        args.use_mil = True

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_pretrain:
        raise ValueError("`do_pretrain` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    args.checkpoint_model = '{}_{}_{}_{}.checkpoint'.format(args.checkpoint_model, args.bert_model, args.max_words,
                                                            args.max_frames)

    return args


def set_seed_logger(args):
    global logger
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu


def init_model(args, device, n_gpu, local_rank):
    if args.init_model:  # None
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    # cache_dir = ""
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = BaseLineModel.from_pretrained(args.cross_model,  cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model


def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):
    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." not in n]

    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_linear', t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model


def dataloader_basetrain(args):
    # if args.local_rank == 0:
    #     logger.info('Loading captions: {}'.format(args.data_path))
    # # data_dict = pickle.load(open(args.data_path, 'rb'))
    # if args.local_rank == 0:
    #     logger.info('Done, data_dict length: {}'.format(len(data_dict)))

    cmumosei_dataset = Base_DataLoader(
        csv=args.cmumosei_csv,
        features_path=args.features_path,
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(cmumosei_dataset)
    dataloader = DataLoader(
        cmumosei_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )

    return dataloader, len(cmumosei_dataset), sampler


def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict


def save_model(epoch, args, model, local_rank, type_name="", global_step=-1, optimizer=None):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_base_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)

    if global_step != -1 and optimizer is not None:
        state_dict = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model_to_save.state_dict(),
            'last_optimizer_state': convert_state_dict_type(optimizer.state_dict()),
        }
        checkpoint_model_file = os.path.join(args.output_dir, args.checkpoint_model)
        torch.save(state_dict, checkpoint_model_file)
        logger.info("Checkpoint is saved. use `load_checkpoint` to recovery it.")

    return output_model_file


def load_model(epoch, args, n_gpu, device, model, global_step=0, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_base_model.bin.{}".format(epoch))

    last_optim_state = None
    checkpoint_model_file = os.path.join(args.output_dir, args.checkpoint_model)
    if epoch == -1 and args.load_checkpoint and os.path.exists(checkpoint_model_file):
        checkpoint_state = torch.load(checkpoint_model_file, map_location='cpu')
        epoch = checkpoint_state['epoch']
        global_step = checkpoint_state['global_step']
        model_state_dict = checkpoint_state['model_state_dict']
        last_optim_state = checkpoint_state['last_optimizer_state']
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed')
        model = BaseLineModel.from_pretrained( args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
        if args.local_rank == 0:
            logger.info("Checkpoint loaded from %s", checkpoint_model_file)
    elif os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)

        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed')
        model = BaseLineModel.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)

    return epoch, global_step, last_optim_state, model


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        audio,  video, emotion_label = batch

        loss = model(audio, video, emotion_label)

        if n_gpu > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scheduler is not None:
                scheduler.step()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader),
                            "-".join([str('%.6f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    model = init_model(args, device, n_gpu, args.local_rank)

    train_dataloader, train_length, sampler = dataloader_basetrain(args)
    num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                    / args.gradient_accumulation_steps) * args.epochs

    global_step = 0
    epoch = -1
    last_optim_state = None
    if args.load_checkpoint:
        epoch, global_step, last_optim_state, model = load_model(epoch, args, n_gpu, device, model,
                                                                 global_step=global_step)
        epoch += 1
        if args.local_rank == 0:
            logger.warning("Will continue to epoch: {}".format(epoch))
    epoch = 0 if epoch < 0 else epoch

    coef_lr = args.coef_lr  # 0.1
    if args.init_model:
        coef_lr = 1.0

    optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu,
                                                 args.local_rank, coef_lr=coef_lr)
    if last_optim_state is not None:
        optimizer.load_state_dict(last_optim_state)

    if args.local_rank == 0:
        logger.info("***** Running pretraining *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

    iter_ls_ = [itm for itm in range(args.epochs) if itm >= epoch]
    for epoch in iter_ls_:
        sampler.set_epoch(epoch)

        tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                           scheduler, global_step, local_rank=args.local_rank)

        if args.local_rank == 0:
            logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
            save_model(epoch, args, model, args.local_rank, type_name="pretrain", global_step=global_step,
                       optimizer=optimizer)


if __name__ == "__main__":
    main()

'''
ROOT_PATH=.
DATA_PATH=${ROOT_PATH}/data
SAVE_PATH=${ROOT_PATH}/models
MODEL_PATH=${ROOT_PATH}/UniAV
python -m torch.distributed.launch  \
${MODEL_PATH}/baseline.py \
 --do_pretrain --num_thread_reader=0 --epochs=1 \
--batch_size=128 --n_pair=1 --n_display=100 \
--do_lower_case --lr 1e-4 \
--max_words 48 --max_frames 512 --batch_size_val 344 \
--output_dir ${SAVE_PATH}/pre_trained/basemodel \
--features_path ${ROOT_PATH}/feature \
--visual_num_hidden_layers 6 --gradient_accumulation_steps 16 

'''