from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import shutil
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from alad.dataset import RetrievalDataset, MyCollate
import yaml
import time

from oscar.utils.tsv_file import TSVFile
from oscar.utils.logger import setup_logger
from oscar.utils.misc import mkdir, set_seed
from transformers.pytorch_transformers import BertTokenizer, BertConfig

from alad.loss import AlignmentContrastiveLoss
from alad.alad_model import ALADModel
from alad.recall_auxiliary import compute_recall
# from utils import get_model, cosine_sim, dot_sim
from alad.evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data
from alad.evaluate_utils.dcg import DCG
# from teran import data

import logging
from torch.utils.tensorboard import SummaryWriter

best_rsum = 0
best_ndcg_sum = 0

def main():
    global best_rsum
    global best_ndcg_sum

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='datasets/coco_ir', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--img_feat_file", default='datasets/coco_ir/features.tsv', type=str, required=False,
                        help="The absolute address of the image feature file.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or model type. required for training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--loss_type", default='sfmx', type=str,
                        help="Loss function types: support kl, sfmx")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=70, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded."
                             "This number is calculated on COCO dataset"
                             "If add object detection labels, the suggested length should be 70.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run performance valuation."
                                                               "do not activate if we want to inference on dataset without gt labels.")
    parser.add_argument("--test_split", default='test', type=str, help='data split name.')
    parser.add_argument("--eval_img_keys_file", default='', type=str,
                        help="image key tsv to select a subset of images for evaluation. "
                             "This is useful in 5-folds evaluation. The topn index file is not "
                             "needed in this case.")
    parser.add_argument("--eval_caption_index_file", default='', type=str,
                        help="index of a list of (img_key, cap_idx) for each image."
                             "this is used to perform re-rank using hard negative samples."
                             "useful for validation set to monitor the performance during training.")
    parser.add_argument("--cross_image_eval", action='store_true',
                        help="perform cross image inference, ie. each image with all texts from other images.")
    parser.add_argument("--add_od_labels", default=False, action='store_true',
                        help="Whether to add object detection labels or not.")
    parser.add_argument("--od_label_type", default='vg', type=str,
                        help="label type, support vg, gt, oid")
    parser.add_argument("--att_mask_type", default='CLR', type=str,
                        help="attention mask type, support ['CL', 'CR', 'LR', 'CLR']"
                             "C: caption, L: labels, R: image regions; CLR is full attention by default."
                             "CL means attention between caption and labels."
                             "please pay attention to the order CLR, which is the default concat order.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=50, type=int,
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int,
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument("--use_img_layernorm", type=int, default=1,
                        help="Normalize image features with bertlayernorm")
    parser.add_argument("--img_layer_norm_eps", default=1e-12, type=float,
                        help="The eps in image feature laynorm layer")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int,
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument("--num_captions_per_img_train", default=5, type=int,
                        help="number of positive matched captions for each training image.")
    parser.add_argument("--num_captions_per_img_val", default=5, type=int,
                        help="number of captions for each testing image.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1,
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--eval_model_dir", type=str, default='',
                        help="Model directory for evaluation.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")

    # -----------------------------------------------------------------------------------------
    # TERAN Arguments
    # -----------------------------------------------------------------------------------------
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', default='/w/31/faghri/vsepp_data/',
    #                     help='path to datasets')
    # parser.add_argument('--data_name', default='precomp',
    #                     help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    # parser.add_argument('--crop_size', default=224, type=int,
    #                     help='Size of an image crop as the CNN input.')
    # parser.add_argument('--workers', default=10, type=int,
    #                     help='Number of data loader workers.')
    parser.add_argument('--load_checkpoint', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads only the model')
    # parser.add_argument('--use_restval', action='store_true',
    #                     help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--config', type=str, help="Which configuration to use. See into 'config' folder")


    args = parser.parse_args()
    print(args)

    # torch.cuda.set_enabled_lms(True)
    # if (torch.cuda.get_enabled_lms()):
    #     torch.cuda.set_limit_lms(11000 * 1024 * 1024)
    #     print('[LMS=On limit=' + str(torch.cuda.get_limit_lms()) + ']')

    # with open(args.config, 'r') as ymlfile:
    #     config = yaml.load(ymlfile)

    # check if checkpoint exists
    filename = args.load_checkpoint
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        loaded_checkpoint = torch.load(filename, map_location='cpu')
    else:
        raise FileNotFoundError("=> no checkpoint found at '{}'".format(filename))

    config = loaded_checkpoint['config']
    args.per_gpu_train_batch_size = config['training']['bs']
    args.per_gpu_eval_batch_size = config['training']['bs']

    # Warn: these flags are misleading: they switch Oscar in the right configuration for the Alad setup (see dataset.py)
    args.do_test = True
    args.do_eval = True

    # override loss-type (we want to test both the alignment and the matching head)
    config['training']['loss-type'] = 'alignment-distillation'

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    global logger
    logger = setup_logger("vlpretrain", None, 0)

    # Train the Model
    loss_type = config['training']['loss-type']
    alignment_mode = config['training']['alignment-mode'] if 'alignment' in loss_type else None

    # validate(val_loader, model, tb_logger, measure=config['training']['measure'], log_step=opt.log_step,
    #          ndcg_scorer=ndcg_val_scorer, alignment_mode=alignment_mode)

    # ------------------------------------------------------------------------------------------------------------------
    # Data initialization (data pipeline is from Oscar)
    # ------------------------------------------------------------------------------------------------------------------

    args = restore_training_settings(args)

    oscar_checkpoint = args.eval_model_dir
    assert op.isdir(oscar_checkpoint)
    logger.info("Evaluate the following checkpoint: %s", oscar_checkpoint)

    config_class, tokenizer_class = BertConfig, BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(oscar_checkpoint)

    split = 'test' #'minival'
    is_train = False

    test_dataset = RetrievalDataset(tokenizer, args, split, is_train=is_train)
    test_collate = MyCollate(dataset=test_dataset, return_oscar_data=False)
    test_loader = DataLoader(test_dataset, shuffle=False,
                              batch_size=config['training']['bs'], num_workers=args.num_workers,
                              collate_fn=test_collate)

    # load the ndcg scorer
    ndcg_val_scorer = None # DCG(config, len(val_loader.dataset), 'val', rank=25, relevance_methods=['rougeL', 'spice'])
    # ndcg_test_scorer = DCG(config, len(test_loader.dataset), 'test', rank=25, relevance_methods=['rougeL', 'spice'])

    # ------------------------------------------------------------------------------------------------------------------
    # Oscar initialization (only if distillation is on)
    # ------------------------------------------------------------------------------------------------------------------

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed, args.n_gpu)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    logger.info('output_mode: {}, #Labels: {}'.format(args.output_mode, args.num_labels))

    # model = model_class.from_pretrained(checkpoint, config=config)

    # model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # inference and evaluation
    #args.do_test or args.do_eval:
    # checkpoint = args.eval_model_dir
    #assert op.isdir(checkpoint)
    # logger.info("Evaluate the following checkpoint: %s", checkpoint)

    # Construct the student model
    student_model = ALADModel(config, oscar_checkpoint)

    # resume from a checkpoint
    student_model.load_state_dict(loaded_checkpoint['model'], strict=True)
    if torch.cuda.is_available():
        student_model.cuda()
    print('Model checkpoint loaded!')

    test(test_loader, student_model, alignment_mode=alignment_mode)


def test(test_loader, model, measure='cosine', log_step=10, ndcg_scorer=None, alignment_mode=None):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, img_lenghts, cap_lenghts = encode_data(
        model, test_loader, log_step, logging.info)

    # initialize similarity matrix evaluator
    sim_matrix_class = AlignmentContrastiveLoss(aggregation=alignment_mode)
    def alignment_sim_fn(img, cap, img_len, cap_len):
        with torch.no_grad():
            scores = sim_matrix_class(img, cap, img_len, cap_len, return_loss=False, return_similarity_mat=True)
        return scores
    sim_matrix_fn = alignment_sim_fn

    print('Evaluating matching head...')
    compute_recall(img_embs[:, 0, :], cap_embs[:, 0, :])

    print('Evaluating alignment head...')
    # caption retrieval
    (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg) = i2t(img_embs, cap_embs, img_lenghts, cap_lenghts, measure=measure, ndcg_scorer=ndcg_scorer, sim_function=sim_matrix_fn, cap_batches=5)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, ndcg_rouge=%.4f ndcg_spice=%.4f" %
                 (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr, mean_rougel_ndcg_i, mean_spice_ndcg_i) = t2i(
        img_embs, cap_embs, img_lenghts, cap_lenghts, ndcg_scorer=ndcg_scorer, measure=measure, sim_function=sim_matrix_fn, im_batches=5)

    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, ndcg_rouge=%.4f ndcg_spice=%.4f" %
                 (r1i, r5i, r10i, medri, meanr, mean_rougel_ndcg_i, mean_spice_ndcg_i))

def restore_training_settings(args):
    assert not args.do_train and (args.do_test or args.do_eval)
    train_args = torch.load(op.join(args.eval_model_dir, 'training_args.bin'))
    # override_params = ['do_lower_case', 'img_feature_type', 'max_seq_length',
    #         'max_img_seq_length', 'add_od_labels', 'od_label_type',
    #         'use_img_layernorm', 'img_layer_norm_eps']
    override_params = ['do_lower_case', 'img_feature_type', 'add_od_labels', 'od_label_type',
             'use_img_layernorm', 'img_layer_norm_eps']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)
    return args

if __name__ == "__main__":
    main()