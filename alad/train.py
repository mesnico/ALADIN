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
    # parser.add_argument("--num_train_epochs", default=20, type=int,
    #                     help="Total number of training epochs to perform.")
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
    parser.add_argument('--num_epochs', default=20, type=int,
                        help='Number of training epochs.')
    # parser.add_argument('--crop_size', default=224, type=int,
    #                     help='Size of an image crop as the CNN input.')
    # parser.add_argument('--lr_update', default=15, type=int,
    #                     help='Number of epochs to update the learning rate.')
    # parser.add_argument('--workers', default=10, type=int,
    #                     help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--test_step', default=100000000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads model, optimizer, scheduler')
    parser.add_argument('--load-teacher-model', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads only the model')
    # parser.add_argument('--use_restval', action='store_true',
    #                     help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--reinitialize-scheduler', action='store_true',
                        help='Reinitialize scheduler. To use with --resume')
    parser.add_argument('--config', type=str, help="Which configuration to use. See into 'config' folder")


    args = parser.parse_args()
    print(args)

    # torch.cuda.set_enabled_lms(True)
    # if (torch.cuda.get_enabled_lms()):
    #     torch.cuda.set_limit_lms(11000 * 1024 * 1024)
    #     print('[LMS=On limit=' + str(torch.cuda.get_limit_lms()) + ']')

    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)

    distillation_on = 'distillation' in config['training']['loss-type']
    args.per_gpu_train_batch_size = config['training']['bs']
    args.per_gpu_eval_batch_size = config['training']['bs']

    # Warn: these flags are misleading: they switch Oscar in the right configuration for the Alad setup (see dataset.py)
    args.do_test = True
    args.do_eval = True

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    global logger
    logger = setup_logger("vlpretrain", None, 0)

    tb_logger = SummaryWriter(log_dir=args.logger_name, comment='')

    # Train the Model
    loss_type = config['training']['loss-type']
    alignment_mode = config['training']['alignment-mode'] if 'alignment' in loss_type else None
    activate_distillation_after = config['training']['activate-distillation-after'] if 'activate-distillation-after' in config['training'] else 0

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

    train_dataset = RetrievalDataset(tokenizer, args, 'train', is_train=True)
    train_collate = MyCollate(dataset=train_dataset, return_oscar_data=distillation_on)
    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=config['training']['bs'], num_workers=args.num_workers, collate_fn=train_collate)

    val_dataset = RetrievalDataset(tokenizer, args, 'minival', is_train=True)
    val_collate = MyCollate(dataset=val_dataset, return_oscar_data=False)
    val_loader = DataLoader(val_dataset, shuffle=False,
                              batch_size=config['training']['bs'], num_workers=args.num_workers,
                              collate_fn=val_collate)

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
    model = ALADModel(config, oscar_checkpoint)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    # LR scheduler
    scheduler_name = config['training']['scheduler']
    if scheduler_name == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['training']['step-size'],
                                                    gamma=config['training']['gamma'])
    elif scheduler_name is None:
        scheduler = None
    else:
        raise ValueError('{} scheduler is not available'.format(scheduler_name))

    # Warmup scheduler
    warmup_scheduler_name = config['training']['warmup'] if not args.resume else None
    if warmup_scheduler_name == 'linear':
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=config['training']['warmup-period'])
    elif warmup_scheduler_name is None:
        warmup_scheduler = None
    else:
        raise ValueError('{} warmup scheduler is not available'.format(warmup_scheduler_name))

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume or args.load_teacher_model:
        filename = args.resume if args.resume else args.load_teacher_model
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            oscar_checkpoint = torch.load(filename, map_location='cpu')
            model.load_state_dict(oscar_checkpoint['model'], strict=False if args.load_teacher_model else True)
            if torch.cuda.is_available():
                model.cuda()
            print('Student model loaded!')
            if args.resume:
                start_epoch = oscar_checkpoint['epoch']
                # best_rsum = checkpoint['best_rsum']
                optimizer.load_state_dict(oscar_checkpoint['optimizer'])
                if oscar_checkpoint['scheduler'] is not None and not args.reinitialize_scheduler:
                    scheduler.load_state_dict(oscar_checkpoint['scheduler'])
                # Eiters is used to show logs as the continuation of another
                # training
                model.Eiters = oscar_checkpoint['Eiters']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, start_epoch))
            else:
                print("=> loaded only model from checkpoint '{}'"
                      .format(args.load_teacher_model))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(filename))

    for epoch in range(start_epoch, args.num_epochs):
        # train for one epoch
        train(args, config, train_loader, model, optimizer, epoch, tb_logger, val_loader, None,
              measure=config['training']['measure'], grad_clip=config['training']['grad-clip'],
              scheduler=scheduler, warmup_scheduler=warmup_scheduler, ndcg_val_scorer=ndcg_val_scorer,
              ndcg_test_scorer=None, alignment_mode=alignment_mode, loss_type=loss_type, distill_epoch=activate_distillation_after)

        # evaluate on validation set
        rsum, ndcg_sum = validate(val_loader, model, tb_logger, measure=config['training']['measure'],
                                  log_step=args.log_step,
                                  ndcg_scorer=ndcg_val_scorer, alignment_mode=alignment_mode, loss_type=loss_type)

        # remember best R@ sum and save checkpoint
        is_best_rsum = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)

        is_best_ndcg = False # ndcg_sum > best_ndcg_sum
        best_ndcg_sum = max(ndcg_sum, best_ndcg_sum)
        #
        # is_best_r1 = r1 > best_r1
        # best_r1 = max(r1, best_r1)

        # is_best_val_loss = val_loss < best_val_loss
        # best_val_loss = min(val_loss, best_val_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'opt': args,
            'config': config,
            'Eiters': model.Eiters,
        }, is_best_rsum, is_best_ndcg, prefix=args.logger_name + '/')


def get_teacher_scores(args, model, batch, subdivs=40):
    batch = tuple(t.to(args.device) for t in batch)
    side_size = int(math.sqrt(len(batch[0])))
    assert side_size**2 == len(batch[0])
    results = []
    attentions = []
    with torch.no_grad():
        # Iterate a number of iterations where each batch is composed of subdivs samples. This is for preventing memory explosion
        iterations = (len(batch[0]) // subdivs) + 1
        for it in range(iterations):
            b = it * subdivs
            if b == len(batch[0]):
                # prevent batch size == 0
                break
            e = (it + 1) * subdivs
            inputs = {
                'input_ids': batch[0][b:e],
                'attention_mask': batch[1][b:e],
                'token_type_ids': batch[2][b:e],
                'img_feats': batch[3][b:e],
                'labels': batch[4][b:e]
            }
            _, logits, attention = model(**inputs)[:3]
            if args.num_labels == 2:
                probs = F.softmax(logits)
                result = probs[:, 1]  # the confidence to be a matched pair
            else:
                result = logits

            # accumulate results
            results.append(result)

            # accumulate attentions
            attention = attention[-1].mean(
                dim=1)  # attentions from the last layer, mean over all the heads: B x (70+70) x (70+70)
            # take only the distribution of every text over the image regions
            attention = attention[:, 1:70, 70:]    # B x 69 x 70
            attentions.append(attention)

        results = torch.cat(results, dim=0)
        scores = results.view(side_size, side_size) # B x B matrix of scores

        attentions = torch.cat(attentions, dim=0)
        attentions = attentions.view(side_size, side_size, attentions.shape[1], attentions.shape[2])
        return scores, attentions
        # result = [_.to(torch.device("cpu")) for _ in result]


def train(opt, config, train_loader, model, optimizer, epoch, tb_logger, val_loader, test_loader,
          measure='cosine', grad_clip=-1, scheduler=None, warmup_scheduler=None, ndcg_val_scorer=None,
          ndcg_test_scorer=None, alignment_mode=None, loss_type=None, distill_epoch=1):
    global best_rsum

    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        model.train()
        if scheduler is not None:
            scheduler.step(epoch)

        if warmup_scheduler is not None:
            warmup_scheduler.dampen()

        optimizer.zero_grad()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        example_imgs, example_txts = train_data
        loss, loss_dict = model(example_imgs, example_txts, epoch=epoch, distill_epoch=distill_epoch)
        # loss = sum(loss for loss in loss_dict.values())

        # compute gradient and do SGD step
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.add_scalar('epoch', epoch, model.Eiters)
        tb_logger.add_scalar('step', i, model.Eiters)
        tb_logger.add_scalar('batch_time', batch_time.val, model.Eiters)
        tb_logger.add_scalar('data_time', data_time.val, model.Eiters)
        tb_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        del train_data
        if i % 10 == 0:
            torch.cuda.empty_cache()

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            rsum, _ = validate(val_loader, model, tb_logger, measure=measure, log_step=opt.log_step, ndcg_scorer=ndcg_val_scorer, alignment_mode=alignment_mode, loss_type=loss_type)

            is_best_rsum = rsum > best_rsum
            best_rsum = max(rsum, best_rsum)

            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'opt': opt,
                'config': config,
                'Eiters': model.Eiters,
            }, is_best_rsum, False, prefix=opt.logger_name + '/')

        # if model.Eiters % opt.test_step == 0:
        #     test(test_loader, model, tb_logger, measure=measure, log_step=opt.log_step, ndcg_scorer=ndcg_test_scorer)


def validate(val_loader, model, tb_logger, measure='cosine', log_step=10, ndcg_scorer=None, alignment_mode=None, loss_type=None):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, img_lenghts, cap_lenghts = encode_data(
        model, val_loader, log_step, logging.info)

    print('Evaluating matching head...')
    r1_match, r5_match, r10_match, r1i_match, r5i_match, r10i_match, rsum_match = compute_recall(img_embs[:, 0, :], cap_embs[:, 0, :])
    rsum = rsum_match

    # record matching metrics in tensorboard
    tb_logger.add_scalar('matching/r1', r1_match, model.Eiters)
    tb_logger.add_scalar('matching/r5', r5_match, model.Eiters)
    tb_logger.add_scalar('matching/r10', r10_match, model.Eiters)
    tb_logger.add_scalar('matching/r1i', r1i_match, model.Eiters)
    tb_logger.add_scalar('matching/r5i', r5i_match, model.Eiters)
    tb_logger.add_scalar('matching/r10i', r10i_match, model.Eiters)
    tb_logger.add_scalar('matching/rsum', rsum_match, model.Eiters)

    # initialize similarity matrix evaluator
    if 'alignment' in loss_type:
        sim_matrix_class = AlignmentContrastiveLoss(aggregation=alignment_mode)

        def alignment_sim_fn(img, cap, img_len, cap_len):
            with torch.no_grad():
                scores = sim_matrix_class(img, cap, img_len, cap_len, return_loss=False, return_similarity_mat=True)
            return scores

        sim_matrix_fn = alignment_sim_fn

        print('Evaluating alignment head...')
        # caption retrieval
        (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg) = i2t(img_embs, cap_embs, img_lenghts, cap_lenghts, measure=measure, ndcg_scorer=ndcg_scorer, sim_function=sim_matrix_fn, cap_batches=5)
        logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, ndcg_rouge=%.4f ndcg_spice=%.4f" %
                     (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg))
        # image retrieval
        (r1i, r5i, r10i, medri, meanr, mean_rougel_ndcg_i, mean_spice_ndcg_i) = t2i(
            img_embs, cap_embs, img_lenghts, cap_lenghts, ndcg_scorer=ndcg_scorer, measure=measure, sim_function=sim_matrix_fn, im_batches=1)

        logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, ndcg_rouge=%.4f ndcg_spice=%.4f" %
                     (r1i, r5i, r10i, medri, meanr, mean_rougel_ndcg_i, mean_spice_ndcg_i))
        # sum of recalls to be used for early stopping
        rsum_align = r1 + r5 + r10 + r1i + r5i + r10i
        # spice_ndcg_sum = mean_spice_ndcg + mean_spice_ndcg_i

        # record alignment metrics in tensorboard
        tb_logger.add_scalar('alignment/r1', r1, model.Eiters)
        tb_logger.add_scalar('alignment/r5', r5, model.Eiters)
        tb_logger.add_scalar('alignment/r10', r10, model.Eiters)
        tb_logger.add_scalar('alignment/r1i', r1i, model.Eiters)
        tb_logger.add_scalar('alignment/r5i', r5i, model.Eiters)
        tb_logger.add_scalar('alignment/r10i', r10i, model.Eiters)
        tb_logger.add_scalar('alignment/medr', medr, model.Eiters)
        tb_logger.add_scalar('alignment/meanr', meanr, model.Eiters)
        tb_logger.add_scalar('alignment/medri', medri, model.Eiters)
        tb_logger.add_scalar('alignment/meanr', meanr, model.Eiters)
        tb_logger.add_scalar('rsum', rsum_align, model.Eiters)

        rsum += rsum_align

    return rsum, 0

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

def save_checkpoint(state, is_best_rsum, is_best_ndcg, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best_rsum:
        shutil.copyfile(prefix + filename, prefix + 'model_best_rsum.pth.tar')
    if is_best_ndcg:
        shutil.copyfile(prefix + filename, prefix + 'model_best_ndcgspice.pth.tar')

if __name__ == "__main__":
    main()