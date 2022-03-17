from __future__ import absolute_import, division, print_function
import argparse
import itertools
import os
import base64
import os.path as op
import random, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from oscar.run_retrieval import restore_training_settings
from oscar.utils.tsv_file import TSVFile
from oscar.utils.logger import setup_logger
from oscar.utils.misc import mkdir, set_seed
from oscar.modeling.modeling_bert import ImageBertForSequenceClassification
from transformers.pytorch_transformers import BertTokenizer, BertConfig
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule

class RetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""

    def __init__(self, tokenizer, args, split='train', is_train=True):
        """
        tokenizer: tokenizer to process caption text.
        args: configureation parameters including max_seq_length, etc.
        split: used to infer the data used for training or testing.
             All files are in .pt format of a dictionary with image keys and
             image features (pytorch tensors), captions (list of str, support multiple
             captions per image), labels (list of dictionary or str of all labels),

        """
        super(RetrievalDataset, self).__init__()
        self.img_file = args.img_feat_file
        caption_file = op.join(args.data_dir, '{}_captions.pt'.format(split))
        self.img_tsv = TSVFile(self.img_file)
        self.captions = torch.load(caption_file)
        self.img_keys = list(self.captions.keys())  # img_id as int
        if not type(self.captions[self.img_keys[0]]) == list:
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}

        # get the image image_id to index map
        imgid2idx_file = op.join(op.dirname(self.img_file), 'imageid2idx.json')
        self.image_id2idx = json.load(open(imgid2idx_file))  # img_id as string

        if args.add_od_labels:
            label_data_dir = op.dirname(self.img_file)
            label_file = os.path.join(label_data_dir, "predictions.tsv")
            self.label_tsv = TSVFile(label_file)
            self.labels = {}
            for line_no in range(self.label_tsv.num_rows()):
                row = self.label_tsv.seek(line_no)
                image_id = row[0]
                if int(image_id) in self.img_keys:
                    results = json.loads(row[1])
                    objects = results['objects'] if type(
                        results) == dict else results
                    self.labels[int(image_id)] = {
                        "image_h": results["image_h"] if type(
                            results) == dict else 600,
                        "image_w": results["image_w"] if type(
                            results) == dict else 800,
                        "class": [cur_d['class'] for cur_d in objects],
                        "boxes": np.array([cur_d['rect'] for cur_d in objects],
                                          dtype=np.float32)
                    }
            self.label_tsv._fp.close()
            self.label_tsv._fp = None

        if is_train:
            self.num_captions_per_img = args.num_captions_per_img_train
        else:
            self.num_captions_per_img = args.num_captions_per_img_val
            if args.eval_img_keys_file:
                # select a subset of image keys for evaluation. eg. COCO 1k and 5k
                # eval_img_keys_file is a list of image keys saved in tsv file
                with open(op.join(args.data_dir, args.eval_img_keys_file), 'r') as f:
                    img_keys = f.readlines()
                self.img_keys = [int(k.strip()) for k in img_keys]
                self.captions = {k: self.captions[k] for k in self.img_keys}
                if args.add_od_labels:
                    self.labels = {k: self.labels[k] for k in self.img_keys}

            if args.eval_caption_index_file:
                # hard negative image/caption indexs for retrieval re-rank setting.
                # useful for mini val set to monitor the performance during training.
                # However, it cannot be used together with cross image evaluation.
                self.has_caption_indexs = True
                assert not args.cross_image_eval
                caption_index_file = op.join(args.data_dir, args.eval_caption_index_file)
                self.caption_indexs = torch.load(caption_index_file)
                if not type(self.caption_indexs[self.img_keys[0]]) == list:
                    self.caption_indexs = {k: json.loads(self.caption_indexs[k]) for k in self.img_keys}
            else:
                self.has_caption_indexs = False
        self.output_mode = args.output_mode
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_length
        self.max_img_seq_len = args.max_img_seq_length
        self.args = args

    def get_image_caption_index(self, index):
        # # return img_idx to access features and [img_key, cap_idx] to access caption
        # if not self.is_train and self.args.cross_image_eval:
        #     img_idx = index // (self.num_captions_per_img * len(self.img_keys))
        #     cap_idx = index % (self.num_captions_per_img * len(self.img_keys))
        #     img_idx1 = cap_idx // self.num_captions_per_img
        #     cap_idx1 = cap_idx % self.num_captions_per_img
        #     return img_idx, [self.img_keys[img_idx1], cap_idx1]
        # if not self.is_train and self.has_caption_indexs:
        #     img_idx = index // self.num_captions_per_img
        #     cap_idx = index % self.num_captions_per_img
        #     img_key1, cap_idx1 = self.caption_indexs[self.img_keys[img_idx]][cap_idx]
        #     return img_idx, [img_key1, cap_idx1]
        img_idx = index // self.num_captions_per_img
        cap_idx = index % self.num_captions_per_img
        return img_idx, [self.img_keys[img_idx], cap_idx]

    def get_label(self, index):
        img_idx, cap_idx = self.get_image_caption_index(index)
        return 1 if self.img_keys[img_idx] == cap_idx[0] else 0

    def get_od_labels(self, img_key):
        if self.args.add_od_labels:
            if type(self.labels[img_key]) == str:
                od_labels = self.labels[img_key]
            else:
                od_labels = ' '.join(self.labels[img_key]['class'])
            return od_labels

    def tensorize_example(self, text_a, img_feat, text_b=None,
                          cls_token_segment_id=0, pad_token_segment_id=0,
                          sequence_a_segment_id=0, sequence_b_segment_id=1, return_lengths=False):
        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.args.max_seq_length - 2:
            tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
        seq_a_len = len(tokens)
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        seq_padding_len = self.max_seq_len - seq_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0: self.max_img_seq_len, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # generate attention_mask
        att_mask_type = self.args.att_mask_type
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len + [0] * seq_padding_len + \
                             [1] * img_len + [0] * img_padding_len
        else:
            # use 2D mask to represent the attention
            max_len = self.max_seq_len + self.max_img_seq_len
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention of C-C, L-L, R-R
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
            attention_mask[c_start: c_end, c_start: c_end] = 1
            attention_mask[l_start: l_end, l_start: l_end] = 1
            attention_mask[r_start: r_end, r_start: r_end] = 1
            if att_mask_type == 'CL':
                attention_mask[c_start: c_end, l_start: l_end] = 1
                attention_mask[l_start: l_end, c_start: c_end] = 1
            elif att_mask_type == 'CR':
                attention_mask[c_start: c_end, r_start: r_end] = 1
                attention_mask[r_start: r_end, c_start: c_end] = 1
            elif att_mask_type == 'LR':
                attention_mask[l_start: l_end, r_start: r_end] = 1
                attention_mask[r_start: r_end, l_start: l_end] = 1
            else:
                raise ValueError("Unsupported attention mask type {}".format(att_mask_type))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        if return_lengths:
            return input_ids, attention_mask, segment_ids, img_feat, seq_a_len, img_len
        else:
            return input_ids, attention_mask, segment_ids, img_feat

    def tensorize_example_disentangled(self, text_a, img_feat, text_b=None,
                          cls_token_segment_id=0, pad_token_segment_id=0,
                          sequence_a_segment_id=0, sequence_b_segment_id=1, return_lengths=False):
        assert (text_a is None and img_feat is not None and text_b is not None) or (text_a is not None and img_feat is None and text_b is None)
        if text_a is not None:
            tokens_a = self.tokenizer.tokenize(text_a)
            if len(tokens_a) > self.args.max_seq_length - 2:
                tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

            tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
            segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
            seq_len = len(tokens)
        if text_b is not None:
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - 2:
                tokens_b = tokens_b[: (self.max_seq_len - 2)]
            tokens = [cls_token_segment_id] + tokens_b + [self.tokenizer.sep_token]
            segment_ids = [cls_token_segment_id] + [sequence_b_segment_id] * (len(tokens_b) + 1)
            seq_len = len(tokens)

        seq_padding_len = self.max_seq_len - seq_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        if img_feat is not None:
            # image features
            img_len = img_feat.shape[0]
            if img_len > self.max_img_seq_len:
                img_feat = img_feat[0: self.max_img_seq_len, :]
                img_len = img_feat.shape[0]
                img_padding_len = 0
            else:
                img_padding_len = self.max_img_seq_len - img_len
                padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
                img_feat = torch.cat((img_feat, padding_matrix), 0)
        else:
            img_len = None
            img_padding_len = None

        # generate attention_mask
        att_mask_type = self.args.att_mask_type
        if att_mask_type == "CLR":
            if img_feat is not None:
                attention_mask = [1] * seq_len + [0] * seq_padding_len + \
                                 [1] * img_len + [0] * img_padding_len
            else:
                attention_mask = [1] * seq_len + [0] * seq_padding_len
        else:
            # use 2D mask to represent the attention
            max_len = self.max_seq_len + self.max_img_seq_len
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention of C-C, L-L, R-R
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
            attention_mask[c_start: c_end, c_start: c_end] = 1
            attention_mask[l_start: l_end, l_start: l_end] = 1
            attention_mask[r_start: r_end, r_start: r_end] = 1
            if att_mask_type == 'CL':
                attention_mask[c_start: c_end, l_start: l_end] = 1
                attention_mask[l_start: l_end, c_start: c_end] = 1
            elif att_mask_type == 'CR':
                attention_mask[c_start: c_end, r_start: r_end] = 1
                attention_mask[r_start: r_end, c_start: c_end] = 1
            elif att_mask_type == 'LR':
                attention_mask[l_start: l_end, r_start: r_end] = 1
                attention_mask[r_start: r_end, l_start: l_end] = 1
            else:
                raise ValueError("Unsupported attention mask type {}".format(att_mask_type))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        if return_lengths:
            return input_ids, attention_mask, segment_ids, img_feat, seq_len, img_len
        else:
            return input_ids, attention_mask, segment_ids, img_feat

    def __getitem__(self, index):
        # if self.is_train:
        #     img_idx, cap_idxs = self.get_image_caption_index(index)
        #     img_key = self.img_keys[img_idx]
        #     feature = self.get_image(img_key)
        #     caption = self.captions[cap_idxs[0]][cap_idxs[1]]
        #     od_labels = self.get_od_labels(img_key)
        #     example = self.tensorize_example(caption, feature, text_b=od_labels)
        #
        #     # select a negative pair
        #     neg_img_indexs = list(range(0, img_idx)) + list(range(img_idx + 1, len(self.img_keys)))
        #     img_idx_neg = random.choice(neg_img_indexs)
        #     if random.random() <= 0.5:
        #         # randomly select a negative caption from a different image.
        #         cap_idx_neg = random.randint(0, self.num_captions_per_img - 1)
        #         caption_neg = self.captions[self.img_keys[img_idx_neg]][cap_idx_neg]
        #         example_neg = self.tensorize_example(caption_neg, feature, text_b=od_labels)
        #     else:
        #         # randomly select a negative image
        #         feature_neg = self.get_image(self.img_keys[img_idx_neg])
        #         od_labels_neg = self.get_od_labels(self.img_keys[img_idx_neg])
        #         example_neg = self.tensorize_example(caption, feature_neg, text_b=od_labels_neg)
        #
        #     example_pair = tuple(list(example) + [1] + list(example_neg) + [0])
        #     return index, example_pair
        # else:
        img_idx, cap_idxs = self.get_image_caption_index(index)
        img_key = self.img_keys[img_idx]
        im_feature = self.get_image(img_key)
        caption = self.captions[cap_idxs[0]][cap_idxs[1]]
        od_labels = self.get_od_labels(img_key)
        # example = self.tensorize_example(caption, feature, text_b=od_labels)
        label = 1 if img_key == cap_idxs[0] else 0
        return index, caption, im_feature, od_labels, label #tuple(list(example) + [label])

    def get_image(self, image_id):
        image_idx = self.image_id2idx[str(image_id)]
        row = self.img_tsv.seek(image_idx)
        num_boxes = int(row[1])
        features = np.frombuffer(base64.b64decode(row[-1]),
                                 dtype=np.float32).reshape((num_boxes, -1))
        t_features = torch.from_numpy(features)
        return t_features

    def __len__(self):
        return len(self.img_keys) * self.num_captions_per_img


# TODO: collate. It should return a list of all possible i-t couples inside the batch (for Oscar), as well as list of i and list of t for teran.
# To return all i-t couples, perform a itertools.product over i and t and call self.tensorize_example. It should come out with 64*64 i-t tuples composed of (input_ids, attention_mask, segment_ids, img_feat)
class MyCollate:
    def __init__(self, dataset, return_oscar_data=False):
        self.dataset = dataset
        self.return_oscar_data = return_oscar_data

    def __call__(self, batch):
        index, caption, im_feature, od_labels, label = zip(*batch)

        # prepare data for oscar
        # if self.return_oscar_data:
        #     examples = []
        #     bs = len(batch)
        #     for i, j in itertools.product(range(bs), repeat=2):
        #         example = self.dataset.tensorize_example(caption[j], im_feature[i], text_b=od_labels[i], return_lengths=False)
        #         label = torch.LongTensor([1 if i == j else 0])
        #         examples.append(tuple(list(example) + [label]))
        #     oscar_data = [torch.stack(t) for t in zip(*examples)]
        #     oscar_index = None # TODO: how to prepare index?
        # else:
        #     oscar_index = None
        #     oscar_data = None

        # prepare data for teran
        examples_text = [self.dataset.tensorize_example_disentangled(text_a=c, img_feat=None, text_b=None, return_lengths=True) for c in caption]
        examples_imgs = [self.dataset.tensorize_example_disentangled(text_a=None, img_feat=im_f, text_b=im_c, return_lengths=True) for im_f, im_c in zip(im_feature, od_labels)]

        examples_text = [torch.stack(t) if isinstance(t[0], torch.Tensor) else t for t in zip(*examples_text)]
        examples_imgs = [torch.stack(t) if isinstance(t[0], torch.Tensor) else t for t in zip(*examples_imgs)]

        return examples_imgs, examples_text



def restore_training_settings(args):
    assert not args.do_train and (args.do_test or args.do_eval)
    train_args = torch.load(op.join(args.eval_model_dir, 'training_args.bin'))
    override_params = ['do_lower_case', 'img_feature_type', 'max_seq_length',
            'max_img_seq_length', 'add_od_labels', 'od_label_type',
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

def main():
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
    args = parser.parse_args()

    global logger
    logger = setup_logger("vlpretrain", None, 0)

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed, args.n_gpu)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    logger.info('output_mode: {}, #Labels: {}'.format(args.output_mode, args.num_labels))

    config_class, tokenizer_class = BertConfig, BertTokenizer
    model_class = ImageBertForSequenceClassification
    # if args.do_train:
    #     config = config_class.from_pretrained(args.config_name if args.config_name else \
    #                                               args.model_name_or_path, num_labels=args.num_labels,
    #                                           finetuning_task='ir')
    #     tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
    #                                                     else args.model_name_or_path, do_lower_case=args.do_lower_case)
    #     config.img_feature_dim = args.img_feature_dim
    #     config.img_feature_type = args.img_feature_type
    #     config.hidden_dropout_prob = args.drop_out
    #     config.loss_type = args.loss_type
    #     config.img_layer_norm_eps = args.img_layer_norm_eps
    #     config.use_img_layernorm = args.use_img_layernorm
    #     model = model_class.from_pretrained(args.model_name_or_path,
    #                                         from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    # else:
    checkpoint = args.eval_model_dir
    assert op.isdir(checkpoint)
    # config = config_class.from_pretrained(checkpoint)
    tokenizer = tokenizer_class.from_pretrained(checkpoint)
    logger.info("Evaluate the following checkpoint: %s", checkpoint)
    # model = model_class.from_pretrained(checkpoint, config=config)

    # model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # if args.do_train:
    #     train_dataset = RetrievalDataset(tokenizer, args, 'train', is_train=True)
    #     if args.evaluate_during_training:
    #         val_dataset = RetrievalDataset(tokenizer, args, 'minival', is_train=False)
    #     else:
    #         val_dataset = None
    #     global_step, avg_loss = train(args, train_dataset, val_dataset, model, tokenizer)
    #     logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)

    # inference and evaluation
    if True: #args.do_test or args.do_eval:
        args = restore_training_settings(args)
        dataset = RetrievalDataset(tokenizer, args, 'train', is_train=True)
        #checkpoint = args.eval_model_dir
        #assert op.isdir(checkpoint)
        #logger.info("Evaluate the following checkpoint: %s", checkpoint)
        #model = model_class.from_pretrained(checkpoint, config=config)
        #model.to(args.device)
        #if args.n_gpu > 1:
        #    model = torch.nn.DataParallel(model)

        collate = MyCollate(dataset=dataset)
        bs = 40
        dataloader = DataLoader(dataset, shuffle=False,
                                     batch_size=bs, num_workers=args.num_workers, collate_fn=collate)
        for b in tqdm(dataloader):
            a = 1

        # pred_file = get_predict_file(args)
        # if False:  # op.isfile(pred_file):
        #     logger.info("Prediction file exist, skip inference.")
        #     if args.do_eval:
        #         test_result = torch.load(pred_file)
        # else:
        #     test_result = test(args, model, test_dataset)
        #     torch.save(test_result, pred_file)
        #     logger.info("Prediction results saved to {}.".format(pred_file))
        #
        # if args.do_eval:
        #     eval_result = evaluate(test_dataset, test_result)
        #     result_file = op.splitext(pred_file)[0] + '.eval.json'
        #     with open(result_file, 'w') as f:
        #         json.dump(eval_result, f)
        #     logger.info("Evaluation results saved to {}.".format(result_file))


if __name__ == "__main__":
    main()
