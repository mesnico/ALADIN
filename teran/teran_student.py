import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from teran.loss import ContrastiveLoss, AlignmentContrastiveLoss, DistillationLoss, \
    AttentionDistillationLoss
from oscar.modeling.modeling_bert import ImageBertForSequenceClassification
from transformers.pytorch_transformers import BertTokenizer, BertConfig

from .utils import l2norm,  Aggregator, DepthAggregatorModel
# from nltk.corpus import stopwords, words as nltk_words


def pairwise_NNs_inner(x):
    """
    Pairwise nearest neighbors for L2-normalized vectors.
    Uses Torch rather than Faiss to remain on GPU.
    """
    # parwise dot products (= inverse distance)
    dots = torch.mm(x, x.t())
    n = x.shape[0]
    dots.view(-1)[::(n+1)].fill_(-1)  # Trick to fill diagonal with -1
    _, I = torch.max(dots, 1)  # max inner prod -> min distance
    return I

class JointTextImageTransformerEncoder(nn.Module):
    """
    This is a bert caption encoder - transformer image encoder (using bottomup features).
    If process the encoder outputs through a transformer, like VilBERT and outputs two different graph embeddings
    """
    def __init__(self, config, oscar_checkpoint):
        super().__init__()

        # Init OSCAR
        config_class = BertConfig
        model_class = ImageBertForSequenceClassification
        bert_config = config_class.from_pretrained(oscar_checkpoint)
        bert_config.output_attentions = True
        bert_config.output_hidden_states = True
        self.oscar_model = model_class.from_pretrained(oscar_checkpoint, config=bert_config)


        # self.txt_enc = EncoderText(config)
        # self.oscar_model = oscar_model
        visual_feat_dim = config['image-model']['feat-dim']
        caption_feat_dim = config['text-model']['word-dim']
        dropout = config['model']['dropout']
        teran_layers = config['model']['teran-layers']
        tern_layers = config['model']['tern-layers']
        embed_size = config['model']['embed-size']
        self.order_embeddings = config['training']['measure'] == 'order'
        # self.img_enc = EncoderImage(config)

        hidden_size = 768
        self.img_proj = nn.Linear(hidden_size, embed_size)
        self.cap_proj = nn.Linear(hidden_size, embed_size)
        self.embed_size = embed_size
        self.shared_transformer = config['model']['shared-transformer']
        self.depth_aggregation = config['model']['depth-aggregation'] if 'depth-aggregation' in config['model'] else False
        if self.depth_aggregation:
            self.depth_aggregator_model = DepthAggregatorModel(config, input_dim=embed_size)
        self.loss_type = config['training']['loss-type']
        self.text_aggregation_type = config['model']['text-aggregation']
        self.img_aggregation_type = config['model']['image-aggregation']

        if teran_layers == 0:
            self.text_aggregation_type = None
            self.img_aggregation_type = None

        if self.text_aggregation_type is not None:
            transformer_layer_1 = nn.TransformerEncoderLayer(d_model=embed_size, nhead=4,
                                                           dim_feedforward=embed_size,
                                                           dropout=dropout)
            self.transformer_encoder_1 = nn.TransformerEncoder(transformer_layer_1,
                                                               num_layers=teran_layers)
        if self.img_aggregation_type is not None:
            if not self.shared_transformer:
                transformer_layer_2 = nn.TransformerEncoderLayer(d_model=embed_size, nhead=4,
                                                                 dim_feedforward=embed_size,
                                                                 dropout=dropout)
                self.transformer_encoder_2 = nn.TransformerEncoder(transformer_layer_2,
                                                                   num_layers=teran_layers)
        if self.depth_aggregation is not None and self.depth_aggregation == 'transformer':
            depth_transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size, dropout=dropout)
            self.depth_transformer = nn.TransformerEncoder(depth_transformer_layer, num_layers=1)

        self.text_aggregation = Aggregator(embed_size, aggregation_type=config['model']['text-aggregation'])
        self.image_aggregation = Aggregator(embed_size, aggregation_type=config['model']['image-aggregation'])
        if 'distillation' in config['training']['loss-type']:
            tern_transformer_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=4,
                                                             dim_feedforward=embed_size,
                                                             dropout=dropout)
            self.final_projection_net = nn.TransformerEncoder(tern_transformer_layer,
                                                               num_layers=tern_layers)

            # self.final_projection_net = nn.Sequential(
            #     nn.Linear(embed_size, embed_size),
            #     nn.Dropout(0.1),
            #     nn.ReLU(),
            #     nn.Linear(embed_size, embed_size)
            # )
        else:
            self.final_projection_net = None

        self.l1_regularization = 'regularizehidden' in config['training']['loss-type']

    def forward(self, examples_imgs, examples_txts):
        # process captions by using oscar
        inputs_txts = {
            'input_ids': examples_txts[0],
            'attention_mask': examples_txts[1],
            'token_type_ids': examples_txts[2],
            'img_feats': None
        }
        txt_bert_output = self.oscar_model.bert(**inputs_txts)

        # process image regions using oscar
        inputs_imgs = {
            'input_ids': examples_imgs[0],
            'attention_mask': examples_imgs[1],
            'token_type_ids': examples_imgs[2],
            'img_feats': examples_imgs[3],
        }
        img_bert_output = self.oscar_model.bert(**inputs_imgs)
        # i_emb = i_emb.permute(1, 0, 2)

        bs = inputs_imgs['img_feats'].shape[0]

        cap_len = examples_txts[4]
        feat_len = examples_imgs[5]

        max_language_token_len = inputs_txts['input_ids'].shape[1]
        max_cap_len = max(cap_len)
        max_img_len = max(feat_len)

        # masks
        txt_mask = torch.zeros(bs, max(cap_len)).bool()
        txt_mask = txt_mask.to(inputs_txts['input_ids'].device)
        for m, c_len in zip(txt_mask, cap_len):
            m[c_len:] = True

        img_mask = torch.zeros(bs, max(feat_len)).bool()
        img_mask = img_mask.to(inputs_imgs['img_feats'].device)
        for m, v_len in zip(img_mask, feat_len):
            m[v_len:] = True

        if self.depth_aggregation:
            hidden_states_img = torch.stack(img_bert_output[2], dim=0)   # depth x B x N x dim
            hidden_states_img = hidden_states_img[:, :, max_language_token_len:max_language_token_len + max_img_len, :]
            i_emb = self.depth_aggregator_model(hidden_states_img, img_mask).permute(1, 0, 2)    # S x B x dim

            hidden_states_txt = torch.stack(txt_bert_output[2], dim=0)
            hidden_states_txt = hidden_states_txt[:, :, :max_cap_len, :]
            c_emb = self.depth_aggregator_model(hidden_states_txt, txt_mask).permute(1, 0, 2)   # S x B x dim

        else:
            c_emb = txt_bert_output[0][:, :max_cap_len].permute(1, 0, 2)
            i_emb = img_bert_output[0][:, max_language_token_len:max_language_token_len + max_img_len].permute(1, 0, 2)

        # forward the captions
        if self.text_aggregation_type is not None:
            # c_emb = self.cap_proj(c_emb)

            set_caption_embeddings = self.transformer_encoder_1(c_emb, src_key_padding_mask=txt_mask)  # S_txt x B x dim
            # full_cap_emb_aggr = self.text_aggregation(full_cap_emb, cap_len, mask)
        # else use the embedding output by the txt model
        else:
            set_caption_embeddings = c_emb

        # forward the regions
        if self.img_aggregation_type is not None:
            # i_emb = self.img_proj(i_emb)

            if self.shared_transformer:
                set_image_embeddings = self.transformer_encoder_1(i_emb, src_key_padding_mask=img_mask)  # S_txt x B x dim
            else:
                set_image_embeddings = self.transformer_encoder_2(i_emb, src_key_padding_mask=img_mask)  # S_txt x B x dim
            # full_img_emb_aggr = self.image_aggregation(full_img_emb, feat_len, mask)
        else:
            set_image_embeddings = i_emb

        if self.l1_regularization:
            # compute L1 regularization losses on the hidden states
            l1_regul_imgs = hidden_states_img.norm(p=1, dim=3).mean()
            l1_regul_txts = hidden_states_txt.norm(p=1, dim=3).mean()
            l1_regul_loss = (l1_regul_imgs + l1_regul_txts) / 2
            l1_regul_loss *= 0.001
        else:
            l1_regul_loss = 0

        # cross_attention_image, cross_attention_caption = set_image_embeddings[0], set_caption_embeddings[0] # self.self_aggregation(img_emb_set, cap_emb_seq, feat_len, cap_len)
        if self.final_projection_net:
            cross_attention_caption = self.final_projection_net(c_emb, src_key_padding_mask=txt_mask)[0]
            cross_attention_image = self.final_projection_net(i_emb, src_key_padding_mask=img_mask)[0]
        else:
            cross_attention_image, cross_attention_caption = set_image_embeddings[0], set_caption_embeddings[0]
        # normalize every vector of the set and the self-aggregated vectors
        set_image_embeddings = F.normalize(set_image_embeddings, p=2, dim=2)
        set_caption_embeddings = F.normalize(set_caption_embeddings, p=2, dim=2)

        cross_attention_caption = l2norm(cross_attention_caption)
        cross_attention_image = l2norm(cross_attention_image)

        # if self.order_embeddings:
        #     cross_attention_caption = torch.abs(cross_attention_caption)
        #     cross_attention_image = torch.abs(cross_attention_image)

        return cross_attention_image, cross_attention_caption, set_image_embeddings, set_caption_embeddings, feat_len, cap_len, l1_regul_loss


class TERANStudent(torch.nn.Module):
    """
    rkiros/uvs model
    """

    def __init__(self, config, oscar_checkpoint):
        # tutorials/09 - Image Captioning
        # Build Models
        super().__init__()
        self.img_txt_enc = JointTextImageTransformerEncoder(config, oscar_checkpoint)
        if torch.cuda.is_available():
            self.img_txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.losses_types = config['training']['loss-type'].split('-')
        self.losses_weights = config['training']['loss-weights']
        if isinstance(self.losses_weights, list):
            assert len(self.losses_types) == len(self.losses_weights)
            self.losses_weights = {k: v for k, v in zip(self.losses_types, self.losses_weights)}
            self.auto_weight = False
        else:
            self.losses_weights = {k: nn.Parameter(-2.3 * torch.ones(1)).cuda() if torch.cuda.is_available() else nn.Parameter(-2.3 * torch.ones(1)) for k in self.losses_types}
            self.auto_weight = True

        if 'distillation' in self.losses_types:
            self.distillation_loss = DistillationLoss(mode=config['training']['distillation-mode'])

        if 'attdistillation' in self.losses_types:
            self.att_distillation_loss = AttentionDistillationLoss()

        # if 'crossattention-all2all' in loss_type:
        #     self.cross_attention_aggregation_all2all = CrossAttentionAggregationAll2All(d_model=config['model']['embed-size'],
        #                                                                  feedforward_dim=config['model']['embed-size'])

        if 'alignment' in self.losses_types:
            self.alignment_criterion = AlignmentContrastiveLoss(margin=config['training']['margin'],
                                                                measure=config['training']['measure'],
                                                                max_violation=config['training']['max-violation'], aggregation=config['training']['alignment-mode'])
        if 'matching' or 'selfaggregation' in self.losses_types:
            self.matching_criterion = ContrastiveLoss(margin=config['training']['margin'],
                                                      measure=config['training']['measure'],
                                                      max_violation=config['training']['max-violation'])

        self.Eiters = 0
        self.config = config

        if 'exclude-stopwords' in config['model'] and config['model']['exclude-stopwords']:
            self.en_stops = set(stopwords.words('english'))
            self.tokenizer = BertTokenizer.from_pretrained(config['text-model']['pretrain'])
        else:
            self.tokenizer = None

        self.pdist = nn.PairwiseDistance(2)

    # def state_dict(self):
    #     state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
    #     return state_dict
    #
    # def load_state_dict(self, state_dict):
    #     self.img_enc.load_state_dict(state_dict[0])
    #     self.txt_enc.load_state_dict(state_dict[1])
    #
    # def train_start(self):
    #     """switch to train mode
    #     """
    #     self.img_enc.train()
    #     self.txt_enc.train()
    #
    # def val_start(self):
    #     """switch to evaluate mode
    #     """
    #     self.img_enc.eval()
    #     self.txt_enc.eval()

    def forward_emb(self, example_imgs, example_txts):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            example_imgs = [c.cuda() if isinstance(c, torch.Tensor) else c for c in example_imgs]
            example_txts = [c.cuda() if isinstance(c, torch.Tensor) else c for c in example_txts]

        # Forward
        img_emb_aggr, cap_emb_aggr, img_feats, cap_feats, img_len, cap_len, regul_loss = self.img_txt_enc(example_imgs, example_txts)

        if self.tokenizer is not None:
            # remove stopwords
            # keep only word indexes that are not stopwords
            good_word_indexes = [[i for i, (tok, w) in enumerate(zip(self.tokenizer.convert_ids_to_tokens(ids), ids)) if
                                  tok not in self.en_stops or w == 0] for ids in captions]  # keeps the padding
            cap_len = [len(w) - (cap_feats.shape[0] - orig_len) for w, orig_len in zip(good_word_indexes, cap_len)]
            min_cut_len = min([len(w) for w in good_word_indexes])
            good_word_indexes = [words[:min_cut_len] for words in good_word_indexes]
            good_word_indexes = torch.LongTensor(good_word_indexes).to(cap_feats.device) # B x S
            good_word_indexes = good_word_indexes.t().unsqueeze(2).expand(-1, -1, cap_feats.shape[2]) # S x B x dim
            cap_feats = cap_feats.gather(dim=0, index=good_word_indexes)

        return img_emb_aggr, cap_emb_aggr, img_feats, cap_feats, img_len, cap_len, regul_loss

    # def get_parameters(self):
    #     lr_multiplier = 1.0 if self.config['text-model']['fine-tune'] else 0.0
    #
    #     ret = []
    #     params = list(self.img_txt_enc.img_enc.parameters())
    #     params += list(self.img_txt_enc.img_proj.parameters())
    #     params += list(self.img_txt_enc.cap_proj.parameters())
    #     params += list(self.img_txt_enc.transformer_encoder_1.parameters())
    #
    #     params += list(self.img_txt_enc.image_aggregation.parameters())
    #     params += list(self.img_txt_enc.text_aggregation.parameters())
    #
    #     if not self.config['model']['shared-transformer']:
    #         params += list(self.img_txt_enc.transformer_encoder_2.parameters())
    #
    #     ret.append(params)
    #
    #     ret.append(list(self.img_txt_enc.txt_enc.parameters()))
    #
    #     return ret, lr_multiplier

    def forward_loss(self, img_emb, cap_emb, img_emb_set, cap_emb_seq, img_lengths, cap_lengths, reg_loss):
        """Compute the loss given pairs of image and caption embeddings
        """
        # bs = img_emb.shape[0]
        losses = {}

        img_emb_set = img_emb_set.permute(1, 0, 2)
        cap_emb_seq = cap_emb_seq.permute(1, 0, 2)

        matching_loss, matching_mat = self.matching_criterion(img_emb, cap_emb, return_similarity_mat=True)
        if  'matching' in self.config['training']['loss-type']:
            losses.update({'matching': matching_loss})
            self.logger.update('matching_loss', matching_loss.item(), img_emb.size(0))

        if 'alignment' in self.losses_types:
            alignment_loss, teacher_scores = self.alignment_criterion(img_emb_set, cap_emb_seq, img_lengths, cap_lengths, return_similarity_mat=True)
            # alignment_loss *= self.losses_weights['alignment']
            losses.update({'alignment': alignment_loss})
            self.logger.update('alignment_loss', alignment_loss.item(), img_emb_set.size(0))

        # if 'crossattention-all2all' in self.config['training']['loss-type']:
        #     matching_loss = self.cross_attention_aggregation_all2all(img_emb_set, cap_emb_seq, img_lengths, cap_lengths)
        #     losses.update({'cross-attention-loss': matching_loss})
        #     self.logger.update('cross_attention_loss', matching_loss.item(), img_emb.size(0))

        if 'selfaggregation' in self.losses_types:
            # img_emb, cap_emb = self.cross_attention_aggregation(img_emb_set, cap_emb_seq, img_lengths, cap_lengths)
            matching_loss, matching_mat = self.matching_criterion(img_emb, cap_emb, return_similarity_mat=True)
            # matching_loss *= self.losses_weights['selfaggregation']
            losses.update({'selfaggregation': matching_loss})
            self.logger.update('self_attention_loss', matching_loss.item(), img_emb.size(0))

        if 'distillation' in self.losses_types:
            distillation_loss = self.distillation_loss(teacher_scores, matching_mat)
            # distillation_loss *= self.losses_weights['distillation']
            losses.update({'distillation': distillation_loss})
            self.logger.update('distillation_loss', distillation_loss.item(), img_emb.size(0))

        if 'attdistillation' in self.losses_types:
            att_distillation_loss = self.att_distillation_loss(img_emb_set, cap_emb_seq, img_lengths, cap_lengths, teacher_attentions)
            losses.update({'attdistillation': att_distillation_loss})
            self.logger.update('att_distillation_loss', att_distillation_loss.item(), img_emb.size(0))

        if 'entropy' in self.losses_types:
            # img_emb, cap_emb = self.cross_attention_aggregation(img_emb_set, cap_emb_seq, img_lengths, cap_lengths)
            # matching_loss = self.matching_criterion(img_emb, cap_emb)
            # losses.update({'cross-attention-loss': matching_loss})
            # self.logger.update('cross_attention_loss', matching_loss.item(), img_emb.size(0))
            # entropy loss
            all_emb = torch.cat([img_emb, cap_emb], 0)
            I = pairwise_NNs_inner(all_emb)
            distances = self.pdist(all_emb, all_emb[I])
            loss_uniform = - torch.log(all_emb.size(0) * distances).mean()
            losses.update({'entropy': loss_uniform})
            self.logger.update('entropy-uniform-loss', loss_uniform.item(), img_emb.size(0))

        if 'regularizehidden' in self.losses_types:
            losses.update({'regularizehidden': reg_loss})
            self.logger.update('regularize_hidden_loss', reg_loss.item(), img_emb.size(0))

        # self.logger.update('Le', matching_loss.item() + alignment_loss.item(), img_emb.size(0) if img_emb is not None else img_emb_set.size(1))
        return losses

    def forward(self, example_imgs, example_txts, epoch=0, distill_epoch=2):
        """One training step given images and captions.
        """
        # assert self.training()
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)

        # compute the embeddings
        img_emb_aggr, cap_emb_aggr, img_feats, cap_feats, img_lengths, cap_lengths, regul_loss = self.forward_emb(example_imgs, example_txts)
        # NOTE: img_feats and cap_feats are S x B x dim

        loss_dict = self.forward_loss(img_emb_aggr, cap_emb_aggr, img_feats, cap_feats, img_lengths, cap_lengths, regul_loss)
        if epoch < distill_epoch:
            #remove distillation loss
            loss_dict.pop('distillation', None)
        if self.auto_weight:
            loss = 0
            for k in loss_dict:
                loss += loss_dict[k] * torch.exp(-self.losses_weights[k]) + self.losses_weights[k]
            loss *= 0.5
        else:
            loss = 0
            for k in loss_dict:
                loss += loss_dict[k] * self.losses_weights[k]
        return loss, loss_dict
