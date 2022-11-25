import json
import tqdm
import collections
import itertools
import os
from alad.extraction.image_retrieval import ImageRetrieval, QueryEncoder
os.environ['CUDA_VISIBLE_DEVICES'] = ""

vbs2019_gt_filename = 'alad/extraction/v3c_gt_evaluation/vbs2019_groundtruth.json'
feat_db = '/media/nicola/SSD/VBS_Features/aladin_v3c1_features.h5'
alad_args = '--data_dir /media/nicola/SSD/OSCAR_Datasets/coco_ir --img_feat_file /media/nicola/Data/Workspace/OSCAR/scene_graph_benchmark/output/X152C5_test/inference/vinvl_vg_x152c4/predictions.tsv --eval_model_dir /media/nicola/SSD/OSCAR_Datasets/checkpoint-0132780 --max_seq_length 50 --max_img_seq_length 34 --load_checkpoint /media/nicola/Data/Workspace/OSCAR/Oscar/alad/runs/alad-alignment-and-distill/model_best_rsum.pth.tar'

out_file = 'alad/extraction/gt_evaluation_results.json'

sub_K = 50000
sq_factor = 1000
# mode = 'multiquery-sum-scores' # multiquery

def compute_rank(ref_img_ids, img_ids):
    rank = 1e20
    img_ids = list(img_ids)
    for ref_id in ref_img_ids:
        try:
            id = img_ids.index(ref_id)
        except ValueError:
            id = rank

        if id < rank:
            rank = id
    return rank

def evaluate(mode, sq_thr=None):
    ranks = []
    with open(vbs2019_gt_filename, 'r') as f:
        vbs2019_gt = json.load(f)
    vbs2019_gt = vbs2019_gt["textualKISList"]
    ir = ImageRetrieval(feat_db)
    qe = QueryEncoder(alad_args)

    enable_scalar_quantization = sq_thr is not None

    for qid, query in enumerate(tqdm.tqdm(vbs2019_gt)):
        ref_img_ids = [k['keyframe'].replace('.png', '').replace('shot', '') for k in query["keyframeList"]]
        if mode == 'multiquery-frequency':
            results = []
            # there are three subqueries inside every query
            for sub_q in tqdm.tqdm(query["textList"]):
                text = sub_q["text"]
                t_emb = qe.get_text_embedding(text)
                res = ir.sequential_search(t_emb, enable_scalar_quantization=enable_scalar_quantization, factor=sq_factor, thr=sq_thr)[0]
                # filter up to K
                results.extend(res[:sub_K])

            # merge the results
            freqs = collections.Counter(results)
            ind, _ = zip(*sorted(freqs.items(), key=lambda item: -item[1]))

            rank = compute_rank(ref_img_ids, ind)
            ranks.append(rank)
        elif mode == 'multiquery-sum-scores':
            # there are three subqueries inside every query
            final_scores = {}
            for sub_q in tqdm.tqdm(query["textList"]):
                text = sub_q["text"]
                t_emb = qe.get_text_embedding(text)
                res = ir.sequential_search(t_emb, enable_scalar_quantization=enable_scalar_quantization, factor=sq_factor, thr=sq_thr)
                # filter up to K
                scores_dict = {i: s for i, s in zip(res[0][:sub_K], res[1][:sub_K])}
                # merge the dictionaries summing values
                for k in scores_dict:
                    if k in final_scores:
                        final_scores[k] += scores_dict[k]
                    else:
                        final_scores[k] = scores_dict[k]
            ind, _ = zip(*sorted(final_scores.items(), key=lambda item: -item[1]))

            rank = compute_rank(ref_img_ids, ind)
            ranks.append(rank)
        elif mode == '3-queries-merged-min':
            text = ' '.join([q['text'] for q in query["textList"]])
            t_emb = qe.get_text_embedding(text)
            ind = ir.sequential_search(t_emb, enable_scalar_quantization=enable_scalar_quantization, factor=sq_factor, thr=sq_thr)[0]
            ind = ind[:sub_K]

            rank = compute_rank(ref_img_ids, ind)
            ranks.append(rank)

        elif mode == '2-queries-merged-min':
            text_queries = [q['text'] for q in query["textList"]]
            min_rank = 1e20
            for sub_q in tqdm.tqdm(itertools.combinations(text_queries, 2)):
                text = ' '.join(sub_q)
                t_emb = qe.get_text_embedding(text)
                ind = ir.sequential_search(t_emb, enable_scalar_quantization=enable_scalar_quantization, factor=sq_factor, thr=sq_thr)[0]
                ind = ind[:sub_K]

                rank = compute_rank(ref_img_ids, ind)
                if rank < min_rank:
                    min_rank = rank
            ranks.append(min_rank)

        elif mode == '1-query-min':
            min_rank = 1e20
            for sub_q in tqdm.tqdm(query["textList"]):
                text = sub_q["text"]
                t_emb = qe.get_text_embedding(text)
                ind = ir.sequential_search(t_emb, enable_scalar_quantization=enable_scalar_quantization, factor=sq_factor, thr=sq_thr)[0]
                # filter up to K
                ind = ind[:sub_K]
                rank = compute_rank(ref_img_ids, ind)
                if rank < min_rank:
                    min_rank = rank
            ranks.append(min_rank)
        else:
            raise ValueError("mode not known")

        # print('Query {} rank: {}'.format(qid, rank))
    # print(ranks)
    return ranks

# def evaluate(mode, sq_thr):
#     return [1,4,7,2,7,5]

if __name__ == '__main__':
    modes = ['3-queries-merged-min', '2-queries-merged-min', '1-query-min']
    sq_thresholds = [None, 18, 22, 26, 30, 40, 50]
    results = []
    for mode, sq_thr in tqdm.tqdm(itertools.product(modes, sq_thresholds)):
        ranks = evaluate(mode, sq_thr)
        print('mode={}, sq_thr={}, ranks={}'.format(mode, sq_thr, ranks))
        results.append({'mode': mode, 'sq_thr': sq_thr if sq_thr is not None else 'None', 'ranks': ranks})

        with open(out_file, 'w') as f:
            json.dump(results, f)