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

out_file = 'alad/extraction/v3c_gt_evaluation/results/gt_evaluation_results_sqtopk.json'

sub_K = 50000
sq_factor = 100
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

def evaluate(mode, ir):
    ranks = []
    with open(vbs2019_gt_filename, 'r') as f:
        vbs2019_gt = json.load(f)
    vbs2019_gt = vbs2019_gt["textualKISList"]
    qe = QueryEncoder(alad_args)

    for qid, query in enumerate(tqdm.tqdm(vbs2019_gt)):
        ref_img_ids = [k['keyframe'].replace('.png', '').replace('shot', '') for k in query["keyframeList"]]
        
        if mode == '3-queries-merged-min':
            text = ' '.join([q['text'] for q in query["textList"]])
            t_emb = qe.get_text_embedding(text)
            ind, _ = ir.search(t_emb, k=sub_K)

            rank = compute_rank(ref_img_ids, ind)
            ranks.append(rank)

        elif mode == '2-queries-merged-min':
            text_queries = [q['text'] for q in query["textList"]]
            min_rank = 1e20
            for sub_q in tqdm.tqdm(itertools.combinations(text_queries, 2)):
                text = ' '.join(sub_q)
                t_emb = qe.get_text_embedding(text)
                ind, _ = ir.search(t_emb, k=sub_K)

                rank = compute_rank(ref_img_ids, ind)
                if rank < min_rank:
                    min_rank = rank
            ranks.append(min_rank)

        elif mode == '1-query-min':
            min_rank = 1e20
            for sub_q in tqdm.tqdm(query["textList"]):
                text = sub_q["text"]
                t_emb = qe.get_text_embedding(text)
                ind, _ = ir.search(t_emb, k=sub_K)

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
    sq_thresholds = [None, 50, 60, 70, 75, 80, 85, 90]
    results = []
    for mode, sq_thr in tqdm.tqdm(itertools.product(modes, sq_thresholds)):
        ir = ImageRetrieval(feat_db, sq_threshold=sq_thr, sq_factor=sq_factor, sq_method='topk')
        density = ir.get_density()
        nz_elems = ir.get_number_nz_elems()
        ranks = evaluate(mode, ir)
        print('mode={}, sq_thr={}, ranks={}, density={}, nz_elems={}'.format(mode, sq_thr, ranks, density, nz_elems))
        results.append({'mode': mode, 'sq_thr': sq_thr if sq_thr is not None else 'None', 'ranks': ranks, 'density': density, 'nz_elems': nz_elems})

        with open(out_file, 'w') as f:
            json.dump(results, f)