# import main Flask class and request object
from alad.extraction.retrieval_utils import scalar_quantization
from flask import Flask, request
from flask import jsonify
from alad.extraction.image_retrieval import QueryEncoder
import os
# import surrogate
import logging
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ""

default_sq_threshold = 30
default_sq_factor = 0
default_nprobe = 1

# create the Flask app
app = Flask(__name__)

# create Image Retrieval object

qe = QueryEncoder('--data_dir /media/nicola/SSD/OSCAR_Datasets/coco_ir --img_feat_file /media/nicola/Data/Workspace/OSCAR/scene_graph_benchmark/output/X152C5_test/inference/vinvl_vg_x152c4/predictions.tsv --eval_model_dir /media/nicola/SSD/OSCAR_Datasets/checkpoint-0132780 --max_seq_length 50 --max_img_seq_length 34 --load_checkpoint /media/nicola/Data/Workspace/OSCAR/Oscar/alad/runs/alad-alignment-and-distill/model_best_rsum.pth.tar')
rotation_matrix = np.load('alad/extraction/random_rotation_matrix.npy').astype(np.float32)

# create the STR encoder
# encoders = {
#     'voronoi': None, # surrogate.load_index('str-encoder-v3c.features.tern.pkl'), # TODO: as of now, voronoi index is not used. Replaced with flat for saving memory
#     'flat': surrogate.load_index('str-encoder-flat-v3c.features.tern.pkl')
#     }

@app.route('/get-text-feature', methods=['GET'])
def query_example():
    text = request.args.get("text")
    sq_factor = request.args.get("sq_factor", type=int, default=default_sq_factor)
    sq_threshold = request.args.get("sq_threshold", type=int, default=default_sq_threshold)
    want_surrogate = request.args.get("surrogate", default=False, type=lambda v: v.lower() == 'true')
    nprobe = request.args.get("nprobe", type=int, default=default_nprobe)
    k = request.args.get("k", type=int, default=260 if nprobe == 1 else 0.9)

    # encoder = encoders['flat'] # if nprobe == 1 else encoders['voronoi']  # TODO: as of now, voronoi index is not used. Replaced with flat for saving memory
    # print('encoder k: {}'.format(encoder.k))

    print('Received: sq-factor: {}, sq_thresh: {}, text: {}. Surrogate representation requested: {} (with nprobe={})'.format(sq_factor, sq_threshold, text, want_surrogate, nprobe))
    if want_surrogate:
        return NotImplementedError
        # first, produce the feature
        text_feature = ir.encode_query(text)
        text_feature = text_feature[np.newaxis, :]  # 1 x 1024

        # encode the features (returns a sparse matrix)
        encoder.nprobe = nprobe
        encoder.k = k
        x_enc = encoder.encode(text_feature, inverted=False, query=True)

        # generate surrogate documents (x_str is a generator of strings)
        x_str = surrogate.generate_documents(x_enc)
        x_str = list(x_str)
        return x_str[0]
    else:
        text_feature = qe.get_text_embedding(text)
        text_feature = scalar_quantization(text_feature, threshold=sq_threshold, factor=sq_factor,
                                             rotation_matrix=rotation_matrix, subtract_mean=False)
        out = jsonify(text_feature.tolist())
    return out

if __name__ == '__main__':
    # run app in debug mode on port 5005
    app.run(debug=False, host='0.0.0.0', port=5005)