# import main Flask class and request object
from flask import Flask, request
from alad.extraction.image_retrieval import QueryEncoder
import os
import surrogate
import logging
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ""

logging.basicConfig(level=logging.DEBUG)

# create the Flask app
app = Flask(__name__)

# create the query encoder object
args_str = '--eval_model_dir checkpoints --max_seq_length 50 --max_img_seq_length 34 --load_checkpoint alad/checkpoints/model_best_rsum.pth.tar'
qe = QueryEncoder(args_str)

# the pkl index is file-mounted on docker
encoder = surrogate.load_index('str-encoder.features.aladin.pkl')

# save default values
sq_factor_default = encoder.sq_factor if hasattr(encoder, 'sq_factor') else None
nprobe_default = encoder.nprobe if hasattr(encoder, 'nprobe') else None
keep_default = encoder.keep if hasattr(encoder, 'keep') else None
threshold_percentile_default = encoder.threshold_percentile_default if hasattr(encoder, 'threshold_percentile') else None

logging.debug(f'Default values: sq_factor={sq_factor_default}; nprobe={nprobe_default}; keep={keep_default}; threshold_percentile={threshold_percentile_default}')

@app.route('/get-text-feature', methods=['GET'])
def query_example():
    text = request.args.get("text")

    # overwrite sq_factor
    if 'sq_factor' in request.args and hasattr(encoder, 'sq_factor'):
        sq_factor = request.args.get("sq_factor", type=int)
        logging.debug(f'Overwriting sq_factor. From {encoder.sq_factor} to {sq_factor}')
        encoder.sq_factor = sq_factor
    else:
        encoder.sq_factor = sq_factor_default

    # overwrite nprobe
    if 'nprobe' in request.args and hasattr(encoder, 'nprobe'):
        nprobe = request.args.get("nprobe", type=int)
        logging.debug(f'Overwriting nprobe. From {encoder.nprobe} to {nprobe}')
        encoder.nprobe = nprobe
    else:
        encoder.nprobe = nprobe_default

    # overwrite keep or sq_threshold
    if 'keep' in request.args:
        keep = request.args.get("keep", type=float)
        if hasattr(encoder, 'keep'):
            logging.debug(f'Overwriting keep. From {encoder.keep} to {keep}')
            encoder.keep = keep
        elif hasattr(encoder, 'threshold_percentile'):
            threshold_percentile = (1 - keep) * 100
            logging.debug(f'Converting keep to threshold_percentile. From {encoder.threshold_percentile} to {threshold_percentile}')
            encoder.threshold_percentile = threshold_percentile
    else:
        if hasattr(encoder, 'keep'):
            encoder.keep = keep_default
        elif hasattr(encoder, 'threshold_percentile'):
            encoder.threshold_percentile = threshold_percentile_default

    # first, produce the feature
    text_feature = qe.get_text_embedding(text)
    text_feature = text_feature[np.newaxis, :]  # 1 x 1024

    # encode the features (returns a sparse matrix)
    x_enc = encoder.encode(text_feature, inverted=False, query=True)

    # generate surrogate documents (x_str is a generator of strings)
    x_str = surrogate.generate_documents(x_enc)
    x_str = list(x_str)
    return x_str[0]

if __name__ == '__main__':
    # run app in debug mode on port 5005
    app.run(debug=False, host='0.0.0.0', port=5021)