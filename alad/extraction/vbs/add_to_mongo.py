import argparse
import itertools

import h5py
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
from tqdm import tqdm

def grouper(iterable, n):
    """ Iterates an iterable in batches. E.g.,
        grouper([1, 2, 3, 4, 5], 2)  -->  [(1, 2), (3, 4), (5,)]
    """
    it = iter(iterable)
    while True:
        chunk_it = itertools.islice(it, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)


def generate_records(feat_db, dataset):
    """ Generatore dei vostri records. Ogni record è un dict. """

    with h5py.File(feat_db, 'r') as image_data:
        features = image_data['features']
        img_ids = image_data['image_names']

        # features, img_ids = ir.encode_features_and_postprocess(enable_scalar_quantization=False)

        # qua ho supposto un file di testo semplice scorso per linee
        for feat, img_id in zip(features, img_ids):
            # parsate qua il vostro formato per produrre un dict come il seguente:
            feat = feat.tolist()
            img_id = img_id.decode()

            record = {
                # useful ids
                '_id': '{}.jpg'.format(img_id),  # '_id' è indicizzato automaticamente da Mongo,
                # quindi è il campo da usare per fare retrival
                # veloce delle info di un frame.

                'video_id': img_id.rsplit('_', 1)[0] if dataset == 'mvk' else img_id.split('_')[0],

                'feature': feat
            }

            yield record


def main(args):
    """
    Requirements: pip install pymongo
    Prima di usare questo script, Mongo deve essere up. Per farlo partire
    sulla macchina di Paolo, basta lanciare in ssh:
    docker start vbs-mongo
    Una volta finita l'indicizzazione, potete spegnere il servizio con:
    docker stop vbs-mongo
    L'istanza di Mongo su bilioso.isti.cnr.it è password-protected,
    ma cmq accedibile dalla rete CNR, meglio stopparlo quando non serve,
    così per scrupolo.
    """

    client = MongoClient(args.url, username=args.username, password=args.password)

    # ho creato un database separato per dataset
    if 'v3c1' in args.input.lower():
        db_name = 'v3c1'
    elif 'v3c2' in args.input.lower():
        db_name = 'v3c2'
    elif 'mvk' in args.input.lower():
        db_name = 'mvk'
    else:
        raise ValueError('v3c1, v3c2, or mvk? :/')

    ans = input('Adding to {} database. Ok? '.format(db_name))
    if ans != 'y':
        quit()

    # all'interno di un db si possono creare più collezioni (parallelo delle tabelle in SQL):
    # si può usare il . per organizzare le collezioni in sottocollezioni per chiarezza,
    # ho pensato di usare 'objects.' come prefisso per le detections
    collection_name = 'features.aladin'

    # la collezione in cui inserire i dati
    collection = client[db_name][collection_name]

    # la lista di dict da inserire (qua è un generatore più che una lista)
    records = generate_records(args.input, db_name)

    # si fanno batch di record da inserire per fare bulk indexing e velocizzare
    batch_size = 1000
    batches_of_records = grouper(records, batch_size)

    # scorre i batch di record
    for batch in tqdm(batches_of_records, unit_scale=batch_size):
        try:
            # si inserisce un batch; questa chiamata solleva un'eccezione alla fine dell'indicizzazione se qualche
            # record non è stato inserito, solitamente se è già presente nella collezione (stesso _id)
            collection.insert_many(batch, ordered=False)
        except BulkWriteError:
            # ignoro eventuali fallimenti di inserimento dei record già esistenti
            print('Duplicate entries discarded.')

    # stampa il numero di oggetti nella collezione
    collection_stats = client[db_name].command('collstats '+collection_name)
    print('n. records in collection:', collection_stats['count'])


def print_record(args):
    """ Vi lascio in questa funzione un esempio per cercare e stampare un record in Mongo. """
    from pprint import pprint  # pretty-print dict

    client = MongoClient(args.url, username=args.username, password=args.password)
    collection = client['v3c1']['features.aladin']

    query = {'_id': '01351_1.jpg'}
    record = collection.find_one(query)

    pprint(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add records to Mongo.')
    parser.add_argument('input', help='path to file containing data')
    parser.add_argument('--url', type=str, help="URL of Mongo database")
    parser.add_argument('--username', type=str, help="Username of Mongo database")
    parser.add_argument('--password', type=str, help="Password of Mongo database")
    args = parser.parse_args()
    main(args)
    # print_record()