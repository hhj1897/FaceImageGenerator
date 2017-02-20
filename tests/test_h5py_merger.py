import EmoData as ED 
import h5py
list_of_paths = [
        './test_data/test_data_64_96_1.h5',
        './test_data/test_data_64_96_1_copy.h5'
        ]

ED.transformer.merge_h5(list_of_paths, '/tmp/out.h5', shuffle=True, inputer='median')


import glob
files = sorted(list(glob.glob('/homes/rw2614/data/NEW/similarity_240_160_3/fera2015/*')))

TRAIN = [
    'F001',
    'F003',
    'F005',
    'F007',
    'F009',
    'k011',
    'F013',
    'F015',
    'F017',
    'F019',
    'F021',
    'F023',
    'M001',
    'M003',
    'M005',
    'M007',
    'M009',
    'M011',
    'M013',
    'M015',
    'M017'
]
TEST = [
    'F002',
    'F004',
    'F006',
    'F008',
    'F010',
    'F012',
    'F014',
    'F016',
    'F018',
    'F020',
    'F022',
    'M002',
    'M004',
    'M006',
    'M008',
    'M010',
    'M012',
    'M014',
    'M016',
    'M018',
]
tr_list, te_list = [],[]
for seq_path in files:
    seq_id = seq_path.split('/')[-1][:4]
    if seq_id in TRAIN:tr_list.append(seq_path)
    if seq_id in TEST:te_list.append(seq_path)

ED.transformer.merge_h5(tr_list, '/homes/rw2614/data/NEW/similarity_240_160_3/fera2015_train.h5', shuffle=True)
ED.transformer.merge_h5(te_list, '/homes/rw2614/data/NEW/similarity_240_160_3/fera2015_test.h5',  shuffle=True)
