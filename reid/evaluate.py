import torch
import scipy.io
import numpy as np
import pandas as pd
import os
import json
import random
#######################################################################
def print_result(root_dir,cast,sort_result):
    f = open(os.path.join(root_dir,'trans_gt.txt'),'a')
    f.write(cast)
    f.write(' ')
    for i,candidates_id in enumerate(sort_result):
        f.write(candidates_id)
        if i != len(sort_result)-1:
            f.write(',')
    f.write('\n')
    f.close()

def find_movie(file,pid):
    query_file = pd.read_csv(file)
    find = query_file[query_file.pid==pid]
    movie = find.movie
    return(movie)

def evaluate(score, ql, gl,gid,qid):
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    sort_result=[]
    sort_result.extend(qid)
    for i in index:
        sort_result.append(gid[i])
    print(len(sort_result))
    query_index = np.argwhere(gl == ql)
    #good_index = query_index
    #ap,CMC_tmp = compute_mAP(index_pick, good_index)
    #return ap,QCMC_tmp,sort_result
    return sort_result
def compute_mAP(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc
    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2
    return ap, cmc
######################################################################
root_dir ='/home/xuqiling/eccvc'
result = scipy.io.loadmat(os.path.join('/home/xuqiling/eccvc/baseline/pytorch_result.mat'))
file = open(root_dir + '/dataset/matching_val_results_p.json', 'r')
match_json = json.load(file)
file.close()
movies = list(match_json.keys())
all_movie = []
all_id = result['gallery_id']
for i in all_id:
    all_movie.append(i[0:-5])


all_feature = result['gallery_f']
all_label = result['gallery_label']
find_id=[]

for movie in movies:
    for j in match_json[movie].keys():
        query_pick_label = j[0:-4]
        match_result = match_json[movie][j]

        match_id = list(match_result[0::2])
        print(match_id)
        weight = list(match_result[1::2])
        print(weight)

        if match_id == -1:
            match_id = random.sample(range(all_movie.count(movie)),5)
        #find_id.append(str(match_id).zfill(4))

        gallery_mask = np.in1d(all_movie,movie)
        gallery_feature = all_feature[gallery_mask]
        gallery_label = all_label[gallery_mask]
        gallery_id = all_id[gallery_mask]
        gallery_id_4 = []
        for i in gallery_id:
            gallery_id_4.append(i[-4:])

        final_score = np.zeros(len(gallery_label)-5)
        features = gallery_feature.copy()
        query_pick_id = []
        query_feature = []
        query_index = []
        for k in match_id:
            query_pick_id.append(movie+'_'+str(k).zfill(4))
            aaa = gallery_id_4.index(str(k).zfill(4))
            query_index.append(gallery_id_4.index(str(k).zfill(4)))
            query_feature.extend(features[query_index])

        gallery_feature = np.delete(gallery_feature, query_index, axis=0)
        gallery_id = np.delete(gallery_id, query_index, axis=0)
        gallery_label = np.delete(gallery_label, query_index, axis=0)
            #score.append(np.dot(gallery_feature,query_feature))
        #query_feature = np.array(query_feature)
        #weight = [1,1,1,1,1]
        for num in range(5):
            if weight[num]<0.3:
                continue
            final_score += np.dot(gallery_feature,query_feature[num])

        cast = movie +'_'+query_pick_label
        sort_result = evaluate(final_score,query_pick_label,gallery_label,gallery_id,query_pick_id)

        print_result(root_dir, cast, sort_result)

