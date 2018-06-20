import scipy.io
import numpy as np
import os
import json
import random
import time
from  re_ranking import re_ranking
#######################################################################
# Evaluate
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

def evaluate(score, ql, gl,gid):
    # predict index
    index = np.argsort(score)  # from small to large
    #index = index[::-1]

    sort_result=[]
    for i in index:
        sort_result.append(gid[i])
    print(len(sort_result))
    query_index = np.argwhere(gl == ql)
    #good_index = query_index
    #ap,CMC_tmp = compute_mAP(index_pick, good_index)
    #return ap,QCMC_tmp,sort_result
    return sort_result

######################################################################
root_dir ='/home/xuqiling/eccvc'
try:
    os.remove(os.path.join(root_dir,'trans_gt.txt'))
except OSError:
    pass
result = scipy.io.loadmat(os.path.join('/home/xuqiling/eccvc/baseline/pytorch_result.mat'))
file = open(root_dir + '/dataset/matching_val_results.json', 'r')
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
        match_id= match_json[movie][j]
        '''
        match_id = list(match_result[0::2])
        print(match_id)
        weight = list(match_result[1::2])
        print(weight)
        '''
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

        #final_score = np.zeros(len(gallery_label))
        features = gallery_feature.copy()
        query_pick_id = []
        query_feature = np.zeros(shape=(5,2048))
        query_index = []
        for i,k in enumerate(match_id):
            query_pick_id.append(movie + '_' + str(k).zfill(4))
            index = gallery_id_4.index(str(k).zfill(4))
            query_index.append(index)
            query_feature[i] = features[index]


        print('calculate initial distance')
        q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
        q_q_dist = np.dot(query_feature, np.transpose(query_feature))
        g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
        #since = time.time()
        re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        final_score = re_rank.sum(axis=0)
        #print(final_score)
        #time_elapsed = time.time() - since
        #print('Reranking complete in {:.0f}m {:.0f}s'.format(
                #time_elapsed // 60, time_elapsed % 60))

        cast = movie + '_' + query_pick_label
        sort_result = evaluate(final_score, query_pick_label, gallery_label, gallery_id)
        print(len(set(sort_result)), len(sort_result))
        print_result(root_dir, cast, sort_result)