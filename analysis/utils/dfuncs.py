import numpy as np
import pandas as pd
import os
from scipy import stats

#########################
#    Loading / Saving   #
#########################

def pathExists(path):
    return os.path.exists(path)

'''
    Return the predicted influence functions
'''
def loadPreds(f, idx, overall_dir='mnist'):
    return pd.Series(np.load('../output/%s/%s/predicted_loss-%s.npy' % (overall_dir, f, idx)))


'''
    Return the predicted and actual retraining losses. 
'''
def loadRetraining(file, idx, overall_dir='mnist', return_idxs=False):
    path = "../output/%s/%s/mnist_small_all_cnn_c_loss_diffs-%s.npz" % (overall_dir, file, idx)
    if not os.path.exists(path):
        print("Retraining path does not exist: ", path)
        return False, False
        
    data = np.load(path, allow_pickle=True)
    lst = data.files
    
    stuff = {}

    for item in lst:
        stuff[item] = list(data[item])

    if return_idxs:
        return stuff
    
    actual = stuff['actual_loss_diffs']
    predicted = stuff['predicted_loss_diffs']
        
    return predicted, actual



#########################
#     Analyze Arrays    #
#########################

'''
    Gets the basic distributional info

    Input: Array of numbers
    Output: dictionary with information
'''
def getBasicDistrInfo(arr):
    arr = pd.Series(arr)

    x = {
        'max': arr.max(),
        'min': arr.min(),
        'mean': arr.mean(),
        'median': arr.median(),
        'std': arr.std(),
    }

    return x


'''
    Gets the idx that are the largest or smallest

    Input: 
        List: list of values
        Largest: True if return largest, false if return smallest
        Num: Number of extreme value idxs to return

    Output: 
        list of idxs of the most extreme values
'''
def getExtremeIdxs(lst, largest=True, num=10):
    if largest:
        return list(reversed(sorted(range(len(lst)), key=lambda i: lst[i])[-num:]))
    else:
        return sorted(range(len(lst)), key=lambda i: lst[i])[:num]


'''
    Does a spearman rank correlation over two lists
'''
def spearman(a, b):
    if len(a) != len(b):
        print("Attempting to get rank correlation of list of lengths %s and %s" % (len(a), len(b)))
    c, p = stats.spearmanr(a, b)
    return c, p


'''
    Def take same test images and get their correlations accross two folders

    Inputs:
        f1: one folder
        f2: second folder
        idx_range: Test images 0-X
        num: number of extreme images to get
        largest: largest or smallest extreme
        overall_dir: where output folders are found
    Output:
        list of correlations for each test image
        list of pvalues for each correlation
'''
def correlateTwoFoldersExtreme(f1, f2, idx_range=10, num=10, largest=True, overall_dir='mnist'):
    cors = []
    ps = []

    for i in range(idx_range):
        p1 = loadPreds(f1, i, overall_dir=overall_dir)
        p2 = loadPreds(f2, i, overall_dir=overall_dir)

        if len(p1) != len(p2):
            continue

        top = getExtremeIdxs(p1, num=num, largest=largest) + getExtremeIdxs(p2, num=num, largest=largest)

        top1 = [p1[i] for i in top]
        top2 = [p2[i] for i in top]

        c, p = stats.spearmanr(top1, top2)
        cors.append(c)
        ps.append(p)
    return cors, ps


'''
    Formats data for making a qqplot

    Inputs:

    Outputs:
        A list of tuples of lists to be compared 
'''
def getQQplotData(f1, f2='', idx_range=10, retrained=False, only_extremes=False, num_extreme=10):
    data = []

    for i in range(idx_range):
        if retrained:
            a, b = loadRetraining(f1, i)
            
            if not a:
                continue
        else:
            a = loadPreds(f1, i)
            b = loadPreds(f2, i)

            if len(a) != len(b):
                continue
        
        if only_extremes:
            extreme_idxs_a = getExtremeIdxs(a, num=num_extreme)
            extreme_idxs_b = getExtremeIdxs(b, num=num_extreme)
            extreme_idxs = extreme_idxs_a + extreme_idxs_b
            
            a = [a[idx] for idx in extreme_idxs]
            b = [b[idx] for idx in extreme_idxs]

        data.append([a, b])



    return data
