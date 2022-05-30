import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

#########################
#     Basic Settings    #
#########################



def getConfigs(size=5, qq=False):   
    cfgs = {
        'size': size,
        'size_x': size,
        'size_y': size,
        'title': 'TITLE',
        'xlabel': 'XLABEL',
        'ylabel': 'YLABEL',
        'title_size': size*8,
        'label_size': size*6,
        'tick_size': size*4

    }

    if qq:
        cfgs['title_size'] = size*6
        cfgs['label_size'] = size*3

    return cfgs


def updateConfigs(cfgs, updates):
    for k, v in updates:
        cfgs[k] = v

    return v


#########################
#     Plotting Funcs    #
#########################

def labeling(graph, ax, cfgs):
    size = cfgs['size']
    if graph == 'qq':

        plt.suptitle(cfgs['title'], size=cfgs['title_size'])
        ax.text(0.5, 0.04, 'common X', ha='center', size=cfgs['label_size'])
        ax.text(0.04, 0.5, 'common Y', va='center', rotation='vertical', size=cfgs['label_size'])

    else:
        plt.title(cfgs['title'], size=cfgs['title_size'])
        plt.xlabel(cfgs['xlabel'], size=cfgs['label_size'])
        plt.ylabel(cfgs['ylabel'], size=cfgs['label_size'])
        plt.xticks(fontsize=cfgs['tick_size'])
        plt.yticks(fontsize=cfgs['tick_size'])


def hist(data):
    sns.histplot(data=data, bins=20)


'''
    Creates a qq plot

    Input: 
        data: list of tuples of lists to compare (make in data file)
        cfgs: standard graphing configs
'''
def qqplot(data):
    subplot_idx = 1

    num_qs = int(len(data[0][0]) / 5)

    num_sub = len(data)
    num_rows = math.floor(math.sqrt(num_sub))
    num_cols = math.ceil(num_sub / num_rows)


    for a, b in data:
        plt.subplot(num_rows, num_cols, subplot_idx)
        subplot_idx += 1
    
        percs = np.linspace(0,100,num_qs)
        qn_a = np.percentile(a, percs)
        qn_b = np.percentile(b, percs)

        plt.plot(qn_a,qn_b, ls="", marker="o")

        x = np.linspace(np.min((qn_a.min(),qn_b.min())), np.max((qn_a.max(),qn_b.max())))
        plt.plot(x,x, color="k", ls="--")


def barHue(data, cfgs):
    sns.barplot(data=data, x=cfgs['x'], y=cfgs['y'], hue=cfgs['hue'])


def graph(graph, data, cfgs=None):
    if cfgs is None:
        cfgs = getConfigs()
        cfgs['ylabel'] = 'Count'

    ax = plt.figure(figsize=(cfgs['size_x'], cfgs['size_y']))

    if graph == 'hist':
        hist(data)
    elif graph == 'qq':
        qqplot(data)
    elif graph == 'bar-hue':
        barHue(data, cfgs)
    else:
        print("Invalid graph type: %s" % graph)

    labeling(graph, ax, cfgs)

    plt.show()
    plt.clf()

