from os import walk

# Get all filenames in directory

def main(args):
    f = []
    mypath = 'output/mnist/%s' % args.dir
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break

    preds = [i for i in f if i[:4] == 'pred']
    alls = [i for i in f if i[:3] == 'all']

    preds = [i.split('-')[1].split('.')[0] for i in preds]
    alls = [i.split('-')[1].split('.')[0] for i in alls]

    todo = []

    for p in preds:
        if p not in alls:
            todo.append(p)

    todo = [int(i) for i in todo]
    todo = sorted(todo)

    start = -10
    curr = -10
    ranges = []
    for i in todo:
        if curr + 1 == i:
            curr = i
            continue
        else:




