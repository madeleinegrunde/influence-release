import numpy as np
import pandas as pd
import argparse

def main(args):

    for idx in range(args.start_idx, args.end_idx):
        print()
        print()
        print(idx)
        PATH = "./output/%s/%s/predicted_loss-%s.npy" % (args.overall_dir, args.dir, idx)


        if idx == args.start_idx:
            print("Loading from %s" % PATH)

        preds = pd.Series(np.load(PATH))

        print()
        print("NUM: %s" % len(preds))
        print("MAX: %s" % preds.max())
        print("MIN: %s" % preds.min())
        print("MEA: %s" % preds.mean())
        print("MED: %s" % preds.median())



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='generic',
                                    help='folder in output')
    
    parser.add_argument('--overall_dir', type=str, default='mnist',
                                    help='folder in output')
    parser.add_argument('--start_idx', type=int, default=0,
                                                       help='Which idx to analyze')
    parser.add_argument('--end_idx', type=int, default=1,
                                                       help='Which idx to analyze')
    args = parser.parse_args()
    print("Saving outputs to %s" % args.dir)

    main(args)
