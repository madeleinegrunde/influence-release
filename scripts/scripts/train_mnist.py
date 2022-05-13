from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import sys
sys.path.append('/gscratch/krishna/mgrunde/inf/influence-release')
import numpy as np
import IPython
import tensorflow as tf
import influence.experiments as experiments
from influence.all_CNN_c import All_CNN_C
from load_mnist import load_small_mnist, load_mnist
import argparse
import os


def makePath(path):
    if not os.path.exists(path):
        print('making directory to', path)
        os.makedirs(path)

def main(args, test_idx):
    if args.train_subset != 0:
        print("loading small subset!")
        data_sets, train, val, test = load_small_mnist('data', args.train_subset)    
    else:
        print("loading entire dataset")
        data_sets, train, val, test = load_mnist('data')
    #print("TRAIN LEN: ", len(train))

    num_classes = 10
    input_side = 28
    input_channels = 1
    input_dim = input_side * input_side * input_channels 
    weight_decay = 0.001
    batch_size = 500

    initial_learning_rate = 0.0001 
    decay_epochs = [10000, 20000]
    hidden1_units = 8
    hidden2_units = 8
    hidden3_units = 8
    conv_patch_size = 3
    keep_probs = [1.0, 1.0]


    model = All_CNN_C(
        input_side=input_side, 
        input_channels=input_channels,
        conv_patch_size=conv_patch_size,
        hidden1_units=hidden1_units, 
        hidden2_units=hidden2_units,
        hidden3_units=hidden3_units,
        weight_decay=weight_decay,
        num_classes=num_classes, 
        batch_size=batch_size,
        data_sets=data_sets,
        data_sets_train=train,
        data_sets_val=val,
        data_sets_test=test,
        initial_learning_rate=initial_learning_rate,
        damping=1e-2,
        decay_epochs=decay_epochs,
        mini_batch=True,
        # changing train dir so that dont need to re-do models
        train_dir='saved_models/mnist',#'output/mnist/%s' % args.output_dir, 
        results_dir='output/mnist/%s' % args.dir,#'output/mnist/%s' % args.output_dir, 
        log_dir='log',
        model_name='mnist_small_all_cnn_c')


    # save output dir if doesnt exist
    if not os.path.exists(model.train_dir):
        print('making directory to', model.train_dir)
        os.makedirs(model.train_dir)

    print("MODEL RESULTS DIR: %s" % model.results_dir)
    if not os.path.exists(model.results_dir):
        print('making directory to', model.results_dir)
        os.makedirs(model.results_dir)


    num_steps = 500000
    
    # Load the model
    if not args.retrain:
        print("Refreshing from iteration %s" % args.iter_to_load)
        model.load_checkpoint(args.iter_to_load, do_checks=False)
        iter_to_load = args.iter_to_load
    else:
        print("Training from scratch")
        model.train(
            num_steps=num_steps, 
            iter_to_switch_to_batch=10000000,
            iter_to_switch_to_sgd=10000000)
    
        iter_to_load = num_steps - 1

    #test_idx = args.test_idx

    # Retrain or just get influences!
    if args.test_retraining:
        for test_idx in range(args.start_idx, args.end_idx):
            print("Testing index %s: " % test_idx)
            actual_loss_diffs, predicted_loss_diffs, indices_to_remove = experiments.test_retraining(
                model, 
                test_idx=test_idx, 
                iter_to_load=iter_to_load, 
                num_to_remove=args.num_test,
                num_steps=30000, 
                remove_type='maxinf',
                force_refresh=args.force_hvp_refresh)

            np.savez(
                'output/mnist/%s/mnist_small_all_cnn_c_iter-500k_retraining-100.npz' % args.dir, 
                actual_loss_diffs=actual_loss_diffs, 
                predicted_loss_diffs=predicted_loss_diffs, 
                indices_to_remove=indices_to_remove
                )
    else:
        for test_idx in range(args.start_idx, args.end_idx):
            print("Testing index %s: " % test_idx)
            experiments.get_train_influences(
                model, 
                test_idx=test_idx, 
                iter_to_load=iter_to_load, 
                #num_to_remove=args.num_test,
                num_steps=30000, 
                remove_type='maxinf',
                force_refresh=args.force_hvp_refresh) # make true if want to generate inverse hvp each time



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='generic',
                                    help='folder in output')
    parser.add_argument('--num_test', type=int, default=1,
                                            help='number test set examples to get influence for')
    parser.add_argument('--retrain', type=bool, default=False,
                                            help='True if refresh from checkpoint, false ow')
    parser.add_argument('--iter_to_load', type=int, default=499999,
                                            help='Which iteration to refresh from')
    parser.add_argument('--start_idx', type=int, default=0,
                                                       help='Which idx to start test')
    parser.add_argument('--end_idx', type=int, default=1,
                                                    help='Which idx to end test')
    parser.add_argument('--test_retraining', type=bool, default=False,
                                                       help='To check closeness to retraining')
    parser.add_argument('--train_subset', type=int, default=0,
                                                    help='Zero if use all train, otherwise divisor')
    parser.add_argument('--force_hvp_refresh', type=bool, default=False,
                                            help='True if refresh hvp')
    args = parser.parse_args()
    print("Saving outputs to %s", args.dir)
    

    # deleted 3 because already have that
    idxs = [1]# [10, 13, 25, 28, 55, 69, 71, 101, 126, 2, 5, 14, 29, 31, 37, 39, 40, 46, 57, 1, 35, 38, 43, 47, 72, 77, 82, 106, 119, 18, 30, 32, 44, 51, 63, 68, 76, 87, 90, 4, 6, 19, 24, 27, 33, 42, 48, 49, 56, 8, 15, 23, 45, 52, 53, 59, 102, 120, 127, 11, 21, 22, 50, 54, 66, 81, 88, 91, 98, 0, 17, 26, 34, 36, 41, 60, 64, 70, 75, 61, 84, 110, 128, 134, 146, 177, 179, 181, 184, 7, 9, 12, 16, 20, 58, 62, 73, 78, 92]

    for idx in idxs:
        print()
        #print("STARTING IDX %s" % idx)
        #args.output_dir = "%s-%s" % (args.output_dir, idx)
        #makePath(args.dir)
        main(args, idx)
