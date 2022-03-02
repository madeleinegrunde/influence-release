from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython
import argparse
import tensorflow as tf
import os

# added in order o be able to access influence
import sys
sys.path.append("/scratch/mgrunde/influence-release/")

import influence.experiments as experiments
from influence.all_CNN_c import All_CNN_C

from load_mnist import load_small_mnist, load_mnist


def main(args):
    data_sets = load_small_mnist('data')    

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
        initial_learning_rate=initial_learning_rate,
        damping=1e-2,
        decay_epochs=decay_epochs,
        mini_batch=True,
        train_dir='output/mnist/%s' % args.output_dir, 
        log_dir='log',
        model_name='mnist_small_all_cnn_c')


    # save output dir if doesnt exist
    if not os.path.exists(model.train_dir):
        print('making directory to', model.train_dir)
        os.makedirs(model.train_dir)


    num_steps = 500000
    
    if args.refresh:
        print("Refreshing from iteration %s" % args.iter_to_load)
        model.load_checkpoint(args.iter_to_load, do_checks=False)
    else:
        print("Training from scratch")
        model.train(
            num_steps=num_steps, 
            iter_to_switch_to_batch=10000000,
            iter_to_switch_to_sgd=10000000)
    
    iter_to_load = num_steps - 1

    test_idx = 6558

    actual_loss_diffs, predicted_loss_diffs, indices_to_remove = experiments.test_retraining(
        model, 
        test_idx=test_idx, 
        iter_to_load=iter_to_load, 
        num_to_remove=args.num_test,
        num_steps=30000, 
        remove_type='maxinf',
        force_refresh=True)

    np.savez(
        'output/mnist/%s/mnist_small_all_cnn_c_iter-500k_retraining-100.npz' % args.output_dir, 
        actual_loss_diffs=actual_loss_diffs, 
        predicted_loss_diffs=predicted_loss_diffs, 
        indices_to_remove=indices_to_remove
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='basic',
                                    help='folder in output')
    parser.add_argument('--num_test', type=int, default=20,
                                            help='number test set examples to get influence for')
    parser.add_argument('--refresh', type=bool, default=False,
                                            help='True if refresh from checkpoint, false ow')
    parser.add_argument('--iter_to_load', type=int, default=499999,
                                            help='Which iteration to refresh from')
    args = parser.parse_args()
    print("Saving outputs to %s", args.output_dir)
    main(args)
