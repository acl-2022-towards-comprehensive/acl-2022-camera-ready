# Author: Zhaoyi Hou (Joey), Yifei Ning (Couson)
# Last Update: 3/18/2022

import sys, getopt
import pandas as pd
import json
from os import listdir
import pickle
import argparse

sys.path.insert(1, '/home/xuanyu/acl_2022/src/')
from trainer import *
# from lattice import *

def test(config):
    print('=> [run.py: test(config)] Running testing!')

    T = Trainer(config)
    T.train_model()
    # test_data = pd.read_csv()

def run_train(config):
    print('=> [run.py: train(config)] Running training on model:', config['model_name'])
    print(config)

    T = Trainer(config)
    T.train_model()

def run_single_test_wrapper(config, dataset = 'test'):
    T = Trainer(config)

    model_name = config['model_name'] #'claim_101_bert_full'
    checkpoint_path = '/data1/xuanyu/checkpoints/' + model_name + '/' + config['from_checkpoint']
    test_dataloader_dir = '/data4/xuanyu/full_data_2021/dataloaders/%s/' % config['dataloader_name']
    test_dataloader = pickle.load(open(test_dataloader_dir + dataset + '_dataloader_claims.pickle', 'rb'))
    if isinstance(test_dataloader, TensorDataset):
        test_dataloader = prepare_dataloader(
            test_dataloader, SequentialSampler, config['batch_size']
        )
    print('test loader dir:', test_dataloader_dir + dataset + '_dataloader_claims.pickle')
    print('=> [run.py: run_single_test] Running testing on model:', checkpoint_path)
    if 'app_feats' in T.model_name:
        app_feats_sample = pickle.load(open(test_dataloader_dir + 'app_feats_sample.pickle', 'rb'))
        print('==>  [Trainer.py: train_model(self)] Shape of app_feats_sample:', app_feats_sample.shape)
        app_feat_length = app_feats_sample[0].shape[0]
        T.trainer_config['app_feat_length'] = app_feat_length
    else:
        T.trainer_config['app_feat_length'] = 0

    _, all_prediction_dict = T.run_single_test(test_dataloader, checkpoint_path, return_prediction_dict = True)
    return all_prediction_dict

def data_cleaning(config):
    pass

def run_grid_search(config, target):
    config["max_training_steps"] = 3000
    config_archive = config.copy()

    lst = config[target] #[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    for i in lst:
        config = config_archive.copy()
        config[target] = i
        config['model_name'] += ('_gridsearch_') + 'lambda%s' % str(i)
        grid_search_handler(config, target)
        torch.cuda.empty_cache()

# def run_grid_search(config, target):
#     config["max_training_steps"] = 3000
#     config_archive = config.copy()
#
#     # Define parameter scope
#     all_mid_dim = [3, 10, 100, 1000]
#     for mid_dim in all_mid_dim:
#         config = config_archive.copy()
#         config['mid_dim'] = mid_dim
#         config['model_name'] += ('_gridsearch_') + str(mid_dim)
#         grid_search_handler(config)

def grid_search_handler(config, target):
    T = Trainer(config)

    original_model_name = '_'.join(config['model_name'].split('_')[:-2])
    # print(original_model_name)
    # T.dataloaders_dir = '/data4/xuanyu/full_data_2021/dataloaders/' + original_model_name + '/'
    # T.trainer_config['dataloaders_dir'] = '/data4/xuanyu/full_data_2021/dataloaders/' + original_model_name + '/'

    print(' =>  [Trainer.py: grid_search_handler()]')
    print('         Currently doing grid search on', original_model_name)
    print('         current renamed model:', T.model_name)
    print('         searching for learning rate:', config['lr'])
    print('         searching for %s:' % target, config[target])
    print('         dataloaders_dir:', T.dataloaders_dir)

    T.train_model()

def display_help_menu():
    print('=> [run.py: parse_arg(argv)] Error parsing command line arguments!')
    print('         -h / --helep: show help menu')
    print('         -m= / --model=: [Required] the model name. [NEED TO HAVE CORRESPONDING JSON FILE IN "./config/" !!]')
    print('         -t / --test: running in test mode (i.e. with a toy dataset in ./data/test/)')
    print('         -r / --run: running in training mode (i.e. with specified dataset in /data4/xuanyu/full_data_2021/')
    print('         -s / --single-test: running single test')
    print('         -g / --grid-search: running grid search')

def parse_arg(argv):
    try:
        opts, args = getopt.getopt(argv, "htrs:g:m:l", ["help", "test", "run", "single-test=", "grid-search", "model=", "lattice"])
    except getopt.GetoptError:
        display_help_menu()
        sys.exit(2)
    # print(opts, args)
    target = None
    model = None
    testing = False
    st_dataset = 'test'
    gs_target = ''

    for opt, arg in opts:
        if opt == '-h':
            display_help_menu()
            sys.exit()
        elif opt in ("-t", "--test"):
            testing = True
        elif opt in ("-r", "--run"):
            target = "run"
        elif opt in ("-s", "--single-test"):
            target = "single-test"
            st_dataset = arg
        elif opt in ("-g", "--grid-search"):
            target = "grid-search"
            gs_target = arg
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-l", "--lattice"):
            target = "lattice"
            model = "lattice"

    # Beilei: Debug only
    # target = "run"
    # model = "claim_102_bigbird_full_1e-6_5"

    return target, model, testing, st_dataset, gs_target

def run_lattice(config):
    LT = LatticeTrainer(config)
    LT.train()
    return 0

def main():

    target, model, testing, st_dataset, gs_target = parse_arg(sys.argv[1:])
    if not target:
        print('=> [run.py: main()] No target given! Running with test...')
        target = 'test'
    if not model:
        print('=> [run.py: main()] No model name given! terminating...')
        return

    config_file_name = model + '.json'

    if config_file_name not in listdir('config'):
        print('=> [run.py: main()] No json file with name:"' + config_file_name + '" found! terminating...')
        return

    # Load json file
    with open('config/' + config_file_name) as json_file:
        config = json.load(json_file)

    # Change the model_name correspondingly
    if testing:
        config['testing'] = 1
        config['model_name'] = config['model_name'] + '_test'
        config['dataloader_name'] = config['dataloader_name'] + '_test'
    else:
        config['testing'] = 0

    if target == "run":
        run_train(config)

    elif target == "single-test":
        if st_dataset == 'all':
            final_dict = {}
            for ds in ['train', 'validation', 'test']:
                all_prediction_dict = run_single_test_wrapper(config, dataset = ds)
                final_dict.update(all_prediction_dict)
        else:
            final_dict = run_single_test_wrapper(config, dataset = st_dataset)

        # Save the prediction dict
        pickle.dump(final_dict, open('/data1/xuanyu/checkpoints/' + config['model_name'] + '/all_prediction_dict.pickle', 'wb'))

    elif target == "grid-search":
        run_grid_search(config, gs_target)

    elif target == "lattice":
        run_lattice(config)

    elif target == "test":
        test(config)

if __name__ == "__main__":
    main()

    # python run.py -m=claim_101_bert_full_app_feats_256
