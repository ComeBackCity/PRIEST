import os, sys

sys.path.append(os.path.abspath("/home/gourab/Desktop/Projects/Covid/Tempel-modified/source_code"))
from src.models import models, train_model, our_models, our_model_2, cnn_model, cnn_with_attn, our_model_3, lstm_with_attn, \
    lstm_whole, lstm_corrected, lstm_whole_refactored, lstm_simplified, mcma
from src.data import make_dataset
from src.features import build_features
from src.utils import utils
import torch
import numpy as np
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import random

# SEQ_LENGTH = sys.argv[1]
# SUBTYPE = sys.argv[1]

def plot_losses(training_losses, val_losses, epochs, cell_type):

    plt.style.use('ggplot')

    # save_path = f'results_cov/{SEQ_LENGTH}/plots/loss/{cell_type}'
    # save_path = f'results_inf/{SUBTYPE}/plots/loss/{cell_type}'
    save_path = f'results_cov/2023/plots/loss/{cell_type}'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    epochs_list = [i for i in range(1, epochs+1)]
    losses = pd.DataFrame({
        'epochs' : epochs_list,
        'training_loss' : training_losses,
        'validation_loss' : val_losses
    })

    plt.figure()

    plot = sns.lineplot(x='epochs', y='loss', hue='mode', 
             data=pd.melt(losses, ['epochs'], var_name='mode', value_name='loss'))

    fig = plot.get_figure()
    fig.savefig(save_path + '/loss.svg', format='svg')

    

def plot_metrics(metric_name, training_metric, val_metric, epochs):

    plt.style.use('ggplot')
    # save_path = f'results_cov/{SEQ_LENGTH}/plots/metrics/'
    # save_path = f'results_inf/{SUBTYPE}/plots/metrics/'
    save_path = f'results_cov/2023/plots/metrics'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    epochs_list = [i for i in range(1, epochs+1)]
    train_metrics = pd.DataFrame({
        'epochs' : epochs_list,
        **training_metric
    })

    val_metrics = pd.DataFrame({
        'epochs' : epochs_list,
        **val_metric
    })

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    sns.lineplot(x='epochs', y=f'Training {metric_name}', hue='Model Type', 
                data=pd.melt(train_metrics, ['epochs'], var_name='Model Type', value_name=f'Training {metric_name}'), 
                ax=axes[0])
    sns.lineplot(x='epochs', y=f'Validation {metric_name}', hue='Model Type', 
                data=pd.melt(val_metrics, ['epochs'], var_name='Model Type', value_name=f'Validation {metric_name}'), 
                ax=axes[1])

    fig.savefig(save_path + f'/{metric_name}.pdf', format='pdf', dpi=1200)

    
def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  


setup_seed(42)

def main():

    seq_length = 3
    # print(SUBTYPE)

    base = '/home/gourab/Desktop/Projects/Covid/Tempel-modified/source_code/'

    # data_set = base + f'source_code/data/quarter_processed_weighted_2000_{SEQ_LENGTH}/cov'
    # data_path = base + f'source_code/data/quarter_processed_weighted_2000_{SEQ_LENGTH}/'
    
    data_set = base + f'data/2023_set/cov'
    data_path = base + f'data/2023_set/'

    parameters = {
            
      # Exlude _train/_test and file ending
      'data_set': data_set,
      
      # raw data path
      'data_path': data_path,
    
      # 'svm', lstm', 'gru', 'attention' (only temporal) or 'da-rnn' (input and temporal attention)
      'model': model,
    
      # Number of hidden units in the encoder
      'hidden_size': 128,
    
      # Droprate (applied at input)
      'dropout_p': 0.5,
    
      # Note, no learning rate decay implemented
      'learning_rate': 1e-3,
    
      # Size of mini batch
      'batch_size': 256,
    
      # Number of training iterations
      'num_of_epochs': 50
    }
    
    print(f'lr = {parameters["learning_rate"]}, batch size = {parameters["batch_size"]}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print(parameters['data_set'])
    # print(parameters['data_path'])

    # torch.manual_seed(1)
    # np.random.seed(1)
    # random.seed(67)
    setup_seed(100)
    # torch.use_deterministic_algorithms(True)
    if position_type == 'epitopes':
        train_trigram_vecs, train_labels = utils.read_dataset(parameters['data_set'] + '_train.csv', parameters['data_path'], concat=False)
        test_trigram_vecs, test_labels = utils.read_dataset(parameters['data_set'] + '_test.csv', parameters['data_path'], concat=False)
    else:
        train_trigram_vecs, train_labels = utils.read_dataset(parameters['data_set']  + '_train_' + str(position) + '.csv', parameters['data_path'], concat=False)
        test_trigram_vecs, test_labels = utils.read_dataset(parameters['data_set'] + '_test_' + str(position) + '.csv', parameters['data_path'], concat=False)
    X_train = torch.tensor(train_trigram_vecs, dtype=torch.float32)
    # print(X_train.shape)
    Y_train = torch.tensor(train_labels, dtype=torch.int64)
    X_test = torch.tensor(test_trigram_vecs, dtype=torch.float32)
    Y_test = torch.tensor(test_labels, dtype=torch.int64)

    #give weights for imbalanced dataset
    _, counts = np.unique(Y_train, return_counts=True)
    train_counts = max(counts)
    train_imbalance = max(counts) / Y_train.shape[0]
    _, counts = np.unique(Y_test, return_counts=True)
    test_counts = max(counts)
    test_imbalance = max(counts) / Y_test.shape[0]    

    # print(X_train.shape)

    # X_train = X_train.to(device)
    # X_test = X_test.to(device)
    # Y_train = Y_train.to(device)
    # Y_test = Y_test.to(device)

    # print(X_train.device)
    # print('Loaded all tensors to gpu')
    
#     if train_counts >= (Y_train.shape[0]-3) or test_counts >= (Y_test.shape[0]-3):
#         return(print('Experiment on cov %d at position %d is not applicable' %(position)))
#     else:
#         if position_type == 'single':
#             print('Experiment on cov %d at position: %d' %(position))
#             print('Class imbalances:')
#             print(' Training %.3f' % train_imbalance)
#             print(' Testing  %.3f' % test_imbalance)
# #            with open(parameters['data_set']  + '_train_' + str(position) +'_baseline.txt', 'r') as f:
# #                print('Train baselines:')
# #                print(f.read())
# #            with open(parameters['data_set']  + '_test_' + str(position) + '_baseline.txt', 'r') as f:
# #                print('Test baselines:')
# #                print(f.read())  
#         else:
#             print('Class imbalances:')
#             print(' Training %.3f' % train_imbalance)
#             print(' Testing  %.3f' % test_imbalance)
#             with open(parameters['data_set'] + '_train_baseline.txt', 'r') as f:
#                 print('Train baselines:')
#                 print(f.read())
#             with open(parameters['data_set'] + '_test_baseline.txt', 'r') as f:
#                 print('Test baselines:')
#                 print(f.read())
          
            
    if parameters['model'] == 'svm':
        window_size = 1
        train_model.logistic_regression_baseline(
            build_features.reshape_to_linear(train_trigram_vecs, window_size=window_size), train_labels, 
            build_features.reshape_to_linear(test_trigram_vecs, window_size=window_size), test_labels)
    elif parameters['model'] == 'random forest':
        window_size = 1
        train_model.random_forest_baseline(
            build_features.reshape_to_linear(train_trigram_vecs, window_size=window_size), train_labels, 
            build_features.reshape_to_linear(test_trigram_vecs, window_size=window_size), test_labels)  
    elif parameters['model'] == 'logistic regression':
        window_size = 1
        train_model.bayes_baseline(
            build_features.reshape_to_linear(train_trigram_vecs, window_size=window_size), train_labels, 
            build_features.reshape_to_linear(test_trigram_vecs, window_size=window_size), test_labels) 
    else:
        input_dim = X_train.shape[2]
        print(input_dim)
        seq_length = X_train.shape[0]
        output_dim = 2
        
        if parameters['model'] == 'lstm':
            net = models.RnnModel(input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'], cell_type='LSTM')
        elif parameters['model'] == 'gru':
            net = models.RnnModel(input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'], cell_type='GRU')
        elif parameters['model'] == 'rnn':
            net = models.RnnModel(input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'], cell_type='RNN')
        elif parameters['model'] == 'attention':
            net = models.AttentionModel(seq_length, input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'])
        elif parameters['model'] == 'da-rnn':
            net = models.DaRnnModel(seq_length, input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'])
        elif parameters['model'] == 'tempo':
            net = models.TransformerModel(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout_p=0.5
            )
        elif parameters['model'] == 'transformer':
            net = our_models.TimeSeriesTransformer(
                input_size=input_dim,
                enc_seq_len=seq_length,
                output_dim=output_dim
            )
        elif parameters['model'] == 'transformer2':
            windows = [i+1 for i in range(seq_length)]
            net = our_model_2.TimeSeriesTransformer(
                input_size=input_dim,
                enc_seq_len=seq_length,
                output_dim=output_dim,
                window_size = windows
            )
        elif parameters['model'] == 'transformer3':
            interval = seq_length // 3
            windows = [i+1 for i in range(0, seq_length, interval)]
            net = our_model_3.TimeSeriesTransformer(
                input_size=input_dim,
                enc_seq_len=seq_length,
                output_dim=output_dim,
                window_size = windows
            )
        elif parameters['model'] == 'lstm1':
            net = lstm_with_attn.LSTMSeriesClassifier(
                seq_length=seq_length,
                input_size=100,
                hidden_size=parameters['hidden_size'],
                num_layers=8,
                num_classes=2,
                device=device
            )
        elif parameters['model'] == 'lstm2':
            net = lstm_whole.LSTMSeriesClassifier(
                seq_length=seq_length,
                input_size=100,
                hidden_size=parameters['hidden_size'],
                num_layers=4,
                num_classes=2,
                device=device
            )
        elif parameters['model'] == 'lstm3':
            net = lstm_corrected.LSTMSeriesClassifier(
                seq_length=seq_length,
                input_size=100,
                hidden_size=parameters['hidden_size'],
                num_layers=4,
                num_classes=2,
                device=device
            )
        elif parameters['model'] == 'lstm4':
            net = lstm_whole_refactored.LSTMSeriesClassifier(
                seq_length=seq_length,
                input_size=100,
                hidden_size=parameters['hidden_size'],
                num_layers=2,
                num_classes=2,
                device=device
            )
        elif parameters['model'] == 'mcma':
            net = lstm_simplified.LSTMSeriesClassifier2(
                seq_length=seq_length,
                input_size=100,
                hidden_size=parameters['hidden_size'],
                num_layers=2,
                num_classes=2,
                dropout = parameters['dropout_p'],
                positional_dropout = 0.1,
                device=device
            )
        elif parameters['model'] == 'mcma2':
            net = lstm_simplified.LSTMSeriesClassifier3(
                seq_length=seq_length,
                input_size=100,
                hidden_size=parameters['hidden_size'],
                num_layers=2,
                num_classes=2,
                dropout = parameters['dropout_p'],
                positional_dropout = 0.1,
                device=device
            )
        elif parameters['model'] == 'mcma3':
            net = lstm_simplified.LSTMSeriesClassifier4(
                seq_length=seq_length,
                input_size=100,
                hidden_size=parameters['hidden_size'],
                num_layers=2,
                num_classes=2,
                dropout = parameters['dropout_p'],
                positional_dropout = 0.1,
                device=device
            )
        elif parameters['model'] == 'PRIEST':
            # print(f"seq_len  = {seq_length}")
            net = lstm_simplified.LSTMSeriesClassifier5(
                seq_length=seq_length,
                input_size=100,
                hidden_size=parameters['hidden_size'],
                num_layers=2,
                num_classes=2,
                dropout = parameters['dropout_p'],
                positional_dropout = 0.1,
                device=device
            )
        elif parameters['model'] == 'mcma5':
            net = mcma.MultiChannelMultiAttention(
                seq_length=seq_length,
                input_size=100,
                hidden_size=parameters['hidden_size'],
                num_layers=2,
                num_classes=2,
                dropout = parameters['dropout_p'],
                positional_dropout = 0.1,
                device=device
            )
        elif parameters['model'] == 'cnn':
            windows = [i+1 for i in range(seq_length)]
            net = cnn_model.CnnTextClassifier(
                num_classes=2,
                window_sizes=[1, 2, 3]
            )

        elif parameters['model'] == 'cnn2':
            windows = [i+1 for i in range(seq_length)]
            net = cnn_with_attn.CnnTextClassifier(
                num_classes=2,
                window_sizes=windows
            )
            
        
        if parameters['model'] == 'attention' or parameters['model'] == 'da-rnn' or parameters['model'] == 'tempo':
            metrics = train_model.train_rnn(net, False, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], X_train, Y_train, X_test, Y_test, True, parameters['model'], device)
        elif parameters['model'] == 'transformer' or parameters['model'] == 'transformer2' or parameters['model'] == 'transformer3':
            metrics = train_model.train_transformer(net, False, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], X_train, Y_train, X_test, Y_test, True, parameters['model'], device, resume=False, retrain=False)
        elif parameters['model'] == 'cnn' or parameters['model'] == 'cnn2':
            metrics = train_model.train_cnn(net, False, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], X_train, Y_train, X_test, Y_test, True, parameters['model'], device, resume=False, retrain=False)
        elif parameters['model'] == 'lstm1' or parameters['model'] == 'lstm2' or parameters['model'] == 'lstm3' \
            or parameters['model'] == 'lstm4' or parameters['model'] == 'mcma' or parameters['model'] == 'mcma2' \
            or parameters['model'] == 'mcma3' or parameters['model'] == 'PRIEST' or parameters['model'] == 'mcma5' :
            metrics = train_model.train_transformer(net, False, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], X_train, Y_train, X_test, Y_test, True, parameters['model'], device, resume=False, retrain=False)
        else:
            metrics = train_model.train_rnn(net, False, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], X_train, Y_train, X_test, Y_test, False, parameters['model'], device)

    return metrics
            

        
if __name__ == '__main__':
    seq_length = 3
    # subtype = ['H1N1', 'H3N2', 'H5N1']
    position_mode = ['epitopes', 'single']   #two mode for position mode selection
    # subtype_flag, data_path = make_dataset.subtype_selection(subtype[0])
    # data_path = '/home/gourab/Projects/CoV research/Tempel-modified/source_code/data/quarter_processed/'
    position_type = position_mode[0]   #select the predicting mode for all epitope sites or single site
    if position_type == 'epitopes':
        model_list = ['tempo', 'cnn', 'attention', 'gru', 'lstm', 'da-rnn', 'rnn', 'PRIEST']
        # model_list = ['attention', 'gru', 'lstm', 'da-rnn', 'rnn']
        # model_list = ['lstm4']
        # model_list = ['mcma5']
        #model = ['attention', 'gru', 'lstm', 'rnn','logistic regression']
        #model = ['logistic regression', 'random forest', 'rnn']
        #model = ['rnn']

        for model in model_list:
            print('\n')
            print("Experimental results with model %s on cov:" % (model))
            # dir = f'results_cov/{SEQ_LENGTH}/plots/metrics/'
            # dir = f'results_inf/10/{SUBTYPE}/plots/metrics/'
            dir = f'results_cov/2023/plots/metrics'
            if not os.path.exists(dir):
                os.makedirs(dir)
            if os.path.exists(dir+f'{model}.json'):
                print(f'Skipping {model}')
                continue
            metrics = main()
            with open(dir+f'{model}.json', 'w') as f:
                json.dump(metrics, f)
            # metrics_list.append(metrics)
            torch.cuda.empty_cache()

        metrics_list = []
        # results_path_cov = f'results_cov/{SEQ_LENGTH}/plots/metrics/'
        # results_path_inf = f'results_inf/10/{SUBTYPE}/plots/metrics'
        results_path_cov = f'results_cov/2023/plots/metrics'
        
        os.makedirs(results_path_cov, exist_ok=True)

        for model in model_list:
            model_file = model
            with open(f'{results_path_cov}{model_file}.json', 'r') as f:
                metric = json.load(f)
                if model == 'PRIEST':
                    metric['type'] = 'PRIEST'
                metrics_list.append(metric)

        trn_acc, trn_pre, trn_rec, trn_fscore, trn_mcc, trn_loss = {}, {}, {}, {}, {}, {}
        val_acc, val_pre, val_rec, val_fscore, val_mcc, val_loss = {}, {}, {}, {}, {}, {}

        for metrics in metrics_list:
            trn_loss[metrics['type']] = metrics['train_loss']
            trn_acc[metrics['type']] = metrics['train_accuracy']
            trn_pre[metrics['type']] = metrics['train_precision']
            trn_rec[metrics['type']] = metrics['train_recall']
            trn_fscore[metrics['type']] = metrics['train_fscore']
            trn_mcc[metrics['type']] = metrics['train_mcc']

        for metrics in metrics_list:
            val_loss[metrics['type']] = metrics['val_loss']
            val_acc[metrics['type']] = metrics['val_accuracy']
            val_pre[metrics['type']] = metrics['val_precision']
            val_rec[metrics['type']] = metrics['val_recall']
            val_fscore[metrics['type']] = metrics['val_fscore']
            val_mcc[metrics['type']] = metrics['val_mcc']

        plot_metrics('Accuracy', trn_acc, val_acc, epochs=metrics['epochs'])
        plot_metrics('Precision', trn_pre, val_pre, epochs=metrics['epochs'])
        plot_metrics('Recall', trn_rec, val_rec, epochs=metrics['epochs'])
        plot_metrics('FScore', trn_fscore, val_fscore, epochs=metrics['epochs'])
        plot_metrics('MCC', trn_mcc, val_mcc, epochs=metrics['epochs'])
        plot_metrics('Loss', trn_loss, val_loss, epochs=metrics['epochs'])

            
            
            
            
            
            
            
            
            