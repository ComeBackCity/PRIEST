from cProfile import label
import os, sys

sys.path.append(os.path.abspath("/home/gourab/Desktop/Projects/Covid/Tempel-modified/source_code"))
from src.models import models, train_model, our_models, predict_model, cnn_model, our_model_2, lstm_with_attn, \
    lstm_whole, lstm_corrected, lstm_whole_refactored, lstm_simplified, mcma
from src.data import make_dataset
from src.features import build_features
from src.utils import utils
import torch
import numpy as np
import sys
from src.utils import validation
from scipy.stats import mode
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve as ROC_curve , precision_recall_curve

# SEQ_LENGTH = sys.argv[1]
  
def mostCommon(lst):
      
    val, _ = mode(lst, axis = 0)
    return val.ravel().tolist()

def plot_attention(weights, path, quarters, xlabel, ylabel, title):
    """
    Plots attention weights in a grid.
    """
    # if ylabel == 'Quarter':
    #     w_min = torch.min(weights)
    #     w_max = torch.max(weights)
    #     weights = (weights - w_min) / (w_max - w_min)
    # else:
    #     w_min = torch.min(weights, dim=1)[0].reshape(-1, 1)
    #     w_max = torch.max(weights, dim=1)[0].reshape(-1, 1)
    #     weights = (weights - w_min) / (w_max - w_min)
    #     print(weights)

    plt.style.use('ggplot')
    cax = plt.matshow(weights.numpy(), cmap='BuGn')
    plt.rcParams.update({'font.size': 10})
    plt.colorbar(cax)
    plt.grid(
        visible=False,
        axis='both',
        which='both',
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks = range(0, 6), labels = quarters, rotation = 45)
    if ylabel == 'Quarter':
        plt.yticks(ticks = range(0, 6), labels = quarters, rotation = 75)
    plt.title(title, pad=15)
    # plt.xticks(ticks = tickvalues ,labels = labellist, rotation = 'vertical')
    #plt.savefig('./reports/figures/attention_weights.png')
    plt.savefig(path, format='pdf', dpi=1200)
    plt.close()

def plot_position_wise_attention(weights, path, quarters, positions, xlabel, ylabel, title):
    
    plt.style.use('ggplot')
    cax = plt.matshow(weights.numpy(), cmap='BuGn')
    # plt.rcParams.update({'font.size': 10})
    plt.colorbar(cax)
    plt.grid(
        visible=False,
        axis='both',
        which='both',
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks = range(0, 6), labels = quarters, rotation = 45)
    plt.yticks(ticks = range(0, len(positions)), labels = positions, rotation = 0)
    plt.title(title, pad=15)
    # plt.xticks(ticks = tickvalues ,labels = labellist, rotation = 'vertical')
    #plt.savefig('./reports/figures/attention_weights.png')
    plt.savefig(path, format='pdf', dpi=1200)
    plt.close()

def plot_pr_curve(result_table):

    plt.style.use('ggplot')
    # save_path = f'results_cov/{SEQ_LENGTH}/plots/metrics/'
    save_path = f'results_cov/2023/plots/metrics'

    fig = plt.figure(figsize=(8,6))

    for i in result_table.index:
        model = result_table.loc[i]['Model']
        print(model)
        if model == 'baseline':
            plt.plot(result_table.loc[i]['Recall'], 
                    result_table.loc[i]['Precision'], 
                    label=f'{model}',
                    color = 'orange',
                    linestyle='--')
        else:
            plt.plot(result_table.loc[i]['Recall'], 
                    result_table.loc[i]['Precision'], 
                    label=f'{model}')
        
    # plt.plot([0,0], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Recall", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("Precision", fontsize=15)

    plt.title('PR Curve', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower left')

    plt.autoscale()
    plt.savefig(save_path + f'/pr_curve.pdf', format='pdf', dpi=1200)


def plot_roc_curve(result_table):

    plt.style.use('ggplot')
    # save_path = f'results_cov/{SEQ_LENGTH}/plots/metrics'
    save_path = f'results_cov/2023/plots/metrics'

    fig = plt.figure(figsize=(8,6))

    for i in result_table.index:
        model = result_table.loc[i]['Model']
        if model == 'baseline':
            plt.plot(result_table.loc[i]['fpr'], 
                    result_table.loc[i]['tpr'], 
                    label=f'{model}',
                    color = 'orange',
                    linestyle='--')
        else:
            plt.plot(result_table.loc[i]['fpr'], 
                    result_table.loc[i]['tpr'], 
                    label=f'{model}')
        
    # plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.autoscale()
    plt.savefig(save_path + f'/roc_curve.pdf', format='pdf', dpi=1200)
    


def main():
    # base = '/home/gourab/Desktop/Projects/Covid/Tempel-modified/'

    # data_set = base + f'source_code/data/quarter_processed_weighted_2000_{SEQ_LENGTH}/cov'
    # data_path = base + f'source_code/data/quarter_processed_weighted_2000_{SEQ_LENGTH}/'
    
    base = '/home/gourab/Desktop/Projects/Covid/Tempel-modified/source_code/'

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
      'learning_rate': 0.0001,
    
      # Size of mini batch
      'batch_size': 32,
    
      # Number of training iterations
      'num_of_epochs': 500
    }
    
    print(f'lr = {parameters["learning_rate"]}, batch size = {parameters["batch_size"]}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(1)
    np.random.seed(1)
    if position_type == 'epitopes':
        test_trigram_vecs, test_labels = utils.read_dataset(parameters['data_set'] + '_test.csv', parameters['data_path'], concat=False)
    else:
        test_trigram_vecs, test_labels = utils.read_dataset(parameters['data_set'] + '_test_' + str(position) + '.csv', parameters['data_path'], concat=False)
    X_test = torch.tensor(test_trigram_vecs, dtype=torch.float32)
    Y_test = torch.tensor(test_labels, dtype=torch.int64)

    if parameters['model'] == 'baseline':
        true_labels = np.array(test_labels)
        counts = np.bincount(test_labels)
        majority_label = np.argmax(counts)
        labels = np.full_like(true_labels, fill_value=majority_label)
        pr_curve = precision_recall_curve(true_labels, labels)
        roc_curve = ROC_curve(true_labels, labels)
        return labels, true_labels, pr_curve, roc_curve
    #     # precision = precision_score(true_labels, labels)
    #     # recall = recall_score(true_labels, labels)
    #     # fscore = f1_score(true_labels, labels, )

    #     # labels, precision, recall, fscore, mcc, acc, pr_curve, roc_curve = 
          
            
    if parameters['model'] == 'svm':
        window_size = 1
        train_model.logistic_regression_baseline(
            build_features.reshape_to_linear(test_trigram_vecs, window_size=window_size), test_labels)
    elif parameters['model'] == 'random forest':
        window_size = 1
        train_model.random_forest_baseline(
            build_features.reshape_to_linear(test_trigram_vecs, window_size=window_size), test_labels)  
    elif parameters['model'] == 'logistic regression':
        window_size = 1
        train_model.bayes_baseline(
            build_features.reshape_to_linear(test_trigram_vecs, window_size=window_size), test_labels) 
    else:
        input_dim = X_test.shape[2]
        print(input_dim)
        seq_length = X_test.shape[0]
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
            windows = [i+1 for i in range(seq_length)]
            net = our_model_2.TimeSeriesTransformer(
                input_size=input_dim,
                enc_seq_len=seq_length,
                output_dim=output_dim,
                window_size = windows
            )
        elif parameters['model'] == 'cnn':
            net = cnn_model.CnnTextClassifier(
                num_classes=2,
                window_sizes=[1, 2, 3]
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
        elif parameters['model'] == 'PRIEST':
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

        labels, precision, recall, fscore, mcc, acc, pr_curve, roc_curve, sam, auroc, auprc = \
            predict_model.test_model(net, parameters['model'], X_test, Y_test, device)

        # print('Ensemble')
        print(f'Precision:{precision:.3f}')
        print(f'Recall:{recall:.3f}')
        print(f'Fscore:{fscore:.3f}')
        print(f'mcc:{mcc:.3f}')
        print(f'acc:{acc:.3f}')
        print(f'auroc:{auroc:.3f}')
        print(f'auprc:{auprc:.3f}')

        return labels, Y_test, pr_curve, roc_curve, sam
            

        
if __name__ == '__main__':
    # subtype = ['H1N1', 'H3N2', 'H5N1']
    position_mode = ['epitopes', 'single']   #two mode for position mode selection
    # data_path = '/home/gourab/Desktop/Projects/CoV research/Tempel-modified/source_code/data/test_covs/3/1.csv/'
    position_type = position_mode[0]   #select the predicting mode for all epitope sites or single site
    pr_df = pd.DataFrame(columns=['Model', 'Precision', 'Recall'])
    roc_df = pd.DataFrame(columns=['Model', 'fpr', 'tpr'])
    if position_type == 'epitopes':
        # model_list = ['our model']
        # model_list = ['mcma5']
        model_list = ['PRIEST', 'tempo', 'cnn', 'attention', 'gru', 'lstm', 'da-rnn', 'rnn', 'baseline']
        # model_list = ['PRIEST']
        preds = []
        true_labels = None
        for model in model_list:
            print('\n')
            print("Experimental results with model %s on cov:" % (model))
            if model == 'baseline':
                pred, true, pr_curve, roc_curve = main()
            else:
                pred, true, pr_curve, roc_curve, sam = main()
            prec, rec, _ = pr_curve
            pr_df = pr_df.append({'Model': model,
                                    'Precision':prec, 
                                    'Recall':rec}, ignore_index=True)
            fpr, tpr, _ = roc_curve
            roc_df = roc_df.append({'Model': model,
                                    'fpr':fpr, 
                                    'tpr':tpr}, ignore_index=True)
            if true_labels is None:
                true_labels = true
            torch.cuda.empty_cache()

        true_labels = np.array(true_labels)
        counts = np.bincount(true_labels)
        majority_label = np.argmax(counts)
        baseline = len(true_labels[true_labels==majority_label]) / len(true_labels)
        # pr_df["Precision"] = pr_df["Precision"].apply(lambda x: ", ".join(map(str, x)))
        # pr_df["Recall"] = pr_df["Recall"].apply(lambda x: ", ".join(map(str, x)))
        # print(pr_df)
        # roc_df["tpr"] = roc_df["tpr"].apply(lambda x: ", ".join(map(str, x)))
        # roc_df["fpr"] = roc_df["fpr"].apply(lambda x: ", ".join(map(str, x)))
        pr_df["Precision"] = pr_df["Precision"]
        pr_df["Recall"] = pr_df["Recall"]
        print(pr_df)
        roc_df["tpr"] = roc_df["tpr"]
        roc_df["fpr"] = roc_df["fpr"]
        # pr_df.to_csv(f'results_cov/tests/pr_2023.csv')
        # roc_df.to_csv(f'results_cov/tests/roc_2023.csv')
        plot_pr_curve(pr_df)
        plot_roc_curve(roc_df)
        # layer_x_year = []
        # # print(sam.shape)
        # quarters = ['Q3 2020', 'Q4 2020', 'Q1 2021', 'Q2 2021', 'Q3 2021', 'Q4 2021']
        # seq_idx = [178,  57, 172,  56 , 97]
        # seq_attn_maps = [[] for i in range(5)]
        # for idx, am in enumerate(sam):
        #     print(f'Plotting attn map {idx}')
        #     print(am.shape)
        #     for i, pos in enumerate(seq_idx):
        #         labels = true_labels[pos*149:(pos+1)*149]
        #         seq_attn_maps[i].append(am[pos*149:(pos+1)*149, : ,:])
        #     am_2d = torch.sum(am, dim=0).cpu()
        #     am_1d = torch.sum(am_2d, dim=0).cpu()
        #     layer_x_year.append(am_1d)
        #     # plot_attention(am_2d, f'results_cov/attn/weight_layer_{idx}.pdf', quarters, xlabel='Quarter', ylabel='Quarter', title=f'Layer-{idx} Self Attention Map')

        # # print(len(seq_attn_maps))
        # # print(len(seq_attn_maps[0]))
        # # print(seq_attn_maps[0][0].shape)
        # epitopes = []
        # hotspots = []
        # with open('data/cov_epitopes/epitopes_sorted.txt', 'r') as f:
        #     for line in f:
        #         epitopes.append(int(line))
        
        # # with open('data/cov_epitopes/epitopes_hotspot.txt', 'r') as f:
        # #     for line in f:
        # #         hotspots.append(int(line))

        # # hotspots.sort()
        
        # for i, (idx, attn_maps) in enumerate(zip(seq_idx, seq_attn_maps)):
        #     dir = f'results_cov/attn/seq_wise/seq_{idx}/'
        #     if not os.path.exists(dir):
        #         os.makedirs(dir)
        #     for layer, attn_map in enumerate(attn_maps):
        #         pos_attn = torch.sum(attn_map, dim=1).cpu()
        #         # labels_seq = true_labels[idx*149:(idx+1)*149]
        #         # labels_seq_np = np.asarray(labels_seq)
        #         epitopes_np = np.asarray(epitopes)
        #         hotspot_idxs = np.in1d(epitopes_np, hotspots)
        #         # epitopes_filtered = epitopes_np[hotspot_idxs]
        #         pos_attn_filtered = pos_attn[hotspot_idxs]
        #         np.savetxt(dir+f'attn_map_layer_{layer}.txt', pos_attn)
        #         # plot_position_wise_attention(pos_attn, dir+f'attn_map_layer_{layer}.pdf', quarters , epitopes, xlabel='Quarter', ylabel='Mutation Sites', title=f'Layer-{layer+1} Self Attention Map')
        #         np.savetxt(dir+f'attn_map_layer_filtered_{layer}.txt', pos_attn_filtered)
        #         # plot_position_wise_attention(pos_attn_filtered, dir+f'attn_map_layer_filtered_{layer}.pdf', quarters , hotspots, xlabel='Quarter', ylabel='Mutation Sites', title=f'Layer-{layer+1} Attention Map')
        # final_am = torch.cat(layer_x_year).cpu()
        # final_am = final_am.view(8, 6)
        # plot_attention(final_am, f'results_cov/attn/final_attn_map.pdf', quarters, xlabel='Quarter', ylabel='Layer', title='Layerwise Attention Map')

        # # model_list = ['attention']
        # for model in model_list:
        #     main()

        # print('Ensemble')
        # ensemble_preds = mostCommon(preds)
        # precision, recall, fscore, mcc, accuracy = validation.evaluate(true_labels, ensemble_preds)
        # print(f'Precision:{precision:.3f}')
        # print(f'Recall:{recall:.3f}')
        # print(f'Fscore:{fscore:.3f}')
        # print(f'mcc:{mcc:.3f}')
        # print(f'acc:{accuracy:.3f}')
        
        



            
            
            
            
            
            
            
            