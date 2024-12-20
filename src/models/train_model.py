from statistics import mode
import torch
import torch.nn.functional as F
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.linear_model import LinearRegression
# from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from src.utils import utils, validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import interp
import os
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, \
                        get_cosine_with_hard_restarts_schedule_with_warmup, \
                          get_linear_schedule_with_warmup

from src.models.lstm_whole_refactored import CustomCategoricalCrossEntropyLoss
import seaborn as sns
from colorama import Fore, Style
from torch.utils.tensorboard import SummaryWriter

import sys
# SUBTYPE = sys.argv[1]

def repackage_hidden(h):
  """
  Wraps hidden states in new Tensors, to detach them from their history.
  """
  if isinstance(h, torch.Tensor):
    return h.detach()
  else:
    return tuple(repackage_hidden(v) for v in h)


def plot_training_history(loss, val_loss, acc, val_acc, mini_batch_scores, mini_batch_labels):
  """
  Plots the loss and accuracy for training and validation over epochs.
  Also plots the logits for a small batch over epochs.
  """
  plt.style.use('ggplot')
    
  # Plot losses
  plt.figure()
  plt.subplot(1,3,1)
  plt.plot(loss, 'b', label='Training')
  plt.plot(val_loss, 'r', label='Validation')
  plt.title('Loss')
  plt.legend()

  # Plot accuracies
  plt.subplot(1,3,2)
  plt.plot(acc, 'b', label='Training')
  plt.plot(val_acc, 'r', label='Validation')
  plt.title('Accuracy')
  plt.legend()

  # Plot prediction dynamics of test mini batch
  plt.subplot(1,3,3)
  pos_label, neg_label = False, False
  for i in range(len(mini_batch_labels)):
    if mini_batch_labels[i]:
      score_sequence = [x[i][1] for x in mini_batch_scores]
      if not pos_label:
        plt.plot(score_sequence, 'b', label='Pos')
        pos_label = True
      else:
        plt.plot(score_sequence, 'b')
    else:
      score_sequence = [x[i][0] for x in mini_batch_scores]
      if not neg_label:
        plt.plot(score_sequence, 'r', label='Neg')
        neg_label = True
      else:
        plt.plot(score_sequence, 'r')
  
  plt.title('Logits')
  plt.legend()
  #plt.savefig('./reports/figures/training_curves.png')
  plt.savefig('/home/gourab/Projects/CoV research/Tempel-modified/source_code/data/cov_figure/loss/loss_fig.eps', dpi=350)


def plot_attention(weights):
  """
  Plots attention weights in a grid.
  """
  cax = plt.matshow(weights.numpy(), cmap='bone')
  plt.colorbar(cax)
  plt.grid(
    b=False,
    axis='both',
    which='both',
  )
  plt.xlabel('Years')
  plt.ylabel('Examples')
  #plt.savefig('./reports/figures/attention_weights.png')
  plt.savefig('/home/gourab/Projects/CoV research/Tempel-modified/source_code/data/cov_figure/attention/weight.eps', dpi=350)


def predictions_from_output(scores):
  """
  Maps logits to class predictions.
  """
  prob = F.softmax(scores, dim=1)
  # print(prob)
  _, predictions = prob.topk(1)
  return predictions

def calculate_prob(scores):
  """
  Maps logits to class predictions.
  """
  prob = F.softmax(scores, dim=1)
  pred_probe, _ = prob.topk(1)
  return pred_probe

def verify_model(model, X, Y, batch_size):
  """
  Checks the loss at initialization of the model and asserts that the
  training examples in a batch aren't mixed together by backpropagating.
  """
  print('Sanity checks:')
  criterion = torch.nn.CrossEntropyLoss()
  scores, _ = model(X, model.init_hidden(Y.shape[0]))
  print(' Loss @ init %.3f, expected ~%.3f' % (criterion(scores, Y).item(), -math.log(1 / model.output_dim)))


  mini_batch_X = X[:, :batch_size, :]
  mini_batch_X.requires_grad_()
  criterion = torch.nn.MSELoss()
  scores, _ = model(mini_batch_X, model.init_hidden(batch_size))

  non_zero_idx = 1
  perfect_scores = [[0, 0] for i in range(batch_size)]
  not_perfect_scores = [[1, 1] if i == non_zero_idx else [0, 0] for i in range(batch_size)]

  scores.data = torch.FloatTensor(not_perfect_scores)
  Y_perfect = torch.FloatTensor(perfect_scores)
  loss = criterion(scores, Y_perfect)
  loss.backward()

  zero_tensor = torch.FloatTensor([0] * X.shape[2])
  for i in range(mini_batch_X.shape[0]):
    for j in range(mini_batch_X.shape[1]):
      if sum(mini_batch_X.grad[i, j] != zero_tensor):
        assert j == non_zero_idx, 'Input with loss set to zero has non-zero gradient.'

  mini_batch_X.detach()
  print(' Backpropagated dependencies OK')

def save_best_model(val_loss, best_val_loss, epoch_loss, best_training_loss, checkpoint, best_path):
  if val_loss < best_val_loss:
    print(f'validation loss improved by {best_val_loss - val_loss}')
    best_val_loss = val_loss
    best_training_loss = epoch_loss
    print('Saving best model :-)')
    torch.save(checkpoint, best_path + 'checkpoint.pth')
  elif val_loss == best_val_loss and epoch_loss <= best_training_loss:
    print(f'training loss improved by {best_training_loss - epoch_loss}')
    best_training_loss = epoch_loss
    print('Saving best model :-)')
    torch.save(checkpoint, best_path + 'checkpoint.pth')

  return best_val_loss, best_training_loss

def save_current_model(checkpoint, resume_directory):
  print('Saving current model')
  Style.RESET_ALL
  torch.save(checkpoint, resume_directory + 'checkpoint.pth')

def save_best_model_auc(val_auc, best_val_auc, checkpoint, best_path):
  if val_auc > best_val_auc:
    print(f'validation auc improved by {val_auc - best_val_auc}')
    best_val_auc = val_auc
    print('Saving best model :-)')
    torch.save(checkpoint, best_path + 'checkpoint.pth')

  return best_val_auc

def train_cnn(model, verify, epochs, learning_rate, batch_size, X, Y, X_test, Y_test, show_attention,cell_type, device, retrain = False, resume = False):
  """
  Training loop for a model utilizing hidden states.

  verify enables sanity checks of the model.
  epochs decides the number of training iterations.
  learning rate decides how much the weights are updated each iteration.
  batch_size decides how many examples are in each mini batch.
  show_attention decides if attention weights are plotted.
  """

  # def init_weights(m):
  #   if isinstance(m, torch.nn.Linear):
  #       torch.nn.init.xavier_uniform(m.weight)
  #       m.bias.data.fill_(0.01)

  writer = SummaryWriter(f'tensorboard/{cell_type}/9/')
  model.to(device)
  print_interval = 1

  optimizer1 = torch.optim.Adam(
    model.parameters(), 
    lr=learning_rate,
    # weight_decay = 0.001,
    # amsgrad = True
  )

  optimizer2 = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=0.9,
    nesterov=True
  )

  optimizer = optimizer1

  print(optimizer.param_groups[0]['lr'])
  # print(optimizer2.param_groups[0]['lr'])

  scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer = optimizer,
    patience = 5,
    cooldown = 3,
    factor=0.25,
    threshold = 1e-4,
    min_lr = 1e-8,
    verbose=True
  )

  # scheduler2 = torch.optim.lr_scheduler.MultiStepLR(
  #   optimizer = optimizer,
  #   gamma = 0.1,
  #   milestones = [1, 50, 75, 100, 125, 150, 175],
  #   verbose = True
  # )

  # scheduler3 = torch.optim.lr_scheduler.ExponentialLR(
  #   optimizer = optimizer,
  #   gamma = 0.2,
  #   verbose = True
  # )

  # scheduler4 = torch.optim.lr_scheduler.CyclicLR(
  #   optimizer=optimizer,
  #   base_lr= 0,
  #   max_lr = learning_rate,
  #   mode='exp_range',
  #   step_size_up = 20,
  #   step_size_down = 5,
  #   cycle_momentum=False,
  #   verbose=True
  # )

  # scheduler4_triangle = torch.optim.lr_scheduler.CyclicLR(
  #   optimizer=optimizer,
  #   base_lr= 0,
  #   max_lr = learning_rate,
  #   mode='triangular',
  #   gamma=0.9,
  #   step_size_up = 30,
  #   step_size_down = 15,
  #   cycle_momentum=False,
  #   verbose=True
  # )

  # scheduler5 = get_cosine_schedule_with_warmup(
  #   optimizer=optimizer,
  #   num_warmup_steps=30,
  #   num_training_steps=200
  # )

  # scheduler6 = get_cosine_with_hard_restarts_schedule_with_warmup(
  #   optimizer=optimizer,
  #   num_warmup_steps=50,
  #   num_training_steps=300,
  #   num_cycles=5
  # )

  # scheduler7 = get_constant_schedule_with_warmup(
  #   optimizer=optimizer,
  #   num_warmup_steps=50
  # )

  # scheduler8 = get_linear_schedule_with_warmup(
  #   optimizer=optimizer,
  #   num_warmup_steps=30, 
  #   num_training_steps=epochs
  # )

  # linear = get_linear_schedule_with_warmup(
  #   optimizer=optimizer,
  #   num_warmup_steps=1000, 
  #   num_training_steps=2000
  # )

  lr_scheduler = scheduler1
  # lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
  #   [
  #     scheduler4,
  #     scheduler1
  #   ]
  # )

  mse = torch.nn.MSELoss(reduction='mean')
  ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

  criterion = ce

  num_of_examples = X.shape[1]
  seq_length = X.shape[0]
  num_of_batches = math.floor(num_of_examples/batch_size)

  if verify:
    verify_model(model, X, Y, batch_size)
  all_losses = []
  all_val_losses = []
  all_accs = []
  all_pres = []
  all_recs = []
  all_fscores = []
  all_mccs = []
  all_val_accs = []
  all_val_pres = []
  all_val_recs = []
  all_val_fscores = []
  all_val_mccs = []

  X_test = X_test.permute((1, 0, 2))
  # print(X_test.shape)

  # Find mini batch that contains at least one mutation to plot
  plot_batch_size = 10
  i = 0
  while not Y_test[i]:
    i += 1

  X_plot_batch = X_test[i:i+plot_batch_size, :, :]
  # X_plot_batch = X_plot_batch.permute((1, 0, 2))
  Y_plot_batch = Y_test[i:i+plot_batch_size]
  plot_batch_scores = []

  # Save checkpoint
  best_val_loss = 9999.0
  best_training_loss = 9999.0

  # best_path = f'checkpoint/{seq_length}/{cell_type}/best_model/'
  # resume_directory = f'checkpoint/{seq_length}/{cell_type}/resume/'
  # best_path = f'checkpoint/{SUBTYPE}/{cell_type}/best_model/'
  # resume_directory = f'checkpoint/{SUBTYPE}/{cell_type}/resume/'
  best_path = f'checkpoint/2023/{cell_type}/best_model/'
  resume_directory = f'checkpoint/2023/{cell_type}/resume/'
  
  if not os.path.exists(best_path):
    os.makedirs(best_path)
  elif retrain:
    print('Retraining')
    ckpt = torch.load(best_path+'checkpoint.pth')
    model.load_state_dict(ckpt['state_dict'])
    best_val_loss = ckpt['best_val_loss']
    best_training_loss = ckpt['best_training_loss']

  start_epoch = 0

  if not os.path.exists(resume_directory):
    os.makedirs(resume_directory)
  elif resume:
    print('Resuming training')
    ckpt = torch.load(resume_directory+'checkpoint.pth')
    if ckpt['epoch'] <= ckpt['total_epoch']:
      start_epoch = ckpt['epoch'] 
      model.load_state_dict(ckpt['state_dict'])
      optimizer.load_state_dict(ckpt['optimizer'])
      lr_scheduler.load_state_dict(ckpt['scheduler'])
      if 'best_val_loss' in ckpt:
        best_val_loss = ckpt['best_val_loss']
      if 'best_training_loss' in ckpt:
        best_training_loss = ckpt['best_training_loss']

      
  # print(best_val_loss)
  # exit()

  # for g in optimizer.param_groups:
  #   g['lr'] = learning_rate

  print(optimizer.param_groups[0]['lr'])

  start_time = time.time()
  
  for epoch in range(start_epoch, epochs):
    print(epoch)
    model.train()
    running_loss = 0
    running_acc = 0
    running_pre = 0
    running_pre_total = 0
    running_rec = 0
    running_rec_total = 0
    epoch_fscore = 0
    running_mcc_numerator = 0
    running_mcc_denominator = 0
    running_rec_total = 0

    # hidden = model.init_hidden(batch_size)

    for count in range(0, num_of_examples - batch_size + 1, batch_size):
      # repackage_hidden(hidden)

      X_batch = X[:, count:count+batch_size, :]
      # print(X_batch.shape)
      X_batch = X_batch.permute((1, 0, 2))
      X_batch = X_batch.to(device)
      Y_batch = Y[count:count+batch_size]
      Y_batch = Y_batch.to(device)
      
      scores, _ = model(X_batch, debug=False)
      # print(scores)
      # print(scores.shape)
      # print(Y_batch.shape)
      # print(scores)
      # exit()
      loss = criterion(scores, Y_batch)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      predictions = predictions_from_output(scores)
      # print(predictions.shape)
      # exit()
      
      conf_matrix = validation.get_confusion_matrix(Y_batch, predictions)
      TP, FP, FN, TN = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
      running_acc += TP + TN
      running_pre += TP
      running_pre_total += TP + FP
      running_rec += TP
      running_rec_total += TP + FN
      running_mcc_numerator += (TP * TN - FP * FN)
      if ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) == 0:
          running_mcc_denominator += 0
      else:
          running_mcc_denominator += math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
      running_loss += loss.item()

    elapsed_time = time.time() - start_time
    epoch_acc = running_acc / Y.shape[0]
    all_accs.append(epoch_acc)
    
    if running_pre_total == 0:
        epoch_pre = 0
    else:
        epoch_pre = running_pre / running_pre_total
    all_pres.append(epoch_pre)
    
    if running_rec_total == 0:
        epoch_rec = 0
    else:
        epoch_rec = running_rec / running_rec_total
    all_recs.append(epoch_rec)
    
    if (epoch_pre + epoch_rec) == 0:
        epoch_fscore = 0
    else:
        epoch_fscore = 2 * epoch_pre * epoch_rec / (epoch_pre + epoch_rec)
    all_fscores.append(epoch_fscore)
    
    if running_mcc_denominator == 0:
        epoch_mcc = 0
    else:
        epoch_mcc = running_mcc_numerator / running_mcc_denominator
    all_mccs.append(epoch_mcc)
    
    epoch_loss = running_loss / num_of_batches
    all_losses.append(epoch_loss)

    writer.add_scalar('Loss/train', epoch_loss, epoch)

    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    with torch.no_grad():
      model.eval()
      # print(X_test.shape)
      test_scores, _ = model(X_test, debug=False)
      # print(test_scores.shape)
      predictions = predictions_from_output(test_scores) 
      # print(predictions.shape)     
      predictions = predictions.view_as(Y_test)
      # print(predictions.shape)
      pred_prob = calculate_prob(test_scores)
      # print(pred_prob.shape)
      # exit()
      precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, predictions)
      
      val_loss = criterion(test_scores, Y_test).item()
      all_val_losses.append(val_loss)
      all_val_accs.append(val_acc)
      all_val_pres.append(precision)
      all_val_recs.append(recall)
      all_val_fscores.append(fscore)
      all_val_mccs.append(mcc)

      X_plot_batch = X_plot_batch.to(device)
      plot_scores, _ = model(X_plot_batch, debug=False)
      plot_batch_scores.append(plot_scores)

    writer.add_scalar('Loss/val', val_loss, epoch)

    if (epoch+1) % print_interval == 0:
      print('Epoch %d Time %s' % (epoch, utils.get_time_string(elapsed_time)))
      print('T_loss %.4f\tT_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f' %(epoch_loss, epoch_acc, epoch_pre, epoch_rec, epoch_fscore, epoch_mcc))
      print('V_loss %.4f\tV_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'%(val_loss, val_acc, precision, recall, fscore, mcc))

    # print('Applying lr scheduler')
    # lr_scheduler.step(val_loss)
    lr_scheduler.step(val_loss)
    # scheduler1.step(val_loss)

    checkpoint = {
      'epoch' : epoch + 1,
      'total_epoch' : epochs+start_epoch,
      'state_dict' : model.state_dict(),
      'optimizer' : optimizer.state_dict(),
      'scheduler' : lr_scheduler.state_dict(),
      'best_training_loss' : best_training_loss,
      'best_val_loss' : best_val_loss
    }

    save_current_model(
      checkpoint=checkpoint,
      resume_directory=resume_directory
    )

    best_val_loss, best_training_loss = save_best_model(
                                          val_loss=val_loss,
                                          best_val_loss=best_val_loss,
                                          epoch_loss=epoch_loss,
                                          best_training_loss=best_training_loss,
                                          checkpoint=checkpoint,
                                          best_path=best_path
                                        )

    # print(f'latest learning rate {lr_scheduler.get_last_lr()[-1]}')
    last_lr = optimizer.param_groups[0]['lr']
    print(last_lr)
    writer.add_scalar('learning_rate', last_lr, epoch)
    # print(f'latest learning rate {last_lr}')

  metrics = {
    'type': cell_type,
    'train_loss' : all_losses,
    'val_loss' : all_val_losses,
    'train_accuracy' : all_accs,
    'val_accuracy' : all_val_accs,
    'train_precision' : all_pres,
    'val_precision' : all_val_pres,
    'train_recall' : all_recs,
    'val_recall' : all_val_recs,
    'train_fscore' : all_fscores,
    'val_fscore' : all_val_fscores,
    'train_mcc' : all_mccs,
    'val_mcc' : all_val_mccs,
    'epochs' : epochs
  }

  return metrics
    # print('Saving current model')
    # torch.save(checkpoint, resume_directory + 'checkpoint.pth')
    # if val_loss < best_val_loss:
    #   best_val_loss = val_loss
    #   best_training_loss = epoch_loss
    #   print('Saving best model')
    #   torch.save(checkpoint, best_path + 'checkpoint.pth')
    # elif val_loss == best_val_loss and epoch_loss < best_training_loss:
    #   best_training_loss = epoch_loss
    #   print('Saving best model')
    #   torch.save(checkpoint, best_path + 'checkpoint.pth')
    # scheduler2.step()
    # lr_scheduler.step()

    # torch.cuda.empty_cache()
  #plot_training_history(all_losses, all_val_losses, all_accs, all_val_accs, plot_batch_scores, Y_plot_batch)

      #roc curve
  if epoch + 1 == 50:
    tpr_rnn, fpr_rnn, _ = roc_curve(Y_test.cpu(), pred_prob.cpu()) 
    print(auc(fpr_rnn, tpr_rnn))
    # plt.figure(1)
    # #plt.xlim(0, 0.8)
    # plt.ylim(0.5, 1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # if cell_type == 'lstm':
    #     plt.plot(fpr_rnn, tpr_rnn, label=cell_type)
    # elif cell_type == 'rnn':
    #     plt.plot(fpr_rnn, tpr_rnn, label=cell_type)
    # elif cell_type == 'gru':
    #     plt.plot(fpr_rnn, tpr_rnn, label='attention')
    # elif cell_type == 'attention':
    #     plt.plot(fpr_rnn, tpr_rnn, label='gru')
    # plt.legend(loc='best')
      
  if show_attention:
    with torch.no_grad():
      model.eval()
      _, attn_weights = model(X_plot_batch)
      # plot_attention(attn_weights.cpu())
  #plt.show()



def train_transformer(model, verify, epochs, learning_rate, batch_size, X, Y, X_test, Y_test, show_attention,cell_type, device, retrain = False, resume = False):
  """
  Training loop for a model utilizing hidden states.

  verify enables sanity checks of the model.
  epochs decides the number of training iterations.
  learning rate decides how much the weights are updated each iteration.
  batch_size decides how many examples are in each mini batch.
  show_attention decides if attention weights are plotted.
  """

  # print(torch.backends.cudnn.enabled)


  def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

  seq_length = X.shape[0]
  
  print(seq_length)
  

  writer = SummaryWriter(f'tensorboard/{cell_type}/{seq_length}/')
  model.to(device)
  print_interval = 1
  optimizer1 = torch.optim.Adam(
    model.parameters(), 
    lr=learning_rate,
    weight_decay = 1e-5,
    # amsgrad = True
  )

  # optimizer2 = torch.optim.AdamW(
  #   model.parameters(), 
  #   lr=learning_rate,
  #   # weight_decay=config['wd'],
  #   amsgrad = True
  # )


  # optimizer3 = torch.optim.SGD(
  #   model.parameters(),
  #   lr=learning_rate,
  #   weight_decay = 1e-6,
  #   momentum=0.9,
  #   nesterov=True
  # )

  optimizer = optimizer1

  print(optimizer.param_groups[0]['lr'])
  # print(optimizer2.param_groups[0]['lr'])

  # scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
  #   optimizer = optimizer,
  #   patience = 5,
  #   cooldown = 3,
  #   factor=0.1,
  #   threshold = 1e-4,
  #   min_lr = 1e-8,
  #   verbose=True
  # )

  # scheduler2 = torch.optim.lr_scheduler.MultiStepLR(
  #   optimizer = optimizer,
  #   gamma = 0.1,
  #   milestones = [5, 20],
  #   verbose = True
  # )

  # scheduler3 = torch.optim.lr_scheduler.ExponentialLR(
  #   optimizer = optimizer,
  #   gamma = 0.2,
  #   verbose = True
  # )

  # scheduler4_triangle = torch.optim.lr_scheduler.CyclicLR(
  #   optimizer=optimizer,
  #   base_lr= 0,
  #   max_lr = learning_rate,
  #   mode='triangular',
  #   gamma=0.9,
  #   step_size_up = 20,
  #   step_size_down = 5,
  #   cycle_momentum=False,
  #   verbose=True
  # )

  scheduler4_exp = torch.optim.lr_scheduler.CyclicLR(
    optimizer=optimizer,
    base_lr= 0,
    max_lr = learning_rate,
    mode='exp_range',
    # gamma=0.9,
    step_size_up = 40,
    step_size_down = 10,
    cycle_momentum=False,
    verbose=True
  )

  # scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(
  #   optimizer=optimizer,
  #   gamma=0.1,
  #   verbose= True
  # )

  # scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
  #   optimizer=optimizer,
  #   T_max=20,
  #   eta_min=1e-6,
  #   verbose=True
  # )

  stop_cyclic = False
  prev_lr = 0.0

  # scheduler5 = get_cosine_schedule_with_warmup(
  #   optimizer=optimizer,
  #   num_warmup_steps=30,
  #   num_training_steps=200
  # )

  # scheduler6 = get_cosine_with_hard_restarts_schedule_with_warmup(
  #   optimizer=optimizer,
  #   num_warmup_steps=50,
  #   num_training_steps=300,
  #   num_cycles=5
  # )

  # scheduler7 = get_constant_schedule_with_warmup(
  #   optimizer=optimizer,
  #   num_warmup_steps=15
  # )

  # scheduler8 = get_linear_schedule_with_warmup(
  #   optimizer=optimizer,
  #   num_warmup_steps=20, 
  #   num_training_steps=epochs
  # )

  # linear = get_linear_schedule_with_warmup(
  #   optimizer=optimizer,
  #   num_warmup_steps=1000, 
  #   num_training_steps=2000
  # )

  lr_scheduler = scheduler4_exp

  mse = torch.nn.MSELoss(reduction='mean')
  ce = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
  cus_ce = CustomCategoricalCrossEntropyLoss()

  print(model)

  criterion = ce

  num_of_examples = X.shape[1]
  num_of_batches = math.floor(num_of_examples/batch_size)
  val_examples = X_test.shape[1]

  if verify:
    verify_model(model, X, Y, batch_size)
  all_losses = []
  all_val_losses = []
  all_accs = []
  all_pres = []
  all_recs = []
  all_fscores = []
  all_mccs = []
  all_val_accs = []
  all_val_pres = []
  all_val_recs = []
  all_val_fscores = []
  all_val_mccs = []

  X_test = X_test.permute((1, 0, 2))
  print(X_test.shape)
  # print(X_test.shape)

  # Find mini batch that contains at least one mutation to plot
  plot_batch_size = 10
  i = 0
  while not Y_test[i]:
    i += 1

  X_plot_batch = X_test[i:i+plot_batch_size, :, :]
  # X_plot_batch = X_plot_batch.permute((1, 0, 2))
  Y_plot_batch = Y_test[i:i+plot_batch_size]
  plot_batch_scores = []

  # Save checkpoint
  best_val_loss = 1.0
  best_training_loss = 1.0
  best_val_auc = 0.0

  # best_path = f'checkpoint/{seq_length}/{cell_type}/best_model/'
  # resume_directory = f'checkpoint/{seq_length}/{cell_type}/resume/'
  # best_path = f'checkpoint/{SUBTYPE}/{cell_type}/best_model/'
  # resume_directory = f'checkpoint/{SUBTYPE}/{cell_type}/resume/'
  best_path = f'checkpoint/2023/{cell_type}/best_model/'
  resume_directory = f'checkpoint/2023/{cell_type}/resume/'
  
  if not os.path.exists(best_path):
    os.makedirs(best_path)
  elif retrain:
    print('Retraining')
    ckpt = torch.load(best_path+'checkpoint.pth')
    model.load_state_dict(ckpt['state_dict'])
    best_val_loss = ckpt['best_val_loss']
    best_training_loss = ckpt['best_training_loss']

  start_epoch = 0

  if not os.path.exists(resume_directory):
    os.makedirs(resume_directory)
  elif resume:
    print('Resuming training')
    ckpt = torch.load(resume_directory+'checkpoint.pth')
    if ckpt['epoch'] <= ckpt['total_epoch']:
      start_epoch = ckpt['epoch'] 
      model.load_state_dict(ckpt['state_dict'])
      optimizer.load_state_dict(ckpt['optimizer'])
      # lr_scheduler.load_state_dict(ckpt['scheduler'])
      if 'best_val_loss' in ckpt:
        best_val_loss = ckpt['best_val_loss']
      if 'best_training_loss' in ckpt:
        best_training_loss = ckpt['best_training_loss']

      
  # print(best_val_loss)
  # exit()

  # for g in optimizer.param_groups:
  #   g['lr'] = learning_rate

  print(optimizer.param_groups[0]['lr'])

  start_time = time.time()
  
  for epoch in range(start_epoch, epochs):
    model.train()
    running_loss = 0
    running_acc = 0
    running_pre = 0
    running_pre_total = 0
    running_rec = 0
    running_rec_total = 0
    epoch_fscore = 0
    running_mcc_numerator = 0
    running_mcc_denominator = 0
    running_rec_total = 0

    # hidden = model.init_hidden(batch_size)

    for count in range(0, num_of_examples - batch_size + 1, batch_size):
      # repackage_hidden(hidden)

      X_batch = X[:, count:count+batch_size, :]
      # print(X_batch.shape)
      X_batch = X_batch.permute((1, 0, 2))
      X_batch = X_batch.to(device)
      
      Y_batch = Y[count:count+batch_size]
      Y_batch = Y_batch.to(device)
      
      scores, _ = model(X_batch, debug=False)
      # print(scores)
      # print(scores.shape)
      # print(scores)
      # exit()
      loss = criterion(scores, Y_batch)

      optimizer.zero_grad()
      loss.backward()
      # torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=1)
      optimizer.step()

      predictions = predictions_from_output(scores)
      # print(predictions)
      # exit()
      
      conf_matrix = validation.get_confusion_matrix(Y_batch, predictions)
      TP, FP, FN, TN = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
      running_acc += TP + TN
      running_pre += TP
      running_pre_total += TP + FP
      running_rec += TP
      running_rec_total += TP + FN
      running_mcc_numerator += (TP * TN - FP * FN)
      if ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) == 0:
          running_mcc_denominator += 0
      else:
          running_mcc_denominator += math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
      running_loss += loss.item()

    elapsed_time = time.time() - start_time
    epoch_acc = running_acc / Y.shape[0]
    all_accs.append(epoch_acc)
    
    if running_pre_total == 0:
        epoch_pre = 0
    else:
        epoch_pre = running_pre / running_pre_total
    all_pres.append(epoch_pre)
    
    if running_rec_total == 0:
        epoch_rec = 0
    else:
        epoch_rec = running_rec / running_rec_total
    all_recs.append(epoch_rec)
    
    if (epoch_pre + epoch_rec) == 0:
        epoch_fscore = 0
    else:
        epoch_fscore = 2 * epoch_pre * epoch_rec / (epoch_pre + epoch_rec)
    all_fscores.append(epoch_fscore)
    
    if running_mcc_denominator == 0:
        epoch_mcc = 0
    else:
        epoch_mcc = running_mcc_numerator / running_mcc_denominator
    all_mccs.append(epoch_mcc)
    
    epoch_loss = running_loss / num_of_batches
    all_losses.append(epoch_loss)

    writer.add_scalar('Loss/train', epoch_loss, epoch)

    X_test = X_test
    Y_test = Y_test.to(device)
    
    print("validation")

    with torch.no_grad():
      logits_list = []
      model.eval()
      for i in range(0, val_examples, batch_size):
        batch = X_test[i:i+batch_size, :, :]
        batch = batch.to(device)
      
        test_scores, _ = model(batch, debug=False)

        logits_list.append(test_scores)
        torch.cuda.empty_cache()
      
      test_scores = torch.cat(logits_list, dim=0)
      predictions = predictions_from_output(test_scores)      
      predictions = predictions.view_as(Y_test)
      pred_prob = calculate_prob(test_scores)
      precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, predictions)
      
      val_loss = criterion(test_scores, Y_test).item()
      all_val_losses.append(val_loss)
      all_val_accs.append(val_acc)
      all_val_pres.append(precision)
      all_val_recs.append(recall)
      all_val_fscores.append(fscore)
      all_val_mccs.append(mcc)

      X_plot_batch = X_plot_batch.to(device)
      plot_scores, _ = model(X_plot_batch, debug=False)
      plot_batch_scores.append(plot_scores)
      
    # exit()

    writer.add_scalar('Loss/val', val_loss, epoch)

    if (epoch+1) % print_interval == 0:
      print('Epoch %d Time %s' % (epoch, utils.get_time_string(elapsed_time)))
      print('T_loss %.4f\tT_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f' %(epoch_loss, epoch_acc, epoch_pre, epoch_rec, epoch_fscore, epoch_mcc))
      print('V_loss %.4f\tV_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'%(val_loss, val_acc, precision, recall, fscore, mcc))

    # print('Applying lr scheduler')
    # lr_scheduler.step(val_loss)
    # if not stop_cyclic:
    #   lr_scheduler.step()

    lr_scheduler.step()

    # scheduler1.step(val_loss)

    checkpoint = {
      'epoch' : epoch + 1,
      'total_epoch' : epochs+start_epoch,
      'state_dict' : model.state_dict(),
      'optimizer' : optimizer.state_dict(),
      # 'scheduler' : lr_scheduler.state_dict(),
      'best_training_loss' : best_training_loss,
      'best_val_loss' : best_val_loss
    }

    save_current_model(
      checkpoint=checkpoint,
      resume_directory=resume_directory
    )

    best_val_loss, best_training_loss = save_best_model(
                                          val_loss=val_loss,
                                          best_val_loss=best_val_loss,
                                          epoch_loss=epoch_loss,
                                          best_training_loss=best_training_loss,
                                          checkpoint=checkpoint,
                                          best_path=best_path
                                        )

    # print(f'latest learning rate {lr_scheduler.get_last_lr()[-1]}')
    cur_lr = optimizer.param_groups[0]['lr']
    # if prev_lr > cur_lr and epoch > epochs // 2:
    #   stop_cyclic = True
    prev_lr = cur_lr
    writer.add_scalar('learning_rate', cur_lr, epoch)
    # print(f'latest learning rate {last_lr}')

  metrics = {
    'type': cell_type,
    'train_loss' : all_losses,
    'val_loss' : all_val_losses,
    'train_accuracy' : all_accs,
    'val_accuracy' : all_val_accs,
    'train_precision' : all_pres,
    'val_precision' : all_val_pres,
    'train_recall' : all_recs,
    'val_recall' : all_val_recs,
    'train_fscore' : all_fscores,
    'val_fscore' : all_val_fscores,
    'train_mcc' : all_mccs,
    'val_mcc' : all_val_mccs,
    'epochs' : epochs
  }

  return metrics
    # print('Saving current model')
    # torch.save(checkpoint, resume_directory + 'checkpoint.pth')
    # if val_loss < best_val_loss:
    #   best_val_loss = val_loss
    #   best_training_loss = epoch_loss
    #   print('Saving best model')
    #   torch.save(checkpoint, best_path + 'checkpoint.pth')
    # elif val_loss == best_val_loss and epoch_loss < best_training_loss:
    #   best_training_loss = epoch_loss
    #   print('Saving best model')
    #   torch.save(checkpoint, best_path + 'checkpoint.pth')
    # scheduler2.step()
    # lr_scheduler.step()

    # torch.cuda.empty_cache()
  #plot_training_history(all_losses, all_val_losses, all_accs, all_val_accs, plot_batch_scores, Y_plot_batch)

      #roc curve
  if epoch + 1 == 50:
    tpr_rnn, fpr_rnn, _ = roc_curve(Y_test.cpu(), pred_prob.cpu()) 
    print(auc(fpr_rnn, tpr_rnn))
    # plt.figure(1)
    # #plt.xlim(0, 0.8)
    # plt.ylim(0.5, 1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # if cell_type == 'lstm':
    #     plt.plot(fpr_rnn, tpr_rnn, label=cell_type)
    # elif cell_type == 'rnn':
    #     plt.plot(fpr_rnn, tpr_rnn, label=cell_type)
    # elif cell_type == 'gru':
    #     plt.plot(fpr_rnn, tpr_rnn, label='attention')
    # elif cell_type == 'attention':
    #     plt.plot(fpr_rnn, tpr_rnn, label='gru')
    # plt.legend(loc='best')
      
  if show_attention:
    with torch.no_grad():
      model.eval()
      _, attn_weights = model(X_plot_batch)
      # plot_attention(attn_weights.cpu())
  #plt.show()

def train_rnn(model, verify, epochs, learning_rate, batch_size, X, Y, X_test, Y_test, show_attention,cell_type, device):
  """
  Training loop for a model utilizing hidden states.

  verify enables sanity checks of the model.
  epochs decides the number of training iterations.
  learning rate decides how much the weights are updated each iteration.
  batch_size decides how many examples are in each mini batch.
  show_attention decides if attention weights are plotted.
  """

  print_interval = 1
  seq_length = X.shape[0]
  writer = SummaryWriter(f'tensorboard/{cell_type}/{seq_length}/')
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  criterion = torch.nn.CrossEntropyLoss()
  num_of_examples = X.shape[1]
  num_of_batches = math.floor(num_of_examples/batch_size)

  best_val_loss = 9999.0
  best_training_loss = 9999.0

  # best_path = f'checkpoint/{seq_length}/{cell_type}/best_model/'
  # resume_directory = f'checkpoint/{seq_length}/{cell_type}/resume/'
  # best_path = f'checkpoint/{SUBTYPE}/{cell_type}/best_model/'
  # resume_directory = f'checkpoint/{SUBTYPE}/{cell_type}/resume/'
  best_path = f'checkpoint/2023/{cell_type}/best_model/'
  resume_directory = f'checkpoint/2023/{cell_type}/resume/'

  if not os.path.exists(best_path):
    os.makedirs(best_path)

  if not os.path.exists(resume_directory):
    os.makedirs(resume_directory)

  if verify:
    verify_model(model, X, Y, batch_size)
  all_losses = []
  all_val_losses = []
  all_accs = []
  all_pres = []
  all_recs = []
  all_fscores = []
  all_mccs = []
  all_val_accs = []
  all_val_pres = []
  all_val_recs = []
  all_val_fscores = []
  all_val_mccs = []

  # Find mini batch that contains at least one mutation to plot
  plot_batch_size = 10
  i = 0
  while not Y_test[i]:
    i += 1

  X_plot_batch = X_test[:, i:i+plot_batch_size, :]
  Y_plot_batch = Y_test[i:i+plot_batch_size]
  plot_batch_scores = []

  start_time = time.time()
  model.to(device)
  for epoch in range(epochs):
    model.train()
    running_loss = 0
    running_acc = 0
    running_pre = 0
    running_pre_total = 0
    running_rec = 0
    running_rec_total = 0
    epoch_fscore = 0
    running_mcc_numerator = 0
    running_mcc_denominator = 0
    running_rec_total = 0

    hidden = model.init_hidden(batch_size)

    for count in range(0, num_of_examples - batch_size + 1, batch_size):
      repackage_hidden(hidden)

      X_batch = X[:, count:count+batch_size, :]
      X_batch = X_batch.to(device)
      Y_batch = Y[count:count+batch_size]
      Y_batch = Y_batch.to(device)
      
      scores, _ = model(X_batch, hidden)
      loss = criterion(scores, Y_batch)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      predictions = predictions_from_output(scores)
      
      conf_matrix = validation.get_confusion_matrix(Y_batch, predictions)
      TP, FP, FN, TN = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
      running_acc += TP + TN
      running_pre += TP
      running_pre_total += TP + FP
      running_rec += TP
      running_rec_total += TP + FN
      running_mcc_numerator += (TP * TN - FP * FN)
      if ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) == 0:
          running_mcc_denominator += 0
      else:
          running_mcc_denominator += math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
      running_loss += loss.item()

    elapsed_time = time.time() - start_time
    epoch_acc = running_acc / Y.shape[0]
    all_accs.append(epoch_acc)
    
    if running_pre_total == 0:
        epoch_pre = 0
    else:
        epoch_pre = running_pre / running_pre_total
    all_pres.append(epoch_pre)
    
    if running_rec_total == 0:
        epoch_rec = 0
    else:
        epoch_rec = running_rec / running_rec_total
    all_recs.append(epoch_rec)
    
    if (epoch_pre + epoch_rec) == 0:
        epoch_fscore = 0
    else:
        epoch_fscore = 2 * epoch_pre * epoch_rec / (epoch_pre + epoch_rec)
    all_fscores.append(epoch_fscore)
    
    if running_mcc_denominator == 0:
        epoch_mcc = 0
    else:
        epoch_mcc = running_mcc_numerator / running_mcc_denominator
    all_mccs.append(epoch_mcc)
    
    epoch_loss = running_loss / num_of_batches
    all_losses.append(epoch_loss)

    writer.add_scalar('Loss/train', epoch_loss, epoch)

    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    with torch.no_grad():
      model.eval()
      test_scores, _ = model(X_test, model.init_hidden(Y_test.shape[0]))
      # print(test_scores)
      predictions = predictions_from_output(test_scores)
      # print(predictions)      
      predictions = predictions.view_as(Y_test)
      pred_prob = calculate_prob(test_scores)
      precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, predictions)
      # val_auc = roc_auc_score(Y_test, test_scores)
      
      val_loss = criterion(test_scores, Y_test).item()
      all_val_losses.append(val_loss)
      all_val_accs.append(val_acc)
      all_val_pres.append(precision)
      all_val_recs.append(recall)
      all_val_fscores.append(fscore)
      all_val_mccs.append(mcc)

      X_plot_batch = X_plot_batch.to(device)
      plot_scores, _ = model(X_plot_batch, model.init_hidden(Y_plot_batch.shape[0]))
      plot_batch_scores.append(plot_scores)

    writer.add_scalar('Loss/val', val_loss, epoch)

    if (epoch+1) % print_interval == 0:
      print('Epoch %d Time %s' % (epoch, utils.get_time_string(elapsed_time)))
      print('T_loss %.3f\tT_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f' %(epoch_loss, epoch_acc, epoch_pre, epoch_rec, epoch_fscore, epoch_mcc))
      print('V_loss %.3f\tV_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'%(val_loss, val_acc, precision, recall, fscore, mcc))

    checkpoint = {
      'epoch' : epoch + 1,
      'total_epoch' : epochs,
      'state_dict' : model.state_dict(),
      'optimizer' : optimizer.state_dict(),
      'best_training_loss' : best_training_loss,
      'best_val_loss' : best_val_loss
    }

    save_current_model(
      checkpoint=checkpoint,
      resume_directory=resume_directory
    )

    best_val_loss, best_training_loss = save_best_model(
                                          val_loss=val_loss,
                                          best_val_loss=best_val_loss,
                                          epoch_loss=epoch_loss,
                                          best_training_loss=best_training_loss,
                                          checkpoint=checkpoint,
                                          best_path=best_path
                                        )
  #plot_training_history(all_losses, all_val_losses, all_accs, all_val_accs, plot_batch_scores, Y_plot_batch)

      #roc curve

  metrics = {
    'type': cell_type,
    'train_loss' : all_losses,
    'val_loss' : all_val_losses,
    'train_accuracy' : all_accs,
    'val_accuracy' : all_val_accs,
    'train_precision' : all_pres,
    'val_precision' : all_val_pres,
    'train_recall' : all_recs,
    'val_recall' : all_val_recs,
    'train_fscore' : all_fscores,
    'val_fscore' : all_val_fscores,
    'train_mcc' : all_mccs,
    'val_mcc' : all_val_mccs,
    'epochs' : epochs
  }

  return metrics

  if epoch + 1 == 50:
    tpr_rnn, fpr_rnn, _ = roc_curve(Y_test.cpu(), pred_prob.cpu()) 
    print(auc(fpr_rnn, tpr_rnn))
    plt.figure(1)
    #plt.xlim(0, 0.8)
    plt.ylim(0.5, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    if cell_type == 'lstm':
        plt.plot(fpr_rnn, tpr_rnn, label=cell_type)
    elif cell_type == 'rnn':
        plt.plot(fpr_rnn, tpr_rnn, label=cell_type)
    elif cell_type == 'gru':
        plt.plot(fpr_rnn, tpr_rnn, label='attention')
    elif cell_type == 'attention':
        plt.plot(fpr_rnn, tpr_rnn, label='gru')
    plt.legend(loc='best')
      
  if show_attention:
    with torch.no_grad():
      model.eval()
      _, attn_weights = model(X_plot_batch, model.init_hidden(Y_plot_batch.shape[0]))
      plot_attention(attn_weights.cpu())
  #plt.show()
  

def svm_baseline(X, Y, X_test, Y_test, method=None):
    clf = SVC(gamma='auto', class_weight='balanced', probability=True).fit(X, Y) 
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))
    
    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, Y_pred)
    print('SVM baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
                % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
                % (val_acc, precision, recall, fscore, mcc))
    if(method!=None):
        with open(f'./reports/results/{method}_SVM.txt', 'a') as f:
            f.write(' T_Accuracy:\t%.3f\n' % train_acc)
            f.write(' T_Precision:\t%.3f\n' % train_pre)
            f.write(' T_Recall:\t%.3f\n' % train_rec)
            f.write(' T_F1-score:\t%.3f\n' % train_fscore)
            f.write(' T_Matthews CC:\t%.3f\n\n' % train_mcc)
            f.write(' V_Accuracy:\t%.3f\n' % val_acc)
            f.write(' V_Precision:\t%.3f\n' % precision)
            f.write(' V_Recall:\t%.3f\n' % recall)
            f.write(' V_F1-score:\t%.3f\n' % fscore)
            f.write(' V_Matthews CC:\t%.3f\n\n' % mcc)

    #roc curve  
    y_pred_roc = clf.predict_proba(X_test)[:, 1]  
    fpr_rt_svm, tpr_rt_svm, _ = roc_curve(Y_test, y_pred_roc)
    print(auc(fpr_rt_svm, tpr_rt_svm))
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_svm, tpr_rt_svm, label='SVM')
    plt.legend(loc='best')

def random_forest_baseline(X, Y, X_test, Y_test, method=None):
    clf = ensemble.RandomForestClassifier().fit(X, Y) 
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))
    
    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, Y_pred)
    print('Rrandom Forest baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
                % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
                % (val_acc, precision, recall, fscore, mcc))
    if(method!=None):
        with open(f'./reports/results/{method}_RF.txt', 'a') as f:
            f.write(' T_Accuracy:\t%.3f\n' % train_acc)
            f.write(' T_Precision:\t%.3f\n' % train_pre)
            f.write(' T_Recall:\t%.3f\n' % train_rec)
            f.write(' T_F1-score:\t%.3f\n' % train_fscore)
            f.write(' T_Matthews CC:\t%.3f\n\n' % train_mcc)
            f.write(' V_Accuracy:\t%.3f\n' % val_acc)
            f.write(' V_Precision:\t%.3f\n' % precision)
            f.write(' V_Recall:\t%.3f\n' % recall)
            f.write(' V_F1-score:\t%.3f\n' % fscore)
            f.write(' V_Matthews CC:\t%.3f\n\n' % mcc)
 
    #roc curve  
    y_pred_roc = clf.predict_proba(X_test)[:, 1]  
    fpr_rt_rf, tpr_rt_rf, _ = roc_curve(Y_test, y_pred_roc)
    plt.figure(1)
    print(auc(fpr_rt_rf, tpr_rt_rf))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_rf, tpr_rt_rf, label='RF')   
    plt.legend(loc='best')        

def knn_baseline(X, Y, X_test, Y_test, method=None):
    clf = KNeighborsClassifier().fit(X, Y) 
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))
    
    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, Y_pred)
    print('knn baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
                % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
                % (val_acc, precision, recall, fscore, mcc))
    if(method!=None):
        with open(f'./reports/results/{method}_SVM.txt', 'a') as f:
            f.write(' T_Accuracy:\t%.3f\n' % train_acc)
            f.write(' T_Precision:\t%.3f\n' % train_pre)
            f.write(' T_Recall:\t%.3f\n' % train_rec)
            f.write(' T_F1-score:\t%.3f\n' % train_fscore)
            f.write(' T_Matthews CC:\t%.3f\n\n' % train_mcc)
            f.write(' V_Accuracy:\t%.3f\n' % val_acc)
            f.write(' V_Precision:\t%.3f\n' % precision)
            f.write(' V_Recall:\t%.3f\n' % recall)
            f.write(' V_F1-score:\t%.3f\n' % fscore)
            f.write(' V_Matthews CC:\t%.3f\n\n' % mcc)

    #roc curve  
    y_pred_roc = clf.predict_proba(X_test)[:, 1]  
    fpr_rt_knn, tpr_rt_knn, _ = roc_curve(Y_test, y_pred_roc)
    print(auc(fpr_rt_knn, tpr_rt_knn))
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_knn, tpr_rt_knn, label='LR')
    plt.legend(loc='best')
    # plt.show()

def bayes_baseline(X, Y, X_test, Y_test, method=None):
    clf = GaussianNB().fit(X, Y) 
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))
    
    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, Y_pred)
    print('bayes baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
                % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
                % (val_acc, precision, recall, fscore, mcc))

    #roc curve  
    y_pred_roc = clf.predict_proba(X_test)[:, 1]  
    fpr_rt_nb, tpr_rt_nb, _ = roc_curve(Y_test, y_pred_roc)
    print(auc(fpr_rt_nb, tpr_rt_nb))
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_nb, tpr_rt_nb, label='LR')
    plt.legend(loc='best')
    # plt.show()



# def xgboost_baseline(X, Y, X_test, Y_test, method=None):
#     clf = XGBClassifier().fit(X, Y) 
#     train_acc = accuracy_score(Y, clf.predict(X))
#     train_pre = precision_score(Y, clf.predict(X))
#     train_rec = recall_score(Y, clf.predict(X))
#     train_fscore = f1_score(Y, clf.predict(X))
#     train_mcc = matthews_corrcoef(Y, clf.predict(X))
    
#     Y_pred = clf.predict(X_test)
#     precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, Y_pred)
#     print('Logistic regression baseline:')
#     print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
#                 % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
#     print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
#                 % (val_acc, precision, recall, fscore, mcc))
#     #roc curve  
#     y_pred_roc = clf.predict_proba(X_test)[:, 1]  
#     fpr_rt_xgb, tpr_rt_xgb, _ = roc_curve(Y_test, y_pred_roc)
#     print(auc(fpr_rt_xgb, tpr_rt_xgb))
#     plt.figure(1)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.plot(fpr_rt_xgb, tpr_rt_xgb, label='LR')
#     plt.legend(loc='best')
#     #plt.show()

     
def logistic_regression_baseline(X, Y, X_test, Y_test, method=None):
    clf = LogisticRegression(random_state=0).fit(X, Y) 
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))
    
    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, Y_pred)
    print('Logistic regression baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
                % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
                % (val_acc, precision, recall, fscore, mcc))
    #roc curve  
    y_pred_roc = clf.predict_proba(X_test)[:, 1]  
    fpr_rt_lr, tpr_rt_lr, _ = roc_curve(Y_test, y_pred_roc)
    print(auc(fpr_rt_lr, tpr_rt_lr))
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_lr, tpr_rt_lr, label='SVM')
    plt.legend(loc='best')
    # plt.show()