from rnn import CharRNN
import numpy as np
from solver import CharSolver
import pickle
from data import Data
import matplotlib.pyplot as plt
from transformation import *


def train(solver):
  solver.model.load_model(24000)
  solver.train()
  tensor = np.load('texts/num_ver/chapter2.npy')[1]
  h0 = solver.model.prev_h
  c0 = solver.model.prev_c
  solver.model.sample(h0, tensor, c0)
  plt.plot(small_lstm_solver.loss_history)
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.title('Training loss history')
  plt.show()


def sample(model, epoch='', state=False):
  with open('dicts/ix2char.pk', 'rb') as f:
    ix2char = pickle.load(f)
  with open('dicts/eng2chn', 'rb') as f:
    eng2chn_dict = pickle.load(f)
  if state:
    x0 = np.load('texts/num_ver/chapter2.npy')[1]
    (h0, c0) = np.load('model/state/ch.npy')
  else:
    x0, h0, c0 = None, None, None
  model.load_model(epoch)
  tensor = model.sample(h0, x0, c0)
  text = num2eng(tensor, ix2char)
  text = eng2chn(text, eng2chn_dict)
  print(text)


with open('dicts/char2ix.pk', 'rb') as f:
    char2ix = pickle.load(f)
data = Data()
small_lstm_model = CharRNN(
  cell_type='lstm',
  word_to_idx=char2ix,
  hidden_dim=256,
  wordvec_dim=128,
  dtype=np.float32,
)

small_lstm_solver = CharSolver(
  small_lstm_model, data,
  update_rule='adam',
  num_epochs=1,
  batch_size=50,
  optim_config={
    'learning_rate': 1e-2,
  },
  lr_decay=0.995,
  verbose=True, print_every=10,
)

# train(small_lstm_solver)
sample(small_lstm_solver.model, state=True)
