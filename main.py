from rnn import CharRNN
import numpy as np
from solver import CharSolver
import pickle
from data import Data


with open('dicts/char2ix.pk', 'rb') as f:
    char2ix = pickle.load(f)
data = Data(1)
small_lstm_model = CharRNN(
  cell_type='lstm',
  word_to_idx=char2ix,
  hidden_dim=128,
  wordvec_dim=128,
  dtype=np.float32,
)

small_lstm_solver = CharSolver(
  small_lstm_model, data,
  update_rule='adam',
  num_epochs=50,
  batch_size=100,
  optim_config={
    'learning_rate': 5e-3,
  },
  lr_decay=0.995,
  verbose=True, print_every=10,
)

small_lstm_solver.train()

# Plot the training losses
plt.plot(small_lstm_solver.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.show()