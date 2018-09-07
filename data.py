import numpy as np


class Data(object):
  def __init__(self):
    self.data = []
    for i in range(1, 81):
      self.data.append(np.load(f'texts/num_ver/chapter{i}.npy'))
    self.reset_batch_pointer()
    self.reset_epoch_pointer()
    self.endlabels = []
    for i in range(80):
      self.endlabels.append(self.data[i][-1])

  def shuffle_data(self):
    np.random.shuffle(self.data)

  def reset_batch_pointer(self):
    self.batch_pointer = -1

  def reset_epoch_pointer(self):
    self.epoch_pointer = 0
    print('new epoch')

  def split_data(self, batch_size):
    self.num_batch = []
    for i in range(80):
      length = len(data[i][:-1])
      self.data[i] = np.split(
        self.data[i][:-1], np.arange(batch_size, length, batch_size).tolist())
      self.num_bath.append(len(Data[i]))

  def next_batch(self):
    if self.batch_pointer >= self.num_bath[self.epoch_pointer] - 1:
      self.reset_batch_pointer()
      self.epoch_pointer += 1
      if self.epoch_pointer >= len(self.data):
        self.reset_epoch_pointer()
    self.batch_pointer += 1
    try:
      extra_label = self.data[self.epoch_pointer][self._pointer + 1][0]
    except Exception:
      extra_label = self.endlabel[self.epoch_pointer]
    return self.data[self.epoch_pointer][self.batch_pointer], extra_label, self.batch_pointer == 0


if __name__ == '__main__':
  data = Data()
  print(len(data.data[0]))
  data.shuffle_data()
  print(len(data.data[0]))
  # data.split_data(25)
  # print(data.next_batch())
