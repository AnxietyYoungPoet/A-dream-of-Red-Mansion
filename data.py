import numpy as np


class Data(object):
  def __init__(self, chapter):
    self.data = np.load(f'texts/num_ver/chapter{chapter}.npy')[:5000]
    self.reset_pointer()
    self.endlabel = self.data[-1]
  
  def reset_pointer(self):
    self._pointer = -1

  def split_data(self, batch_size):
    self.data = np.split(
      self.data[:-1], np.arange(batch_size, len(self.data), batch_size).tolist())
    self.num_bath = len(self.data)

  def next_batch(self):
    if self._pointer >= self.num_bath - 1:
      self.reset_pointer()
      # print('reset')
      # print(self.data[self._pointer + 1])
    self._pointer += 1
    try:
      extra_label = self.data[self._pointer + 1][0]
    except Exception:
      extra_label = self.endlabel
    return self.data[self._pointer], extra_label, self._pointer == 0


if __name__ == '__main__':
  data = Data(1)
  print(data.data[:25])
  data.split_data(25)
  print(data.next_batch())
