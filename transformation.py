import re
import pickle
import numpy as np


def chn2eng(text, chn2eng_dict):
  encoded_text = ''
  for word in text:
    encoded_text = encoded_text + chn2eng_dict[word] + ' '
  return encoded_text


def eng2chn(text, eng2chn_dict):
  decoded_text = ''
  split_text = text.split()
  for word in split_text:
    try:
      char = eng2chn_dict[word]
    except Exception:
      char = ''
    decoded_text = decoded_text + char
  return decoded_text


def eng2num(text, char2ix):
  tensor = np.array(list(map(char2ix.get, text)))
  return tensor


def num2eng(tensor, ix2char):
  text = ''
  for num in tensor:
    text = text + ix2char[num]
  return text


def num_encoding():
  with open('dicts/char2ix.pk', 'rb') as f:
    char2ix = pickle.load(f)
  for i in range(1, 111):
    with open(f'texts/eng_ver/chapter{i}.txt', 'r') as f:
      text = f.read()
    tensor = eng2num('0' + text + '1', char2ix)
    np.save(f'texts/num_ver/chapter{i}.npy', tensor)


def num_decoding():
  with open('dicts/ix2char.pk', 'rb') as f:
    ix2char = pickle.load(f)
  for i in range(1, 111):
    tensor = np.load(f'texts/num_ver/chapter{i}.npy')
    text = num2eng(tensor[1:-1], ix2char)
    with open(f'texts/temp_eng/chapter{i}.txt', 'w') as f:
      f.write(text)


def encoding():
  with open('dicts/chn2eng', 'rb') as f:
    chn2eng_dict = pickle.load(f)
  for i in range(1, 111):
    with open(f'texts/chapter{i}.txt', 'r', encoding='utf-8') as f:
      text = f.read()
    encoded_text = chn2eng(text, chn2eng_dict)
    with open(f'texts/eng_ver/chapter{i}.txt', 'w') as f:
      f.write(encoded_text)


def decoding():
  with open('dicts/eng2chn', 'rb') as f:
    eng2chn_dict = pickle.load(f)
  for i in range(1, 111):
    with open(f'texts/eng_ver/chapter{i}.txt', 'r') as f:
      text = f.read()
    decoded_text = eng2chn(text, eng2chn_dict)
    with open(f'texts/chn_ver/chapter{i}.txt', 'w', encoding='utf-8') as f:
      f.write(decoded_text)


def gen_vocab2num_dict():
  with open('texts/eng_ver/chapter1.txt', 'r') as f:
    text = f.read()
  text = '0' + text + '1'
  chars = set(text)
  char2ix = {ch: i for i, ch in enumerate(chars)}
  ix2char = {i: ch for i, ch in enumerate(chars)}
  with open('dicts/char2ix.pk', 'wb') as f:
    pickle.dump(char2ix, f)
  with open('dicts/ix2char.pk', 'wb') as f:
    pickle.dump(ix2char, f)


def cal_weights(text):
  text_dict = {}
  for word in text:
    try:
      text_dict[word] = text_dict[word] + 1
    except Exception:
      text_dict[word] = 1
  return text_dict


def gen_dict():
  with open('texts/original.txt', 'r', encoding='utf-8') as f:
    text = f.read()
  with open('spider/vocabs.txt', 'r') as f:
    vocabs = f.readlines()
  weights_dict = cal_weights(text)
  words = sorted(weights_dict.items(), key=lambda d: d[1], reverse=True)
  chn2eng_dict = {}
  eng2chn_dict = {}
  for chn, eng in zip(words, vocabs):
    chn2eng_dict[chn[0]] = eng.strip().lower()
    eng2chn_dict[eng.strip().lower()] = chn[0]
  with open('dicts/chn2eng', 'wb') as f:
    pickle.dump(chn2eng_dict, f)
  with open('dicts/eng2chn', 'wb') as f:
    pickle.dump(eng2chn_dict, f)


def split2chaps():
  with open('texts/original.txt', 'r', encoding='utf-8') as f:
    text = f.read()
  s = re.split(r'第\w+回\s+', text)
  s.pop(0)
  print(len(s))
  for index, chap in enumerate(s):
    title = f'texts/chapter{index + 1}.txt'
    with open(title, 'w', encoding='utf-8') as f:
      f.write(chap)


if __name__ == '__main__':
  # split2chaps()
  # gen_dict()
  # encoding()
  # decoding()
  num_encoding()
  num_decoding()
