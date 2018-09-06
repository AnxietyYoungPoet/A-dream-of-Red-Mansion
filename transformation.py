import re
import pickle


def chn2eng(text, chn2eng_dict):
  encoded_text = ''
  for word in text:
    encoded_text = encoded_text + chn2eng_dict[word] + ' '
  return encoded_text


def eng2chn(text, eng2chn_dict):
  decoded_text = ''
  split_text = text.split()
  for word in split_text:
    decoded_text = decoded_text + eng2chn_dict[word]
  return decoded_text


def encoding():
  with open('dicts/chn2eng', 'rb') as f:
    chn2eng_dict = pickle.load(f)
  for i in range(1, 121):
    with open(f'texts/chapter{i}.txt', 'r', encoding='utf-8') as f:
      text = f.read()
    encoded_text = chn2eng(text, chn2eng_dict)
    with open(f'texts/eng_ver/chapter{i}.txt', 'w') as f:
      f.write(encoded_text)


def decoding():
  with open('dicts/eng2chn', 'rb') as f:
    eng2chn_dict = pickle.load(f)
  for i in range(1, 121):
    with open(f'texts/eng_ver/chapter{i}.txt', 'r') as f:
      text = f.read()
    decoded_text = eng2chn(text, eng2chn_dict)
    with open(f'texts/chn_ver/chapter{i}.txt', 'w', encoding='utf-8') as f:
      f.write(decoded_text)


def cal_weights(text):
  text_dict = {}
  for word in text:
    try:
      text_dict[word] = text_dict[word] + 1
    except Exception:
      text_dict[word] = 1
  return text_dict


def gen_dict():
  with open('texts/original2.txt', 'r', encoding='utf-8') as f:
    text = f.read()
  with open('spider/vocabs.txt', 'r') as f:
    vocabs = f.readlines()
  weights_dict = cal_weights(text)
  words = sorted(weights_dict.items(), key=lambda d: d[1], reverse=True)
  chn2eng_dict = {}
  eng2chn_dict = {}
  for chn, eng in zip(words, vocabs):
    chn2eng_dict[chn[0]] = eng.strip()
    eng2chn_dict[eng.strip()] = chn[0]
  with open('dicts/chn2eng', 'wb') as f:
    pickle.dump(chn2eng_dict, f)
  with open('dicts/eng2chn', 'wb') as f:
    pickle.dump(eng2chn_dict, f)


def split2chaps():
  with open('texts/original2.txt', 'r', encoding='utf-8') as f:
    text = f.read()
  s = re.split(r'第\w+回\s+', text)
  for _ in range(94):
    s.pop(0)
  for index, chap in enumerate(s):
    title = f'texts/chapter{index + 1}.txt'
    with open(title, 'w', encoding='utf-8') as f:
      f.write(chap)


if __name__ == '__main__':
  decoding()
