import re
from huffman import huffman_coding


def chn2char(text, dcits):
  with open('texts/original2.txt', 'r', encoding='utf-8') as f:
    text = f.read()
  s = re.split(r'第\w+回\s+', text)
  for _ in range(94):
    s.pop(0)


def char2chn(text, dicts):
  pass


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
  weights_dict = cal_weights(text)
  sorted_s = sorted(weights_dict.items(), key=lambda d: d[1], reverse=True)
  print(sorted_s[:10])


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
  gen_dict()
