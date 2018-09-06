import re


def chn2char(text, dcits):
  pass


def char2chn(text, dicts):
  pass


def gen_dict(text):
  text_dict = {}
  for word in text:
    try:
      text_dict[word] = text_dict[word] + 1
    except Exception:
      text_dict[word] = 1
  print(len(text_dict.keys()))


if __name__ == '__main__':
  with open('texts/original2.txt', 'r', encoding='utf-8') as f:
    text = f.read()
  # s = re.split(r'第\w+回\s+', text)
  # s.pop(0)
  gen_dict(text)
