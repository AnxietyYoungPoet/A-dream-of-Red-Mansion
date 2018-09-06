import requests
import re
root = 'https://www.shanbay.com'
text = requests.get(root + '/wordbook/139771/')
urls = re.findall(r'/wordlist/139771/\d+/', text.text)
with open('vocabs.txt', 'w') as f:
  for url in urls:
    for index in range(1, 11):
      text = requests.get(root + url + f'?page={index}').text
      words = re.findall(r'<strong>\w+</strong>', text)
      for word in words:
        f.write(word[8:-9] + '\n')
