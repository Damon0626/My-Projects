# -*-coding:utf-8-*-
import pandas as pd
import random
import time
import requests
import json


tomato = pd.DataFrame(columns=['date', 'score', 'city', 'comment', 'nick'])

for i in range(0, 1000):
	j = random.randint(1, 1000)
	print(str(i)+' '+str(j))
	try:
		time.sleep(2)
		url = 'http://m.maoyan.com/mmdb/comments/movie/1212592.json?_v_=yes&offest=' + str(j)
		html = requests.get(url=url).content
		data = json.loads(html.decode('utf-8'))['cmts']
		for item in data:
			tomato = tomato.append({'date': item['time'].split(' ')[0],
			                        'city': item['cityName'],
			                        'score': item['score'],
			                        'comment': item['content'],
			                        'nick': item['nick']}, ignore_index=True)
		tomato.to_csv('西红市首富.csv', index=False)
	except:
		continue