# -*-coding:utf-8-*-
import json
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from os import path
from scipy.misc import imread
import pandas as pd
import time
import random

datacsv = pd.DataFrame(columns=['nickname', 'gender', 'age', 'cityChn', 'educationChn',
                                'incomeChn', 'marriageChn', 'familyDescription'])
d = path.dirname(__file__)
lover = imread(path.join(d, 'alice_color.png'))

headers = {
	"Cookie": "******", # yourself
	"Host": "search.baihe.com",
	"Origin": "http://search.baihe.com",
	"Referer": "http://search.baihe.com/"
	}

for i in range(2000):
	print(i)
	url = "http://search.baihe.com/search/noLogin?&jsonCallBack=jQuery183017813270680585913_1533221775"+str(random.randint(100, 999))
	time.sleep(2)
	response = requests.post(url, headers=headers)
	data = response.text
	data = data[42:len(data)-2]
	info = json.loads(data)
	num_of_person = len(info['data'])
	for data in info['data']:
		item = {
			'nickname': data['nickname'],
			'gender': data['gender'],
			'age': data['age'],
			'cityChn': data['cityChn'],
			'educationChn': data['educationChn'],
			'incomeChn': data['incomeChn'],
			'marriageChn': data['marriageChn'],
			'familyDescription': data['familyDescription']
		}
		datacsv = datacsv.append(item, ignore_index=True)
	datacsv.to_csv('data.csv', index=False)
