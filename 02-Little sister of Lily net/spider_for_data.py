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
	"Cookie": "accessID=20180802221815489960; cookie_pcc=701%7C%7Cwww.baidu.com%7C%7C%7C%7Chttps%3A//www.baidu.com/link/url%3DStgFObDWACLrQlCvfhBxihGNHYC9T3lrTc-ISdoP57O%26ck%3D1978.1.62.318.155.314.141.4333%26shh%3Dwww.baidu.com%26sht%3Dbaiduhome_pg%26wd%3D%26eqid%3Dd152bbb300003625000000045b63129d; tempID=7430290578; NTKF_T2D_CLIENTID=guest7D336862-8DE5-84B9-E134-FB00E55EDF84; orderSource=10130301; lastLoginDate=Thu%20Aug%2002%202018%2023%3A09%3A44%20GMT+0800%20%28%u4E2D%u56FD%u6807%u51C6%u65F6%u95F4%29; accessToken=BH1533222585448369776; Hm_lvt_5caa30e0c191a1c525d4a6487bf45a9d=1533221743,1533221777,1533222109,1533222586; nTalk_CACHE_DATA={uid:kf_9847_ISME9754_guest7D336862-8DE5-84,tid:1533222586073023}; _fmdata=BJasv9VNXfN6fjLGGLSoX8jchsyL9uW%2BOVzKd28YiwXUjSlX6OwqP8zGLoxbfxOKIoTcKsTsFZhQ2U%2FCubVhRAWFOfcqF9xXEIIvd1J0sJU%3D; Hm_lpvt_5caa30e0c191a1c525d4a6487bf45a9d=1533222766",
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
