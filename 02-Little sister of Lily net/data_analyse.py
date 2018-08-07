# -*-coding:utf-8-*-
import pandas as pd
from pyecharts import Map
from pyecharts import Bar, Line, Overlap, Scatter
from os import path
from wordcloud import WordCloud, ImageColorGenerator
from scipy.misc import imread
import matplotlib.pyplot as plt


def city_dist(info):
	grouped = info.groupby(['cityChn'])
	grouped_city = grouped['cityChn']
	city_com = grouped_city.agg(['count'])
	city_com.reset_index(inplace=True)
	city_distribute_data = [(city_com['cityChn'][i], city_com['count'][i])
	                        for i in range(0, city_com.shape[0])]

	map = Map('北京征婚小姐姐分布', width=1200, height=600)
	attr, value = map.cast(city_distribute_data)
	attr = [loc[2:]+'区' for loc in attr]  # 命名规则与pyecharts中名字一致

	map.add("密度", attr, value, maptype='北京', is_map_symbol_show=True, is_visualmap=True, visual_text_color='#000', visual_range=[0, 2000])
	map.show_config()
	map.render('北京征婚小姐姐分布.html')


# 索引输出
def index_out(tidy, needed_index):
	return tidy[:, needed_index].index.tolist()


# 数值输出
def data_out(tidy, needed_index):
	return tidy[:, needed_index].tolist()


# group
def group(info, by):  # by为类表.eg by = ['a', 'b']
	grouped = info.groupby(by=by)
	return grouped


# 地区-学历-人数关系
def loc_edu_num(info):
	grouped = group(info, ['cityChn', 'educationChn'])
	tidy = grouped.size()
	doctor_loc = index_out(tidy, '博士')
	master_loc = index_out(tidy, '硕士')
	bachelor_loc = index_out(tidy, '本科')
	specialty_loc = index_out(tidy, '大专')

	doctor_num = data_out(tidy, '博士')
	master_num = data_out(tidy, '硕士')
	bachelor_num = data_out(tidy, '本科')
	specialty_num = data_out(tidy, '大专')

	bar = Bar("小姐姐-城区与学历分布")

	bar.add("博士", doctor_loc, doctor_num, xaxis_rotate=30, mark_point=['max'])
	bar.add("硕士", master_loc, master_num, xaxis_rotate=30, mark_point=['max'])
	bar.add("大专", specialty_loc, specialty_num, xaxis_rotate=30, mark_point=['max'])
	bar.add("本科", bachelor_loc, bachelor_num, xaxis_rotate=30, mark_point=['max'], yaxis_name='人数')
	bar.show_config()
	bar.render("北京小姐姐城区与学历分布.html")


# 年龄-学历-人数关系
def age_edu_num(info):
	grouped = group(info, ['age', 'educationChn'])
	tidy = grouped.size()
	doctor_age = index_out(tidy, '博士')
	master_age = index_out(tidy, '硕士')
	bachelor_age = index_out(tidy, '本科')
	specialty_age = index_out(tidy, '大专')

	doctor_num = data_out(tidy, '博士')
	master_num = data_out(tidy, '硕士')
	bachelor_num = data_out(tidy, '本科')
	specialty_num = data_out(tidy, '大专')

	bar = Bar("小姐姐-年龄与学历分布")

	bar.add("博士", doctor_age, doctor_num, is_stack=True)
	bar.add("硕士", master_age, master_num, is_stack=True)
	bar.add("大专", specialty_age, specialty_num, is_stack=True)
	bar.add("本科", bachelor_age, bachelor_num, is_stack=True, xaxis_name='年龄', yaxis_name='人数')
	bar.show_config()
	bar.render("北京百合小姐姐年龄与学历分布.html")


# 年龄-人数关系
def age_num(info):
	grouped = group(info, ['age'])
	tidy = grouped.size()
	attr = tidy.index.tolist()
	value = tidy.tolist()

	bar = Bar("小姐姐-年龄分布")
	bar.add("单位：人", attr, value, mark_point=['max'], mark_line=['average'], xaxis_name='年龄', yaxis_name='人数')
	bar.render("北京百合小姐姐年龄分布.html")


# 薪酬-地区关系
def pay_loc(info):
	grouped = group(info, ['cityChn', 'incomeChn'])
	tidy = grouped.size()
	income_l1 = index_out(tidy, '20000-50000')
	income_l2 = index_out(tidy, '10000-20000')
	income_l3 = index_out(tidy, '5000-10000')
	income_l4 = index_out(tidy, '2000-5000')
	income_l5 = index_out(tidy, '2000以下')

	income_l1_loc = data_out(tidy, '20000-50000')
	income_l2_loc = data_out(tidy, '10000-20000')
	income_l3_loc = data_out(tidy, '5000-10000')
	income_l4_loc = data_out(tidy, '2000-5000')
	income_l5_loc = data_out(tidy, '2000以下')
	bar = Bar("小姐姐-收入与城区分布")

	bar.add("2w-5w", income_l1, income_l1_loc, xaxis_rotate=30, mark_point=['max'])
	bar.add("1w-2w", income_l2, income_l2_loc, xaxis_rotate=30, mark_point=['max'])
	bar.add("<2k", income_l5, income_l5_loc, xaxis_rotate=30, mark_point=['max'])
	bar.add("2k-5k", income_l4, income_l4_loc, xaxis_rotate=30, mark_point=['max'])
	bar.add("5k-10k", income_l3, income_l3_loc, xaxis_rotate=30, mark_point=['max'], yaxis_name='人数')
	bar.show_config()
	bar.render("北京百合小姐姐收入与城区分布.html")
	# scatter = Scatter("小姐姐-收入与城区分布")
	# print(income_l4_loc)
	# scatter.add("2w-5w", income_l1, income_l1_loc)
	# scatter.add("1w-2w", income_l2, income_l2_loc)
	# scatter.add("<2k", income_l5, income_l5_loc)
	# scatter.add("2k-5k", income_l4, income_l4_loc)
	# scatter.add("5k-10k", income_l3, income_l3_loc)
	# scatter.show_config()
	# scatter.render("北京百合小姐姐收入与城区分布.html")

# 地区-人数关系
def loc_and_mean_age(info):
	grouped = group(info, ['cityChn'])
	tidy = grouped['age']
	tidy_com = tidy.agg(['mean', 'count'])
	tidy_com.reset_index(inplace=True)
	tidy_com['mean'] = round(tidy_com['mean'], 2)

	attr = tidy_com['cityChn']
	num = tidy_com['count']
	aver = tidy_com['mean']

	line = Line("小姐姐-平均年龄")
	line.add("年龄", attr, aver, is_stack=True, xaxis_rotate=30, mark_point=['max', 'min'],
	         yaxis_min=26, is_splitline_show=False)

	bar = Bar("小姐姐-城区分布")
	bar.add("单位：人", attr, num, mark_point=['max'], mark_line=['average'], xaxis_rotate=30, yaxis_min=0)

	overlap = Overlap()
	overlap.add(bar)
	overlap.add(line, yaxis_index=1, is_add_yaxis=True)
	overlap.render("北京百合小姐姐平均年龄与分布.html")


def word_cloud(info):
	d = path.dirname(__file__)
	alice_coloring = imread(path.join(d, 'alice_color.png'))
	image_colors = ImageColorGenerator(alice_coloring)

	desc_list = info['familyDescription']
	text = []
	for item in desc_list:
		text.append(str(item))
	desc_list = " ".join(text)  # 部分内容为浮点，全部转换为str

	wc = WordCloud(background_color='white', font_path='./simsun.ttf', mask=alice_coloring, max_words=2000,
	               max_font_size=40, random_state=42).generate(desc_list)

	plt.figure()
	plt.imshow(wc.recolor(color_func=image_colors))
	plt.axis('off')
	plt.show()


if __name__ == "__main__":
	data = pd.read_excel('百合网北京范围征婚信息.xlsx')
	# loc_edu_num(data)
	# age_num(data)
	# loc_and_mean_age(data)
	pay_loc(data)
	# age_edu_num(data)
	# city_dist(data)
	# word_cloud(data)
