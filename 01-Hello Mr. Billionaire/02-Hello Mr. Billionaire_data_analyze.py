# -*-coding:utf-8-*-
import pandas as pd
from pyecharts import Geo
from pyecharts import Line, Bar
from pyecharts import Overlap


tomato_com = pd.read_excel('西红市首富.xlsx')
grouped = tomato_com.groupby(['city'])
grouped_pct = grouped['score']
city_com = grouped_pct.agg(['mean', 'count'])  # 计算平均分和计数
city_com.reset_index(inplace=True)
city_com['mean'] = round(city_com['mean'], 2)  # 保留2位小数
data = [(city_com['city'][i], city_com['count'][i]) for i in range(0, city_com.shape[0])]


# 评论分布的热力图
geo = Geo('《西红市首富》全国热力图', title_color="#fff", title_pos="center",
          width=1200, height=600, background_color="#404a59")
attr, value = geo.cast(data)
geo.add("", attr, value, type="heatmap", maptype='china', visual_range=[0, 120], visual_text_color="#fff",
        symbol_size=10, is_visualmap=True, is_roam=False)
geo.render('西红市首富全国热力图.html')

# 评论数和分数的折线图
city_main = city_com.sort_values('count', ascending=False)[0:20]
attr = city_main['city']
v1 = city_main['count']
v2 = city_main['mean']
line = Line("主要城市评分")
line.add('城市', attr, v2, is_stack=True, xaxis_rotate=30, yaxis_min=4.2,
         mark_point=['min', 'max'], xaxis_interval=0, line_color='lightblue',
         line_width=4, mark_point_textcolor='black', mark_point_color='lightblue', is_splitline_show=False)
bar = Bar("主要城市评论数")
bar.add("城市", attr, v1, is_stack=True, xaxis_rotate=30, yaxis_min=0,
        xaxis_interval=0, is_splitline_show=False)

overlap = Overlap()
overlap.add(bar)
overlap.add(line, yaxis_index=1, is_add_yaxis=True)
overlap.render("主要城市评论数_平均分.html")