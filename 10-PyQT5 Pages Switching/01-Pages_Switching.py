# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 18-12-23 下午8:49
# @Email : wwymsn@163.com
# @Software: PyCharm


import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# class MainWindow(QMainWindow):
# 	def __init__(self, *args, **kwargs):
# 		super().__init__(*args, **kwargs)
# 		self.setWindowTitle('主界面')
# 		self.showMaximized()

class logindialog(QDialog):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.setWindowTitle('登录界面')
		self.resize(200, 200)
		self.setFixedSize(self.width(), self.height())
		self.setWindowFlags(Qt.WindowCloseButtonHint)

		self.frame = QFrame(self)
		self.verticalLayout = QVBoxLayout(self.frame)
		self.lineEdit_account = QLineEdit()
		self.lineEdit_account.setPlaceholderText("请输入账号")
		self.verticalLayout.addWidget(self.lineEdit_account)

		self.lineEdit_password = QLineEdit()
		self.lineEdit_password.setPlaceholderText("请输入密码")
		self.verticalLayout.addWidget(self.lineEdit_password)

		self.pushButton_enter = QPushButton()
		self.pushButton_enter.setText("进入下一个界面")
		self.verticalLayout.addWidget(self.pushButton_enter)

		self.frame1 = QFrame(self)
		self.verticalLayout = QVBoxLayout(self.frame1)
		self.pushButton_quit = QPushButton()
		self.pushButton_quit.setText("回到主页面")
		self.verticalLayout.addWidget(self.pushButton_quit)
		self.frame1.setVisible(False)
		self.pushButton_enter.clicked.connect(self.on_pushButton_enter_clicked)
		self.pushButton_quit.clicked.connect(self.on_pushButton_enter_clicked_1)

	def on_pushButton_enter_clicked(self):
		self.frame1.setVisible(True)
		self.frame.setVisible(False)

	def on_pushButton_enter_clicked_1(self):
		self.frame1.setVisible(False)
		self.frame.setVisible(True)


if __name__ == "__main__":
	app = QApplication(sys.argv)
	dialog = logindialog()
	if dialog.exec_() == QDialog.Accepted:
		# the_window = MainWindow()
		# the_window.show()
		sys.exit(app.exec_())
