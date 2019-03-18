import math
import matplotlib as mpl
import warnings
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

n_features = 2
x, y = make_blobs(n_samples=300, n_features=n_features, centers=6)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)

clf1 = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
clf2 = RandomForestClassifier(n_estimators=10, max_features=math.sqrt(n_features), max_depth=None, min_samples_split=2,
                              bootstrap=True)
clf3 = ExtraTreesClassifier(n_estimators=10, max_features=math.sqrt(n_features), max_depth=None, min_samples_split=2,
                            bootstrap=False)

'''
cross validation
'''
clf1.fit(x_train, y_train)
clf2.fit(x_train, y_train)
clf3.fit(x_train, y_train)

'''
prediction
'''
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
area_smaple_point = np.stack((x1.flat, x2.flat), axis=1)
area1_predict = clf1.predict(area_smaple_point)
area1_predict = area1_predict.reshape(
    x1.shape)

area2_predict = clf2.predict(area_smaple_point)
area2_predict = area2_predict.reshape(x1.shape)

area3_predict = clf3.predict(area_smaple_point)
area3_predict = area3_predict.reshape(x1.shape)

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

classifier_area_color = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

#first
plt.subplot(2, 2, 1)

plt.pcolormesh(x1, x2, area1_predict, cmap=classifier_area_color)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker='o', s=50, cmap=cm_dark)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='x', s=50, cmap=cm_dark)

plt.xlabel('data_x', fontsize=8)
plt.ylabel('data_y', fontsize=8)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'DecisionTreeClassifier:', fontsize=8)
plt.text(x1_max - 9, x2_max - 2, u'$o---train ; x---test$')

#second
plt.subplot(2, 2, 2)

plt.pcolormesh(x1, x2, area2_predict, cmap=classifier_area_color)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker='o', s=50, cmap=cm_dark)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='x', s=50, cmap=cm_dark)

plt.xlabel('data_x', fontsize=8)
plt.ylabel('data_y', fontsize=8)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'RandomForestClassifier:', fontsize=8)
plt.text(x1_max - 9, x2_max - 2, u'$o---train ; x---test$')

#third
plt.subplot(2, 2, 3)

plt.pcolormesh(x1, x2, area3_predict, cmap=classifier_area_color)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker='o', s=50, cmap=cm_dark)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='x', s=50, cmap=cm_dark)

plt.xlabel('data_x', fontsize=8)
plt.ylabel('data_y', fontsize=8)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'ExtraTreesClassifier:', fontsize=8)
plt.text(x1_max - 9, x2_max - 2, u'$o---train ; x---test$')

#forth
plt.subplot(2, 2, 4)
y = []
scores1 = cross_val_score(clf1, x_train, y_train)
y.append(scores1.mean())
scores2 = cross_val_score(clf2, x_train, y_train)
y.append(scores2.mean())
scores3 = cross_val_score(clf3, x_train, y_train)
y.append(scores3.mean())

x = [0, 1, 2]
plt.bar(x, y, 0.4, color="green")
plt.xlabel("0--DecisionTreeClassifier;1--RandomForestClassifier;2--ExtraTreesClassifie", fontsize=8)
plt.ylabel("average accuracy", fontsize=8)
plt.ylim(0.9, 0.99)
plt.title("cross validation", fontsize=8)
for a, b in zip(x, y):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

plt.show()