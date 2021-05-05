# 信用卡客户用户画像 及 贷款违约预测模型
##### 一、思维导图

![avatar](https://cdn.kesci.com/upload/image/qbe38xnfsu.png)

##### 二、表结构

![avatar](https://cdn.kesci.com/upload/image/qbe1nq4507.png)

##### 三、数据可视化相关报表

![avatar](https://cdn.kesci.com/upload/image/qbe1w3ptz8.png?imageView2/0/w/960/h/960)

![avatar](https://cdn.kesci.com/upload/image/qbe1witbz4.png?imageView2/0/w/960/h/960)

![avatar](https://cdn.kesci.com/upload/image/qbe1xak0lw.png?imageView2/0/w/960/h/960)

![avatar](https://cdn.kesci.com/upload/image/qbe21qfamz.png?imageView2/0/w/960/h/960)

![avatar](https://cdn.kesci.com/upload/image/qbe22vazt1.png?imageView2/0/w/960/h/960)

![avatar](https://cdn.kesci.com/upload/image/qbe24sry08.png?imageView2/0/w/960/h/960)

![avatar](https://cdn.kesci.com/upload/image/qbe28pnfwu.png?imageView2/0/w/960/h/960)

![avatar](https://cdn.kesci.com/upload/image/qbe28pnfwu.png?imageView2/0/w/960/h/960)


【信用卡客户画像总结分析】：
（一）总体趋势（近六年）：
    1.逐年发卡量：金卡、普通卡均呈逐年上升趋势，青年卡在1997年的发行量同比降低，但总体为上升趋势；
                逐年发行量占比排名为：普通卡 > 青年卡 > 金卡
    2.总发卡量：总体发卡量占比排名为：普通卡 > 青年卡 > 金卡,其中普通卡占比接近总发卡量的3/4
（二）基本属性特征
    1.不同卡类型的 性别 比较：
        普通卡和青年卡 男女性比例较为均衡，基本为1：1；金卡的男性持有者比例相较女性持有者明显更多
    2.不同卡类型的 年龄 比较:
        普通卡和金卡的持有者年龄主要集中在30~60岁之间；而青年卡则普遍集中在25岁以内，卡类型设计与目标对象相符
    3.不同类型卡的持卡人在办卡前一年内的 平均帐户余额 对比：
        金卡持有者的办卡前一年的 平均余额 是要显著高于 普通卡 和 青年卡 的，卡类型设计与目标对象相符
    4.不同类型持卡人在办卡前一年内的 平均收入和平均支出 对比：
        三种类型的 平均收入、平均支出 排序均符合：金卡 > 普通卡 > 青年卡，金卡的持有人群为收入较高的群体，
        同样其支出情况也相应高于普通持卡人群，而青年卡，由于其持卡人群多为年龄层较小的人群，收入支出均较低，
        卡类型设计与目标对象情况相符





##### 四、贷款违约预测模型

###### 使用 热力图 查看个变量间的相关性
![avatar](https://cdn.kesci.com/upload/image/qbe26cbtlr.png?imageView2/0/w/960/h/960)


###### 选择最终模型使用的变量


从热力图观察可知：
    贷款信息、居住地信息、经济状况信息内各变量具有高相关性，对变量进行筛选及转换
    1. 贷款信息中：保留amount
    2. 居住地信息：
                1) 采用人均GDP，即对变量进行转换；
                2) 采用失业增长率
    3. 经济状况信息：
                1) 客户放款前近一年总结息（反应实际存款数额）
                2) 收支比（反应客户消费水平）
                3) 可用余额变异系数（反应客户生活状态稳定系数）


```
data_model['GDP_per'] = data_model['GDP']/data_model['A4']  # 人均GDP人民生活水平的一个标准
data_model['unemployment'] = data_model['A13']/data_model['A12']    # 失业增长率一定程度上反应经济增长率
data_model['out/in'] =  data_model['out']/data_model['income']  # 消费占收入比重，一定程度反应客户消费水平
data_model['balance_a']  =  data_model['balance_std']/data_model['balance_mean']    # 可用余额变异系数
var = ['account_id','sex_kind','age','amount','GDP_per','unemployment','out/in','balance_a']
# print(data_model)
# print(list(data_model))

# 逻辑回归构建
data_model = data_model[var+[y]]
for_predict = data_model[data_model[y]==2]  # loan_status 为2表示状态C，即：待定
data_model = data_model[data_model[y]!=2]

# 定义自变量和因变量
import numpy as np
X = data_model[var]
Y = data_model[y]

# 将样本数据建立训练集和测试集，测试集取20%的数据
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1234)

```
###### 建模(使用逻辑回归L1正则化参数)
```
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(penalty='l1',solver='liblinear')
var_temp = ['sex_kind','age','amount','GDP_per','unemployment','out/in','balance_a']
x_train_temp = x_train[var_temp]
clf = LR.fit(x_train_temp,y_train) # 拟合

x_test_temp = x_test[var_temp]
y_pred = clf.predict(x_test_temp)   # 预测测试集数据
test_result = pd.DataFrame({'account_id':x_test['account_id'],'y_predict':clf.predict(x_test_temp)})
new_test_result = test_result.reset_index(drop=True)
print(test_result)  # 输出测试集中 account_id 对应的贷款状态预测
print(new_test_result)  # 输出测试集中 account_id 对应的贷款状态预测


print(clf.coef_)    #查看各变量的回归系数
```

###### 建模结果评价
```
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

模型的精确率0.87，召回率0.84，f1_score为0.82

######  绘制ROC曲线
```
from sklearn.metrics import roc_curve, auc
fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange',label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC_curve')
plt.legend(loc="lower right")
plt.show()
```

![avatar](https://cdn.kesci.com/upload/image/qbe272hidt.png?imageView2/0/w/960/h/960)



<!-- ![avatar](https://raw.githubusercontent.com/fzzhzdev/creditdemo/main/dataanalysicresult/pic/ROC.png) -->