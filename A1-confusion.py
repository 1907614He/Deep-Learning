#%%导入库
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#%% 1.数据处理
#数据导入
data = pd.read_csv('D:/HeCloud/Master of Data Science/2023 Trimester 3/Deep Learning Fundamentals/A1/data/diabetes.csv')
data.head()
X = data[['Pregnancies','Glucose','BloodPressure','SkinThickness',
                      'Insulin','BMI','DiabetesPedigreeFunction','Age']]
Y = data['Outcome']

#使用过采样方法将数据集变为平衡数据集
smote = SMOTE(random_state=42)
X, Y = smote.fit_resample(X, Y)

#划分训练集和测试集
Y = Y*2-1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1907614)


## 数据转化
#将dataframe形式转化为array形式
X_train=np.array(X_train)
Y_train=np.array(Y_train)
X_test=np.array(X_test)
Y_test=np.array(Y_test)
# 将数据转换为PyTorch张量  
X_train = torch.tensor(X_train, dtype=torch.float32)  
Y_train = torch.tensor(Y_train, dtype=torch.float32)  
X_test = torch.tensor(X_test, dtype=torch.float32)  
Y_test = torch.tensor(Y_test, dtype=torch.float32)
#%% 2.创建一个MLP感知机模型 
class Perceptron(nn.Module):  
    def __init__(self):  
        super(Perceptron, self).__init__() 
        #定义了要在forward函数中调用的全连接网络nn.Linear
        # 每一层用self引导，d1是这一层的名字
        self.d1 = nn.Linear(8, 10)  #input_dim:神经元个数,这里因为共有8个参数所以设置为8.
        self.tanh = nn.Tanh()
        self.d2 = nn.Linear(10, 1) #输出只有一个标签，所以为1
        
  
    def forward(self, x):  
        #调用self.d1网络结构块，实现从输入x到输出y的前向传播
        out = self.d1(x)
        out = self.tanh(out)
        out = self.d2(out)
        return out 
    
#初始化模型
model = Perceptron()
print(model)
#%% 3.配置训练方法
#创建 DataLoader 用于批量加载数据
batch_size =50
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#创建一个随机梯度下降（SGD）（stochastic gradient descent）优化器
Optimizer = optim.SGD(model.parameters(), lr=0.01)#选择SGD优化器，学习率设置为0.01
#创建一个损失函数（MSE均方误差）
criterion = nn.MSELoss() 
#%%# 4.训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for batch_X, batch_Y in train_loader:
        Optimizer.zero_grad()# 清空梯度 
        model_outputs = model.forward(batch_X) # 前向传播
        loss = criterion(model_outputs, batch_Y.view(-1, 1))# 计算损失 
        loss.backward() # 反向传播，计算梯度  
        Optimizer.step()# 更新权重
    # 打印每个训练周期的损失值  
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
#%% 5.评估模型
model.eval()
with torch.no_grad():
    predictions = model.forward(X_test)
    predictions = torch.sign(predictions)
    accuracy = (predictions == Y_test.view(-1, 1)).float().mean()
    print("Accuracy:", accuracy.item())

confusion=confusion_matrix(predictions, Y_test)
# 可视化混淆矩阵
plt.figure(figsize=(6, 4))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
            yticklabels=["Predicted Negative", "Predicted Positive"],
            xticklabels=["Actual Negative", "Actual Positive"])
plt.ylabel('Predicted Lable')
plt.xlabel('Actual Lable')
plt.title('Confsion Matrix')
plt.show()


    
    
    
    