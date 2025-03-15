#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
泰坦尼克号生存预测 - 训练模型
使用图神经网络（GCN）学习乘客关系并预测生存率
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, add_self_loops
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# 数据处理函数
def dataprocessor_titanic(dataframe, is_training=True):
    """优化的数据处理函数，支持训练和测试模式"""
    df = dataframe.copy()  # 避免修改原始数据
    
    # 标题分组处理（更简洁）
    title_mapping = {
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
        'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare', 
        'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
    }
    df['Title'] = df['Title'].replace(title_mapping)
    
    # 删除不需要的列
    df = df.drop(['Cabin', 'Ticket'], axis=1)
    
    # 处理Embarked（保留原始数据，用众数填充）
    if is_training:
        # 训练模式：记录众数
        most_frequent_embarked = df['Embarked'].mode()[0]
        df['Embarked'] = df['Embarked'].fillna(most_frequent_embarked)
    else:
        # 测试模式：使用S作为默认值(训练集众数)
        df['Embarked'] = df['Embarked'].fillna('S')
    
    # 更高效的年龄填充
    age_means = df.groupby(['Pclass', 'Title'])['Age'].transform('mean')
    pclass_means = df.groupby(['Pclass'])['Age'].transform('mean')
    overall_mean = df['Age'].mean()
    
    # 多级填充策略
    df['Age'] = df['Age'].fillna(age_means)
    mask = df['Age'].isna()
    df.loc[mask, 'Age'] = pclass_means[mask]
    df['Age'] = df['Age'].fillna(overall_mean)
    
    return df

# 边添加函数
def add_edge(edge_index, edge_attr, relationships, node1, node2, weight):
    """辅助函数，添加无向边并更新关系集合"""
    if (node1, node2) not in relationships:
        edge_index.append([node1, node2])
        edge_index.append([node2, node1])
        edge_attr.extend([weight, weight])
        relationships.add((node1, node2))
        relationships.add((node2, node1))

# 图构建函数
def build_titanic_graph(df, use_self_loops=True):
    """改进的图构建函数，添加了自环选项，使用Embarked特征"""
    # 1. 特征处理
    # 确保所有特征都有值
    df_clean = df.copy()
    
    # 处理可能的缺失值
    df_clean['Fare'] = df_clean['Fare'].fillna(df_clean['Fare'].median())
    
    # 扩展特征集
    features = pd.concat([
        pd.DataFrame(StandardScaler().fit_transform(df_clean[['Age', 'SibSp', 'Parch', 'Fare']]), 
                    columns=['Age', 'SibSp', 'Parch', 'Fare']),
        pd.get_dummies(df_clean[['Pclass', 'Sex', 'Embarked']])
    ], axis=1)
    features = features.astype(np.float32)

    x = torch.tensor(features.values, dtype=torch.float)
    
    # 只有训练数据有存活标签
    if 'Survived' in df_clean.columns:
        y = torch.tensor(df_clean['Survived'].values, dtype=torch.long)
    else:
        y = None
    
    # 2. 构建边关系
    edge_index = []
    edge_attr = []
    relationships = set()
    
    # 按姓氏分组
    families = defaultdict(list)
    for i in range(len(df_clean)):
        families[df_clean.iloc[i]['Last Name']].append(i)
    
    # 处理每个家庭
    for members in families.values():
        if len(members) <= 1:
            continue  # 忽略单人家庭
        
        # 创建家庭成员信息字典，方便查询
        family_info = {}
        for idx in members:
            family_info[idx] = {
                'sex': df_clean.iloc[idx]['Sex'],
                'sibsp': df_clean.iloc[idx]['SibSp'],
                'parch': df_clean.iloc[idx]['Parch']
            }
        
        # 按SibSp/Parch分组找兄弟姐妹
        sibling_groups = defaultdict(list)
        for idx in members:
            key = (family_info[idx]['sibsp'], family_info[idx]['parch'])
            sibling_groups[key].append(idx)
        
        # 处理兄弟姐妹关系
        for siblings in sibling_groups.values():
            if len(siblings) > 1:
                for i in range(len(siblings)):
                    for j in range(i+1, len(siblings)):
                        add_edge(edge_index, edge_attr, relationships, siblings[i], siblings[j], 0.8)
        
        # 处理父母子女关系
        for parent in members:
            parent_parch = family_info[parent]['parch']
            if parent_parch > 0:  # 可能有子女
                for key, children in sibling_groups.items():
                    if parent not in children and len(children) == parent_parch:
                        for child in children:
                            add_edge(edge_index, edge_attr, relationships, parent, child, 0.9)
        
        # 处理夫妻关系
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                i_idx, j_idx = members[i], members[j]
                if (i_idx, j_idx) in relationships:
                    continue
                
                # 优化的夫妻关系判断
                if (family_info[i_idx]['sex'] != family_info[j_idx]['sex'] and  # 性别不同
                    family_info[i_idx]['sibsp'] >= 1 and family_info[j_idx]['sibsp'] >= 1 and  # 都有配偶
                    family_info[i_idx]['parch'] == family_info[j_idx]['parch']):  # 子女数量相同
                    add_edge(edge_index, edge_attr, relationships, i_idx, j_idx, 1.0)
    
    # 3. 构建图
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # 如果需要自环
        if use_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# 教师GCN模型定义
class TitanicGCN(torch.nn.Module):
    """用于泰坦尼克号生存预测的GCN模型"""
    def __init__(self, in_channels, hidden_channels=32, dropout=0.5):
        super(TitanicGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels//2, add_self_loops=True)
        self.out = torch.nn.Linear(hidden_channels//2, 2)  # 二分类问题
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # 第一层GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层GCN
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # 输出层
        x = self.out(x)
        return x
    
    def get_embeddings(self, x, edge_index):
        """获取节点嵌入"""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.relu(x)

# 新增：蒸馏学生模型定义（更大的容量）
class TitanicGCNDistilled(torch.nn.Module):
    """蒸馏版泰坦尼克号GCN模型，有更大的容量"""
    def __init__(self, in_channels, hidden_channels=64, dropout=0.4):
        super(TitanicGCNDistilled, self).__init__()
        # 第一层使用更宽的隐藏层
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        # 中间层
        self.conv2 = GCNConv(hidden_channels, hidden_channels//2, add_self_loops=True)
        # 额外添加一层全连接层
        self.fc = torch.nn.Linear(hidden_channels//2, hidden_channels//2)
        # 输出层
        self.out = torch.nn.Linear(hidden_channels//2, 2)
        self.dropout = dropout
        
    def forward(self, x, edge_index, return_emb=False):
        # 第一层GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层GCN
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # 保存嵌入用于返回
        emb = x
        
        # 额外的全连接层
        x = self.fc(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout/2, training=self.training)
        
        # 输出层
        x = self.out(x)
        
        if return_emb:
            return x, emb
        return x
        
    def get_embeddings(self, x, edge_index):
        _, emb = self.forward(x, edge_index, return_emb=True)
        return emb

# 训练单个轮次
def train_epoch(model, graph, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    out = model(graph.x, graph.edge_index)
    loss = criterion(out, graph.y)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    # 计算准确率
    pred = out.argmax(dim=1)
    acc = accuracy_score(graph.y.cpu().numpy(), pred.cpu().numpy())
    
    return loss.item(), acc

# 评估函数
def evaluate_model(model, graph):
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index)
        pred = out.argmax(dim=1)
        
        acc = accuracy_score(graph.y.cpu().numpy(), pred.cpu().numpy())
        f1 = f1_score(graph.y.cpu().numpy(), pred.cpu().numpy())
        
    return acc, f1

# 新增：训练教师模型（使用交叉验证）
def train_teacher_models(graph, k=5, epochs=150):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    indices = np.arange(graph.num_nodes)
    
    teacher_models = []
    fold_val_accs = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"训练教师模型 {fold+1}/{k}")
        
        # 创建训练掩码和验证掩码
        train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        
        # 初始化模型
        model = TitanicGCN(in_channels=graph.x.shape[1], hidden_channels=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        # 针对不平衡类别的损失函数
        n_samples = len(train_idx)
        n_survived = graph.y[train_idx].sum().item()
        n_died = n_samples - n_survived
        weight = torch.tensor([n_samples/(2*n_died), n_samples/(2*n_survived)]).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        
        best_val_acc = 0
        best_state = None
        
        # 训练循环
        for epoch in range(epochs):
            # 训练
            model.train()
            optimizer.zero_grad()
            out = model(graph.x, graph.edge_index)
            loss = criterion(out[train_mask], graph.y[train_mask])
            loss.backward()
            optimizer.step()
            
            # 验证
            model.eval()
            with torch.no_grad():
                out = model(graph.x, graph.edge_index)
                pred = out.argmax(dim=1)
                
                # 计算验证集准确率
                val_correct = (pred[val_mask] == graph.y[val_mask]).sum().item()
                val_acc = val_correct / val_mask.sum().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = model.state_dict().copy()
                    
            if (epoch + 1) % 30 == 0:
                print(f"Fold {fold+1}, Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")
        
        # 加载最佳模型
        model.load_state_dict(best_state)
        teacher_models.append(model)
        fold_val_accs.append(best_val_acc)
        
        print(f"Fold {fold+1} 完成. 最佳验证准确率: {best_val_acc:.4f}")
        
    print(f"所有折的平均验证准确率: {np.mean(fold_val_accs):.4f}")
    return teacher_models

# 新增：知识蒸馏到学生模型
def distill_knowledge(teacher_models, graph, epochs=100, temp=3.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    
    # 初始化学生模型(容量更大)
    student_model = TitanicGCNDistilled(
        in_channels=graph.x.shape[1], 
        hidden_channels=64
    ).to(device)
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=15, verbose=True
    )
    
    # 标准分类损失
    hard_loss_fn = torch.nn.CrossEntropyLoss()
    
    # 蒸馏损失权重
    alpha = 0.7  # 软标签权重
    
    best_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        student_model.train()
        optimizer.zero_grad()
        
        # 学生模型输出
        student_logits = student_model(graph.x, graph.edge_index)
        
        # 教师模型集成输出
        with torch.no_grad():
            teacher_logits = torch.zeros((graph.num_nodes, 2), device=device)
            for teacher in teacher_models:
                teacher.eval()
                teacher_logits += F.softmax(teacher(graph.x, graph.edge_index) / temp, dim=1)
            teacher_logits /= len(teacher_models)  # 平均教师概率
        
        # 计算蒸馏损失(软标签)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / temp, dim=1),
            teacher_logits,
            reduction='batchmean'
        ) * (temp * temp)
        
        # 计算真实标签损失(硬标签)
        hard_loss = hard_loss_fn(student_logits, graph.y)
        
        # 总损失
        loss = alpha * soft_loss + (1 - alpha) * hard_loss
        
        loss.backward()
        optimizer.step()
        
        # 评估模型
        student_model.eval()
        with torch.no_grad():
            pred = student_model(graph.x, graph.edge_index).argmax(dim=1)
            correct = (pred == graph.y).sum().item()
            acc = correct / graph.num_nodes
            
            if acc > best_acc:
                best_acc = acc
                best_state = student_model.state_dict().copy()
        
        # 学习率调整
        scheduler.step(acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")
    
    print(f"蒸馏完成. 最佳准确率: {best_acc:.4f}")
    
    # 加载最佳状态
    student_model.load_state_dict(best_state)
    return student_model

def main():
    # 加载训练数据
    print("加载和处理训练数据...")
    train_df = pd.read_csv("/Users/yangdajing/Desktop/CS224W/Datasets/titanic/titanic_name_splited_train.csv")
    
    # 数据处理
    processed_train = dataprocessor_titanic(train_df, is_training=True)
    print(f"训练集大小: {len(processed_train)}")
    
    # 构建训练图
    print("构建训练图...")
    train_graph = build_titanic_graph(processed_train, use_self_loops=True)
    print(f"训练图节点数量: {train_graph.num_nodes}")
    print(f"训练图边数量: {train_graph.num_edges}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 训练多个教师模型(交叉验证)
    print("\n开始训练教师模型(交叉验证)...")
    teacher_models = train_teacher_models(train_graph, k=5, epochs=150)
    
    # 2. 将知识蒸馏到学生模型
    print("\n开始知识蒸馏过程...")
    final_model = distill_knowledge(teacher_models, train_graph, epochs=100, temp=3.0)
    
    # 3. 保存最终蒸馏模型
    distilled_model_path = 'best_titanic_gcn_distilled.pt'
    torch.save(final_model.state_dict(), distilled_model_path)
    print(f"蒸馏模型已保存至: {distilled_model_path}")
    
    # 评估最终模型
    train_acc, train_f1 = evaluate_model(final_model, train_graph)
    print(f'最终蒸馏模型训练集性能 - 准确率: {train_acc:.4f}, F1分数: {train_f1:.4f}')
    
    # 混淆矩阵
    with torch.no_grad():
        pred = final_model(train_graph.x, train_graph.edge_index).argmax(dim=1).cpu().numpy()
        cm = confusion_matrix(train_graph.y.cpu().numpy(), pred)
        print("混淆矩阵:")
        print(cm)
        print("\n分类报告:")
        print(classification_report(train_graph.y.cpu().numpy(), pred))
    
    # 为了向后兼容也保存一个普通模型
    # 初始化标准模型
    print("\n同时训练标准GCN模型(向后兼容)...")
    standard_model = TitanicGCN(
        in_channels=train_graph.x.shape[1],
        hidden_channels=32,
        dropout=0.3
    ).to(device)
    
    # 将数据移动到设备
    train_graph = train_graph.to(device)
    
    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(standard_model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练循环
    print("训练标准模型...")
    epochs = 200
    best_f1 = 0
    patience = 20
    counter = 0
    
    for epoch in range(epochs):
        # 训练
        loss, acc = train_epoch(standard_model, train_graph, optimizer, criterion)
        
        # 每10轮评估一次
        if (epoch + 1) % 10 == 0:
            val_acc, val_f1 = evaluate_model(standard_model, train_graph)
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Train Acc: {acc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
            
            # 早停
            if val_f1 > best_f1:
                best_f1 = val_f1
                counter = 0
                # 保存最佳模型
                torch.save(standard_model.state_dict(), 'best_titanic_gcn.pt')
                print(f"标准模型已保存: best_titanic_gcn.pt (F1={best_f1:.4f})")
            else:
                counter += 1
                if counter >= patience:
                    print(f'早停! 最佳F1: {best_f1:.4f}')
                    break
    
    print("训练完成！")
    print(f"1. 蒸馏模型已保存到: {distilled_model_path}")
    print(f"2. 标准模型已保存到: best_titanic_gcn.pt")
    
    return final_model, train_graph

if __name__ == "__main__":
    main() 