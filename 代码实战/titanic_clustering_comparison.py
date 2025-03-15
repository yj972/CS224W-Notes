import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, add_self_loops

# 数据处理函数
def dataprocessor_titanic(dataframe, is_training=True):
    """优化的数据处理函数，支持训练和测试模式"""
    df = dataframe.copy()  # 避免修改原始数据
    
    # 标题分组处理
    title_mapping = {
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
        'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare', 
        'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
    }
    df['Title'] = df['Title'].replace(title_mapping)
    
    # 删除不需要的列
    df = df.drop(['Cabin', 'Ticket'], axis=1)
    
    # 处理Embarked
    if is_training:
        most_frequent_embarked = df['Embarked'].mode()[0]
        df['Embarked'] = df['Embarked'].fillna(most_frequent_embarked)
    else:
        df['Embarked'] = df['Embarked'].fillna('S')
    
    # 年龄填充
    age_means = df.groupby(['Pclass', 'Title'])['Age'].transform('mean')
    pclass_means = df.groupby(['Pclass'])['Age'].transform('mean')
    overall_mean = df['Age'].mean()
    
    df['Age'] = df['Age'].fillna(age_means)
    mask = df['Age'].isna()
    df.loc[mask, 'Age'] = pclass_means[mask]
    df['Age'] = df['Age'].fillna(overall_mean)
    
    return df

# 辅助函数，添加无向边并更新关系集合
def add_edge(edge_index, edge_attr, relationships, node1, node2, weight):
    if (node1, node2) not in relationships:
        edge_index.append([node1, node2])
        edge_index.append([node2, node1])
        edge_attr.extend([weight, weight])
        relationships.add((node1, node2))
        relationships.add((node2, node1))

# 构建图函数
def build_titanic_graph(df):
    """构建泰坦尼克号乘客关系图，保留家庭关系"""
    df_reset = df.reset_index(drop=True)
    
    # 处理节点特征
    features = pd.concat([
        pd.DataFrame(StandardScaler().fit_transform(df_reset[['Age', 'SibSp', 'Parch']]), 
                    columns=['Age', 'SibSp', 'Parch']),
        pd.get_dummies(df_reset[['Pclass', 'Sex']])
    ], axis=1)
    features = features.astype(np.float32)

    x = torch.tensor(features.values, dtype=torch.float)
    y = torch.tensor(df_reset['Survived'].values, dtype=torch.long) if 'Survived' in df_reset.columns else None
    
    # 构建边关系
    edge_index = []
    edge_attr = []
    relationships = set()
    
    # 按姓氏分组
    families = defaultdict(list)
    for i in range(len(df_reset)):
        families[df_reset.iloc[i]['Last Name']].append(i)
    
    # 处理每个家庭
    for members in families.values():
        if len(members) <= 1:
            continue  # 忽略单人家庭
        
        # 创建家庭成员信息字典，方便查询
        family_info = {}
        for idx in members:
            family_info[idx] = {
                'sex': df_reset.iloc[idx]['Sex'],
                'sibsp': df_reset.iloc[idx]['SibSp'],
                'parch': df_reset.iloc[idx]['Parch']
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
    
    # 构建图
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# 定义GCN模型
class ImprovedGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ImprovedGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x

# 主函数
def main():
    print("加载数据...")
    # 加载泰坦尼克号数据集
    # 请根据实际路径修改
    titanic = pd.read_csv("Datasets/titanic/titanic_name_splited_train.csv")
    
    print("处理数据...")
    processed_df = dataprocessor_titanic(titanic)
    
    print("构建图...")
    G = build_titanic_graph(processed_df)
    
    # 转换为networkx图进行可视化
    G_nx = nx.Graph()
    for i in range(G.x.shape[0]):
        G_nx.add_node(i)
    
    edge_index = G.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        G_nx.add_edge(edge_index[0, i], edge_index[1, i])
    
    print(f"图节点数量: {G_nx.number_of_nodes()}")
    print(f"图边数量: {G_nx.number_of_edges()}")
    
    # 确保图是连通的
    valid_nodes = list(range(len(G.y)))
    G_clean = G_nx.subgraph(valid_nodes)
    
    # 将NetworkX图转回PyG格式
    G_clean_pyg = from_networkx(G_clean)
    G_clean_pyg.x = G.x[:len(G.y)]
    G_clean_pyg.y = G.y
    
    edge_index = torch.tensor(list(G_clean.edges())).t().contiguous()
    edge_index, _ = add_self_loops(edge_index, num_nodes=G_clean_pyg.x.shape[0])
    G_clean_pyg.edge_index = edge_index
    
    print("生成图嵌入...")
    # 初始化并训练GCN模型获取节点嵌入
    in_channels = G_clean_pyg.x.shape[1]
    model = ImprovedGCN(in_channels=in_channels, hidden_channels=16)
    
    # 获取节点嵌入
    model.eval()
    with torch.no_grad():
        node_embeddings = model(G_clean_pyg.x, G_clean_pyg.edge_index)
    
    # 检查嵌入向量是否包含NaN
    X = node_embeddings.detach().cpu().numpy()
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"嵌入向量中NaN值的数量: {nan_count}")
        X = np.nan_to_num(X, nan=0.0)  # 用0填充NaN
    
    print("进行降维...")
    # 使用TSNE进行降维
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    embeddings_2d = tsne.fit_transform(X)
    
    print("进行聚类分析...")
    # 设置聚类数量
    n_clusters = 5
    
    # 1. 对原始特征进行聚类
    original_features = processed_df[['Age', 'SibSp', 'Parch', 'Fare']].copy()
    original_features = original_features.fillna(original_features.mean())
    original_scaled = StandardScaler().fit_transform(original_features)
    
    kmeans_original = KMeans(n_clusters=n_clusters, random_state=42)
    original_clusters = kmeans_original.fit_predict(original_scaled)
    
    # 2. 对图嵌入向量进行聚类
    kmeans_embedding = KMeans(n_clusters=n_clusters, random_state=42)
    embedding_clusters = kmeans_embedding.fit_predict(X)
    
    print("可视化聚类结果...")
    # 3. 可视化原始特征聚类结果
    pca = PCA(n_components=2)
    original_2d = pca.fit_transform(original_scaled)
    
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(original_2d[:, 0], original_2d[:, 1], 
               c=original_clusters, cmap='viridis',
               alpha=0.7)
    plt.title("原始特征聚类 (PCA降维)")
    plt.colorbar(label='聚类')
    
    # 4. 可视化图嵌入聚类结果
    plt.subplot(1, 2, 2)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
               c=embedding_clusters, cmap='viridis',
               alpha=0.7)
    plt.title("图嵌入聚类 (TSNE降维)")
    plt.colorbar(label='聚类')
    
    plt.tight_layout()
    plt.savefig('titanic_clustering_comparison.png')
    plt.show()
    
    # 5. 分析各聚类的生存率差异
    plt.figure(figsize=(12, 5))
    
    # 原始特征聚类的生存率
    plt.subplot(1, 2, 1)
    survival_rates_original = []
    for i in range(n_clusters):
        cluster_indices = np.where(original_clusters == i)[0]
        if len(cluster_indices) > 0:
            survival_rate = G_clean_pyg.y[cluster_indices].float().mean().item()
            survival_rates_original.append(survival_rate)
        else:
            survival_rates_original.append(0)
    
    plt.bar(range(n_clusters), survival_rates_original)
    plt.xlabel("聚类")
    plt.ylabel("存活率")
    plt.title("原始特征聚类的存活率")
    plt.ylim(0, 1)
    
    # 图嵌入聚类的生存率
    plt.subplot(1, 2, 2)
    survival_rates_embedding = []
    for i in range(n_clusters):
        cluster_indices = np.where(embedding_clusters == i)[0]
        if len(cluster_indices) > 0:
            survival_rate = G_clean_pyg.y[cluster_indices].float().mean().item()
            survival_rates_embedding.append(survival_rate)
        else:
            survival_rates_embedding.append(0)
    
    plt.bar(range(n_clusters), survival_rates_embedding)
    plt.xlabel("聚类")
    plt.ylabel("存活率")
    plt.title("图嵌入聚类的存活率")
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('titanic_survival_rates_comparison.png')
    plt.show()
    
    # 6. 计算标准差，检验聚类质量
    original_survival_std = np.std(survival_rates_original)
    embedding_survival_std = np.std(survival_rates_embedding)
    
    print(f"\n聚类生存率标准差对比:")
    print(f"原始特征聚类生存率标准差: {original_survival_std:.4f}")
    print(f"图嵌入聚类生存率标准差: {embedding_survival_std:.4f}")
    print(f"标准差提升比例: {(embedding_survival_std/original_survival_std - 1)*100:.2f}%")
    
    # 7. 输出每种聚类方法的聚类大小
    print("\n原始特征聚类大小:")
    for i in range(n_clusters):
        print(f"聚类 {i}: {np.sum(original_clusters == i)} 个乘客, 存活率: {survival_rates_original[i]:.2f}")
    
    print("\n图嵌入聚类大小:")
    for i in range(n_clusters):
        print(f"聚类 {i}: {np.sum(embedding_clusters == i)} 个乘客, 存活率: {survival_rates_embedding[i]:.2f}")

if __name__ == "__main__":
    main() 