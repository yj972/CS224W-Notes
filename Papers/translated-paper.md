# 基于图神经网络的链接预测

**作者**：
Muhan Zhang
华盛顿大学圣路易斯分校
计算机科学与工程系
muhan@wustl.edu

Yixin Chen
华盛顿大学圣路易斯分校
计算机科学与工程系
chen@cse.wustl.edu

## 摘要

链接预测是网络结构数据的关键问题。链接预测启发式方法使用一些评分函数，如共同邻居和Katz指数，来衡量链接的可能性。由于其简单性、可解释性以及对于一些方法而言的可扩展性，它们已获得广泛的实际应用。然而，每种启发式方法对于两个节点何时可能链接都有强烈的假设，这限制了它们在这些假设失效的网络上的有效性。在这方面，一个更合理的方式应该是从给定网络中学习适合的启发式方法，而不是使用预定义的方法。通过提取每个目标链接周围的局部子图，我们旨在学习一个将子图模式映射到链接存在的函数，从而自动学习适合当前网络的"启发式方法"。在本文中，我们研究这种链接预测的启发式学习范式。首先，我们提出了一个新颖的γ衰减启发式理论。该理论在单一框架中统一了广泛的启发式方法，并证明所有这些启发式方法都可以从局部子图中得到很好的近似。我们的结果表明，局部子图保留了与链接存在相关的丰富信息。其次，基于γ衰减理论，我们提出了一种使用图神经网络(GNN)从局部子图学习启发式方法的新方法。其实验结果显示了前所未有的性能，在广泛的问题上始终表现良好。

## 1 引言

链接预测是预测网络中两个节点是否可能有链接[1]。鉴于网络的无处不在，它有很多应用，如好友推荐[2]、电影推荐[3]、知识图谱补全[4]和代谢网络重建[5]。

链接预测的一类简单而有效的方法被称为启发式方法。启发式方法计算一些启发式节点相似度分数作为链接的可能性[1, 6]。现有的启发式方法可以根据计算分数所需的最大跳数邻居进行分类。例如，共同邻居(CN)和优先连接(PA)[7]是一阶启发式方法，因为它们只涉及两个目标节点的一跳邻居。Adamic-Adar(AA)和资源分配(RA)[8]是二阶启发式方法，因为它们是从目标节点的最多两跳邻域计算的。我们定义h阶启发式方法为那些需要知道目标节点的最多h跳邻域的启发式方法。也有一些需要知道整个网络的高阶启发式方法。例子包括Katz、根页面排名(PR)[9]和SimRank(SR)[10]。附录A的表3总结了八种流行的启发式方法。

虽然在实践中效果良好，但启发式方法对链接可能存在的情况有强烈的假设。例如，共同邻居启发式假设两个节点如果有许多共同邻居，它们更可能连接。这一假设在社交网络中可能是正确的，但在蛋白质-蛋白质相互作用(PPI)网络中被证明是失效的——共享许多共同邻居的两个蛋白质实际上不太可能相互作用[11]。

实际上，启发式方法属于更通用的类别，即图结构特征。图结构特征是那些位于网络中观察到的节点和边结构内的特征，可以直接从图中计算。由于启发式方法可以被视为预定义的图结构特征，一个自然的想法是从网络中自动学习这些特征。Zhang和Chen[12]首次研究了这个问题。他们提取链接周围的局部封闭子图作为训练数据，并使用全连接神经网络学习哪些封闭子图对应于链接存为Weisfeiler-Lehm在。他们的方法称an神经机器(WLNM)，已经达到了最先进的链接预测性能。节点对(x, y)的封闭子图是由x和y的最多h跳邻居的并集从网络中诱导的子图。图1说明了(A, B)和(C, D)的1跳封闭子图。这些封闭子图对链接预测非常有信息——所有一阶启发式方法，如共同邻居，都可以直接从1跳封闭子图中计算出来。

然而，高阶启发式方法如根页面排名和Katz通常比一阶和二阶方法具有更好的性能[6]。为了有效地学习好的高阶特征，似乎我们需要一个非常大的跳数h，使得封闭子图成为整个网络。这对于大多数实际网络来说会导致不可承受的时间和内存消耗。但我们真的需要这么大的h来学习高阶启发式方法吗？

幸运的是，作为我们的第一个贡献，我们表明我们不一定需要一个非常大的h来学习高阶图结构特征。我们深入研究链接预测启发式方法的内在机制，发现大多数高阶启发式方法可以通过γ衰减理论统一起来。我们证明，在温和条件下，任何γ衰减启发式方法都可以从h跳封闭子图中有效近似，其中近似误差至少以h为指数速度减小。这意味着我们可以安全地使用甚至很小的h来学习好的高阶特征。这也意味着这些高阶启发式方法的"有效阶数"并不那么高。

基于我们的理论结果，我们提出了一个新颖的链接预测框架SEAL，用于从局部封闭子图中学习一般图结构特征。SEAL修复了WLNM的多个缺点。首先，使用图神经网络(GNN)[13, 14, 15, 16, 17]代替WLNM中的全连接神经网络，这使得更好的图特征学习能力成为可能。其次，SEAL允许不仅从子图结构中学习，还可以从潜在和显式节点特征中学习，从而吸收多种类型的信息。我们通过实验验证了其大幅提高的性能。

我们的贡献总结如下：1) 我们提出了一个学习链接预测启发式方法的新理论，证明了从局部子图而不是整个网络中学习的合理性；2) 我们提出了SEAL，一个基于GNN的新型链接预测框架（如图1所示）。SEAL在性能上大幅超过所有启发式方法、潜在特征方法和最近的网络嵌入方法。SEAL也优于之前最先进的方法WLNM。

## 2 预备知识

**符号** 设G = (V, E)为一个无向图，其中V是顶点集，E ⊆ V × V是观察到的链接集。其邻接矩阵为A，其中如果(i, j) ∈ E则Ai,j = 1，否则Ai,j = 0。

对于任何节点x, y ∈ V，设Γ(x)为x的1跳邻居，d(x, y)为x和y之间的最短路径距离。一个行走w = ⟨v0, ··· , vk⟩是一个节点序列，其中(vi, vi+1) ∈ E。我们使用|⟨v0, ··· , vk⟩|表示行走w的长度，这里是k。

**潜在特征和显式特征** 除了图结构特征外，潜在特征和显式特征也被用于链接预测。潜在特征方法[3, 18, 19, 20]分解网络的一些矩阵表示来学习每个节点的低维潜在表示/嵌入。例子包括矩阵分解[3]和随机块模型[18]等。最近，提出了许多网络嵌入技术，如DeepWalk[19]、LINE[21]和node2vec[20]，这些也是潜在特征方法，因为它们也隐式地分解了一些矩阵[22]。显式特征通常以节点属性的形式出现，描述了关于个别节点的各种附加信息。已经证明，将图结构特征与潜在特征和显式特征结合起来可以提高性能[23, 24]。

**图神经网络** 图神经网络(GNN)是一种用于图上学习的新型神经网络[13, 14, 15, 16, 25, 26]。在这里，我们只简要介绍GNN的组成部分，因为本文不是关于GNN创新，而是GNN的一个新颖应用。GNN通常由1)图卷积层，用于提取个别节点的局部子结构特征，和2)图聚合层，用于将节点级特征聚合为图级特征向量。许多图卷积层可以统一到消息传递框架中[27]。

**监督启发式学习** 之前有一些尝试学习链接预测的监督启发式方法。与我们最接近的工作是Weisfeiler-Lehman神经机器(WLNM)[12]，它也从局部子图中学习。然而，WLNM有几个缺点。首先，WLNM在子图的邻接矩阵上训练全连接神经网络。由于全连接神经网络只接受固定大小的张量作为输入，WLNM需要将不同的子图截断为相同的大小，这可能会丢失很多结构信息。其次，由于邻接矩阵表示的限制，WLNM不能从潜在或显式特征中学习。第三，理论证明也缺失。我们在附录D中包含了更多关于WLNM的讨论。另一个相关的研究方向是在不同启发式方法的组合上训练监督学习模型。例如，路径排名算法[28]在不同路径类型的概率上训练逻辑回归来预测知识图谱中的关系。Nickel等人[23]提出将启发式特征纳入张量分解模型。然而，这些模型仍然依赖于预定义的启发式方法——它们不能学习一般的图结构特征。

## 3 统一链接预测启发式方法的理论

在本节中，我们旨在更深入地理解各种链接预测启发式方法背后的机制，从而激发从局部子图中学习启发式方法的想法。由于图学习技术的数量众多，请注意，我们不关心特定方法的泛化误差，而是专注于子图中为计算现有启发式方法保留的信息。

**定义1**（封闭子图）对于图G = (V, E)，给定两个节点x, y ∈ V，(x, y)的h跳封闭子图是由G中节点集{i | d(i, x) ≤ h 或 d(i, y) ≤ h}诱导的子图G^h_x,y。

封闭子图描述了(x, y)的"h跳周围环境"。由于G^h_x,y包含x和y的所有h跳邻居，我们自然有以下定理。

**定理1**。(x, y)的任何h阶启发式方法都可以从G^h_x,y中准确计算。

例如，2跳封闭子图将包含计算任何一阶和二阶启发式方法所需的所有信息。然而，虽然一阶和二阶启发式方法被局部封闭子图很好地覆盖，但学习高阶启发式方法似乎仍需要一个极大的h使得封闭子图成为整个网络。令人惊讶的是，我们接下来的分析表明，即使使用小的h也可以学习高阶启发式方法。我们首先通过定义γ衰减启发式方法来支持这一点。我们将证明，在某些条件下，γ衰减启发式方法可以从h跳封闭子图中得到很好的近似。此外，我们将证明几乎所有著名的高阶启发式方法都可以统一到这个γ衰减启发式框架中。

**定义2**（γ衰减启发式方法）(x, y)的γ衰减启发式方法具有以下形式：

$$H(x, y) = \eta \sum_{l=1}^{\infty} \gamma^l f(x, y, l)$$

其中γ是0到1之间的衰减因子，η是γ的正常数或正函数，在给定网络下，f是x, y, l的非负函数。

接下来，我们将证明，在某些条件下，γ衰减启发式方法可以从h跳封闭子图中近似，且近似误差至少以h为指数速度减小。

**定理2**。给定一个γ衰减启发式方法H(x, y) = η∑_{l=1}^{∞} γ^l f(x, y, l)，如果f(x, y, l)满足：
* （性质1）f(x, y, l) ≤ λ^l，其中λ < 1/γ；以及
* （性质2）f(x, y, l)可以从G^h_x,y中计算得出，其中l = 1, 2, ··· , g(h)，其中g(h) = ah + b，a, b ∈ N且a > 0，

那么H(x, y)可以从G^h_x,y中近似，且近似误差至少以h为指数速度减小。

**证明**：我们可以通过对其前g(h)项求和来近似这样的γ衰减启发式方法。

$$\tilde{H}(x, y) := \eta \sum_{l=1}^{g(h)} \gamma^l f(x, y, l)$$

近似误差可以如下界定：

$$|H(x, y) - \tilde{H}(x, y)| = \eta \sum_{l=g(h)+1}^{\infty} \gamma^l f(x, y, l) \leq \eta \sum_{l=ah+b+1}^{\infty} \gamma^l \lambda^l = \eta (\gamma\lambda)^{ah+b+1} \frac{1}{1-\gamma\lambda}$$

在实践中，小的γλ和大的a导致更快的减小速度。接下来我们将证明三种流行的高阶启发式方法：Katz、根页面排名和SimRank，都是满足定理2中性质的γ衰减启发式方法。首先，我们需要以下引理。

**引理1**。任何长度l ≤ 2h + 1的x和y之间的行走都包含在G^h_x,y中。

**证明**：给定任何长度为l的行走w = ⟨x, v1, ··· , vl-1, y⟩，我们将证明每个节点vi都包含在G^h_x,y中。考虑任意vi。假设d(vi, x) ≥ h + 1且d(vi, y) ≥ h + 1。那么，
2h + 1 ≥ l = |⟨x, v1, ··· , vi⟩| + |⟨vi, ··· , vl-1, y⟩| ≥ d(vi, x) + d(vi, y) ≥ 2h + 2，这是一个矛盾。

因此，d(vi, x) ≤ h或d(vi, y) ≤ h。根据G^h_x,y的定义，vi必须包含在G^h_x,y中。

接下来我们将逐一分析Katz、根页面排名和SimRank。

### 3.1 Katz指数

(x, y)的Katz指数[29]定义为

$$Katz_{x,y} = \sum_{l=1}^{\infty} \beta^l |walks^{\langle l \rangle}(x, y)| = \sum_{l=1}^{\infty} \beta^l [A^l]_{x,y}$$

其中walks^⟨l⟩(x, y)是x和y之间长度为l的行走集合，A^l是网络邻接矩阵的l次方。Katz指数对x和y之间所有行走的集合求和，其中长度为l的行走由β^l (0 < β < 1)衰减，给较短的行走更多权重。

Katz指数直接以η = 1，γ = β和f(x, y, l) = |walks^⟨l⟩(x, y)|的形式定义为γ衰减启发式方法。根据引理1，|walks^⟨l⟩(x, y)|可以从G^h_x,y中计算得出，其中l ≤ 2h + 1，因此定理2中的性质2得到满足。现在我们证明性质1何时得到满足。

**命题1**。对于任何节点i, j，[A^l]_{i,j}由d^l界定，其中d是网络的最大节点度。

**证明**：我们通过归纳法证明。当l = 1时，对于任何(i, j)，A_{i,j} ≤ d。因此基本情况是正确的。
现在，通过归纳假设[A^l]_{i,j} ≤ d^l对于任何(i, j)成立，我们有

$$[A^{l+1}]_{i,j} = \sum_{k=1}^{|V|} [A^l]_{i,k} A_{k,j} \leq d^l \sum_{k=1}^{|V|} A_{k,j} \leq d^l d = d^{l+1}$$

取λ = d，我们可以看到，只要d < 1/β，Katz指数就会满足定理2中的性质1。在实践中，衰减因子β通常设置为很小的值，如5E-4[1]，这意味着Katz可以从h跳封闭子图中得到很好的近似。

### 3.2 PageRank

节点x的根页面排名计算从x开始的随机行走者的稳态分布，该行走者迭代地以概率α移动到其当前位置的随机邻居，或以概率1-α返回到x。设π_x表示稳态分布向量。设[π_x]_i表示随机行走者在稳态分布下位于节点i的概率。

设P为转移矩阵，其中如果(i, j) ∈ E，则P_{i,j} = 1/|Γ(v_j)|，否则P_{i,j} = 0。设e_x为一个向量，其第x个元素为1，其他为0。稳态分布满足

$$\pi_x = \alpha P \pi_x + (1-\alpha) e_x$$

用于链接预测时，(x, y)的分数由[π_x]_y（或对称性地由[π_x]_y + [π_y]_x）给出。

为了证明根页面排名是γ衰减启发式方法，我们引入逆P距离理论[30]，该理论表明[π_x]_y可以等价地写为：

$$[\pi_x]_y = (1-\alpha) \sum_{w:x\rightarrow y} P[w]\alpha^{len(w)}$$

其中求和是对所有从x开始并在y结束的行走w（可能多次触及x和y）进行的。对于行走w = ⟨v_0, v_1, ··· , v_k⟩，len(w) := |⟨v_0, v_1, ··· , v_k⟩|是行走的长度。术语P[w]定义为∏^{k-1}_{i=0} 1/|Γ(v_i)|，可以解释为遍历w的概率。现在我们有以下定理。

**定理3**。根页面排名启发式方法是满足定理2中性质的γ衰减启发式方法。

**证明**：我们首先以下面的形式写[π_x]_y。

$$[\pi_x]_y = (1-\alpha) \sum_{l=1}^{\infty} \sum_{w:x\rightarrow y, len(w)=l} P[w]\alpha^l$$

定义f(x, y, l) := ∑_{w:x→y, len(w)=l} P[w]导致γ衰减启发式方法的形式。注意，f(x, y, l)是从x开始的随机行走者恰好在l步后停在y的概率，这满足∑_{z∈V} f(x, z, l) = 1。因此，f(x, y, l) ≤ 1 < 1/α（性质1）。根据引理1，f(x, y, l)也可以从G^h_x,y中计算得出，其中l ≤ 2h + 1（性质2）。

### 3.3 SimRank

SimRank分数[10]的动机是，如果两个节点的邻居也相似，那么这两个节点是相似的。它以以下递归方式定义：如果x = y，则s(x, y) := 1；否则，

$$s(x, y) := \gamma \frac{\sum_{a\in\Gamma(x)} \sum_{b\in\Gamma(y)} s(a, b)}{|\Gamma(x)| \cdot |\Gamma(y)|}$$

其中γ是0到1之间的常数。根据[10]，SimRank有一个等价定义：

$$s(x, y) = \sum_{w:(x,y)\rightsquigarrow(z,z)} P[w]\gamma^{len(w)}$$

其中w : (x, y) ⇝ (z, z)表示所有同时行走，使得一个行走从x开始，另一个行走从y开始，它们首次在任何顶点z相遇。对于同时行走w = ⟨(v_0, u_0), ··· ,(v_k, u_k)⟩，len(w) = k是行走的长度。术语P[w]类似地定义为∏^{k-1}_{i=0} 1/(|Γ(v_i)|·|Γ(u_i)|)，描述这个行走的概率。现在我们有以下定理。

**定理4**。SimRank是满足定理2中性质的γ衰减启发式方法。

**证明**：我们将s(x, y)写为以下形式。

$$s(x, y) = \sum_{l=1}^{\infty} \sum_{w:(x,y)\rightsquigarrow(z,z), len(w)=l} P[w]\gamma^l$$

定义f(x, y, l) := ∑_{w:(x,y)⇝(z,z), len(w)=l} P[w]揭示SimRank是一个γ衰减启发式方法。注意，f(x, y, l) ≤ 1 < 1/γ。很容易看出f(x, y, l)也可以从G^h_x,y中计算得出，其中l ≤ h。

**讨论** 存在几种其他基于路径计数或随机行走的高阶启发式方法[6]，它们也可以纳入γ衰减启发式框架。我们在这里省略分析。我们的结果揭示，大多数高阶启发式方法本质上共享相同的γ衰减启发式形式，因此可以从h跳封闭子图中有效近似，误差指数级减小。我们相信γ衰减启发式方法的普遍性并非偶然——它意味着一个成功的链接预测启发式方法最好对远离目标的结构赋予指数级较小的权重，因为网络的远端部分直观上对链接存在的贡献很小。我们的结果为从局部子图学习启发式方法奠定了基础，因为它们意味着局部封闭子图已经包含足够的信息来学习链接预测的良好图结构特征，这是非常希望的，考虑到从整个网络学习通常是不可行的。总结来说，从链接周围提取的小封闭子图中，我们能够准确计算一阶和二阶启发式方法，并以小误差近似广泛的高阶启发式方法。因此，给定模型使用的足够的特征学习能力，从这种封闭子图学习预期至少能达到与广泛的启发式方法一样好的性能。有一些相关工作经验性地验证了局部方法通常可以很好地估计PageRank和SimRank[31, 32]。另一个相关的理论工作[33]建立了h的条件，以达到普通PageRank的一些固定近似误差。

## 4 SEAL：使用GNN实现理论

在本节中，我们描述我们的SEAL框架用于链接预测。SEAL不限制学习的特征是某些特定形式如γ衰减启发式方法，而是学习链接预测的一般图结构特征。它包含三个步骤：1）封闭子图提取，2）节点信息矩阵构建，和3）GNN学习。给定一个网络，我们旨在自动学习一个最能解释链接形成的"启发式方法"。受理论结果的启发，这个函数将链接周围的局部封闭子图作为输入，并输出链接存在的可能性。为了学习这样一个函数，我们在封闭子图上训练图神经网络（GNN）。因此，SEAL中的第一步是为一组采样的正链接（观察到的）和一组采样的负链接（未观察到的）提取封闭子图，以构建训练数据。

GNN通常将(A, X)作为输入，其中A（稍微滥用符号）是输入封闭子图的邻接矩阵，X是节点信息矩阵，其每一行对应于一个节点的特征向量。SEAL中的第二步是为每个封闭子图构建节点信息矩阵X。这一步对于训练成功的GNN链接预测模型至关重要。在以下内容中，我们讨论这个关键步骤。SEAL中的节点信息矩阵X有三个组成部分：结构节点标签、节点嵌入和节点属性。

### 4.1 节点标记

X中的第一个组成部分是每个节点的结构标签。节点标记是函数fl: V → N，它为封闭子图中的每个节点i分配一个整数标签fl(i)。目的是使用不同的标签来标记节点在封闭子图中的不同角色：1）中心节点x和y是链接所在的目标节点。2）与中心不同相对位置的节点对链接有不同的结构重要性。适当的节点标记应该标记这些差异。如果我们不标记这些差异，GNN将无法分辨目标节点（即应该预测链接存在的节点对）在哪里，并会丢失结构信息。

我们的节点标记方法源于以下标准：1）两个目标节点x和y总是有独特的标签"1"。2）如果d(i, x) = d(j, x)且d(i, y) = d(j, y)，则节点i和j有相同的标签。第二个标准是因为，直观上，节点i在封闭子图中的拓扑位置可以通过其相对于两个中心节点的半径来描述，即(d(i, x), d(i, y))。因此，我们让处于相同轨道上的节点具有相同的标签，这样节点标签可以反映节点在子图中的相对位置和结构重要性。

基于上述标准，我们提出了一种双半径节点标记（DRNL）方法如下。首先，给x和y分配标签1。然后，对于任何半径为(d(i, x), d(i, y)) = (1, 1)的节点i，分配标签fl(i) = 2。半径为(1, 2)或(2, 1)的节点获得标签3。半径为(1, 3)或(3, 1)的节点获得4。半径为(2, 2)的节点获得5。半径为(1, 4)或(4, 1)的节点获得6。半径为(2, 3)或(3, 2)的节点获得7。以此类推。换句话说，我们反复为相对于两个中心节点具有更大半径的节点分配更大的标签，其中标签fl(i)和双半径(d(i, x), d(i, y))满足：

1) 如果d(i, x) + d(i, y) ≠ d(j, x) + d(j, y)，则d(i, x) + d(i, y) < d(j, x) + d(j, y) ⟺ fl(i) < fl(j)；
2) 如果d(i, x) + d(i, y) = d(j, x) + d(j, y)，则d(i, x)·d(i, y) < d(j, x)·d(j, y) ⟺ fl(i) < fl(j)。

DRNL的一个优点是它有一个完美的哈希函数：

fl(i) = 1 + min(dx, dy) + (d/2)[(d/2) + (d%2) - 1]

其中dx := d(i, x), dy := d(i, y), d := dx + dy, (d/2)和(d%2)分别是d除以2的整数商和余数。这种完美哈希允许快速的闭形式计算。

对于d(i, x) = ∞或d(i, y) = ∞的节点，我们给它们一个空标签0。请注意，DRNL不是节点标记的唯一可能方式，但我们经验性地验证了它比无标记和其他简单标记方式的性能更好。我们在附录B中讨论更多关于节点标记的内容。得到标签后，我们使用它们的独热编码向量来构建X。

### 4.2 结合潜在和显式特征

除了结构节点标签外，节点信息矩阵X还提供了包含潜在和显式特征的机会。通过将每个节点的嵌入/属性向量连接到X中相应的行，我们可以使SEAL同时从三种类型的特征中学习。

为SEAL生成节点嵌入是非常重要的。假设我们有观察到的网络G = (V, E)，一组采样的正训练链接Ep ⊆ E，以及一组采样的负训练链接En，其中En ∩ E = ∅。如果我们直接在G上生成节点嵌入，则节点嵌入将记录训练链接的链接存在信息（因为Ep ⊆ E）。我们观察到GNN可以快速找出这种链接存在信息并仅通过拟合这部分信息来优化。这在我们的实验中导致了不良的泛化性能。我们的技巧是暂时将En加入E，并在G' = (V, E ∪ En)上生成嵌入。这样，正负训练链接将在嵌入中记录相同的链接存在信息，因此GNN不能仅通过拟合这部分信息来分类链接。我们经验性地验证了这一技巧对SEAL的大幅性能提升。我们称这一技巧为负注入。

我们将我们提出的框架命名为SEAL（从子图、嵌入和属性学习用于链接预测），强调其从三种类型特征联合学习的能力。

## 5 实验结果

我们进行了广泛的实验来评估SEAL。我们的结果表明，SEAL是一个出色且稳健的链接预测框架，在各种网络上取得了前所未有的强大性能。我们使用AUC和平均精度（AP）作为评估指标。我们运行所有实验10次，并报告平均AUC结果和标准偏差。我们将AP和时间结果放在附录F中。SEAL在使用什么GNN或节点嵌入方面是灵活的。因此，我们选择最近的架构DGCNN[17]作为默认的GNN，以及node2vec[20]作为默认的嵌入。代码和数据可在https://github.com/muhanzhang/SEAL获取。

**数据集** 使用的八个数据集是：USAir、NS、PB、Yeast、C.ele、Power、Router和E.coli（详见附录C）。我们从每个数据集中随机移除10%的现有链接作为正测试数据。遵循基于学习的链接预测的标准方式，我们随机抽样相同数量的不存在链接（未连接的节点对）作为负测试数据。我们使用剩余90%的现有链接以及相同数量的额外采样的不存在链接来构建训练数据。

**与启发式方法的比较** 首先，我们将SEAL与仅使用图结构特征的方法进行比较。我们包括八种流行的启发式方法（在附录A的表3中显示）：共同邻居（CN）、Jaccard、优先连接（PA）、Adamic-Adar（AA）、资源分配（RA）、Katz、PageRank（PR）和SimRank（SR）。我们还包括集成（ENS），它在八个启发式分数上训练逻辑回归分类器。我们还包括两种启发式学习方法：Weisfeiler-Lehman图核（WLK）[34]和WLNM[12]，它们也从（截断的）封闭子图中学习。我们省略了路径排名方法[28]以及其他最近专为知识图谱或推荐系统设计的方法[23, 35]。由于所有基线只使用图结构特征，我们限制SEAL不包括任何潜在或显式特征。在SEAL中，跳数h是一个重要的超参数。在这里，我们只从{1, 2}中选择h，因为一方面我们经验性地验证了性能通常在h ≥ 3后不会增加，这验证了我们的理论结果，即最有用的信息位于局部结构中。另一方面，甚至h = 3有时也会因为包含一个集线器节点而导致非常大的子图。这引发了子图中采样节点的想法，我们将其留给未来的工作。选择原则非常简单：如果二阶启发式方法AA在10%验证数据上优于一阶启发式方法CN，则我们选择h = 2；否则我们选择h = 1。对于数据集PB和E.coli，我们一致使用h = 1以适应内存。我们在附录D中包括更多关于基线和超参数的详细信息。

表1展示了结果。首先，我们观察到从封闭子图学习的方法（WLK、WLNM和SEAL）通常比预定义的启发式方法表现得更好。这表明学习的"启发式方法"比手动设计的启发式方法更能捕捉网络特性。在基于学习的方法中，SEAL表现最佳，展示了GNN相对于图核和全连接神经网络的卓越图特征学习能力。从Power和Router的结果中，我们可以看到，尽管现有启发式方法的表现与随机猜测相似，基于学习的方法仍然保持高性能。这表明我们甚至可以为没有现有启发式方法有效的网络发现新的"启发式方法"。

**与潜在特征方法的比较** 接下来，我们将SEAL与六种最先进的潜在特征方法进行比较：矩阵分解（MF）、随机块模型（SBM）[18]、node2vec（N2V）[20]、LINE[21]、谱聚类（SPC）和变分图自编码器（VGAE）[36]。其中，VGAE也使用GNN。请注意VGAE和SEAL之间的区别：VGAE使用节点级GNN学习最能重建网络的节点嵌入，而SEAL使用图级GNN来分类封闭子图。因此，VGAE仍然属于潜在特征方法。对于SEAL，我们额外在节点信息矩阵X中包括128维的node2vec嵌入。由于数据集没有节点属性，所以不包括显式特征。

表2展示了结果。正如我们所见，SEAL相对于潜在特征方法显示出显著的改进。一个原因是SEAL同时从图结构和潜在特征中学习，从而增强了那些仅使用潜在特征的方法。我们观察到，使用node2vec嵌入的SEAL大幅优于纯node2vec。这意味着网络嵌入本身可能无法捕捉位于局部结构中的最有用的链接预测信息。有趣的是，与不使用node2vec嵌入的SEAL（表1）相比，联合学习并不总是提高性能。更多的实验和讨论包含在附录F中。

## 6 结论

自动学习链接预测启发式方法是一个新领域。在本文中，我们提出了从局部封闭子图学习的理论证明。特别地，我们提出了一个γ衰减理论，统一了广泛的高阶启发式方法，并证明了它们可以从局部子图中得到良好的近似。受理论启发，我们提出了一个新型链接预测框架SEAL，基于图神经网络从局部封闭子图、嵌入和属性中同时学习。实验表明，SEAL通过与各种启发式方法、潜在特征方法和网络嵌入算法的比较，实现了前所未有的强大性能。我们希望SEAL不仅能够启发链接预测研究，还能为其他关系机器学习问题，如知识图谱补全和推荐系统开辟新方向。

## 致谢

这项工作部分由国家科学基金会的III-1526012和SCH-1622678拨款以及国家健康研究所的1R21HS024581拨款支持。

## 参考文献

[参考文献列表略]