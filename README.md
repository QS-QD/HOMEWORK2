[README.md](https://github.com/user-attachments/files/33931166/README.6.md)
# 深度学习理论 · 课后作业

> 课程：深度学习基础
> 内容：ResNet 权重梯度推导、Highway Network 与 ResNet 对比、DenseNet 与 ResNet 对比
> 提交格式：Markdown（GitHub 仓库）

---

## 1. 理论推导：ResNet 中权重层梯度的表达式及其与深层梯度的关联

### 1.1 残差块的前向传播

ResNet 的基本残差块（Residual Block）跳过一层或多层权重变换，其前向传播为

$$x_{l+1} = x_l + F(x_l, W_l)$$

其中：

- xₗ 为第 l 个残差块的输入特征；
- F(xₗ, Wₗ) 为残差分支学到的"修正量"，由权重层 Wₗ（通常是卷积 + BN + 激活的堆叠）参数化；
- 恒等路径 xₗ 直接绕过了 F，构成**快捷连接（shortcut / skip connection）**。

### 1.2 权重梯度 ∂L/∂Wₗ 的推导

记损失函数为 L。注意 xₗ 本身不依赖于 Wₗ（它是之前各层计算的结果），由链式法则：

$$\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial x_{l+1}} \cdot \frac{\partial x_{l+1}}{\partial W_l} = \frac{\partial L}{\partial x_{l+1}} \cdot \frac{\partial F(x_l, W_l)}{\partial W_l}$$

即**权重梯度 = 本块输出的深层梯度 × 残差分支对权重的雅可比**。

### 1.3 深层梯度 ∂L/∂xₗ 的推导与关联

先求快捷连接的雅可比。对 xₗ 求导时注意 xₗ 同时出现在恒等项和残差项中：

$$\frac{\partial x_{l+1}}{\partial x_l} = I + \frac{\partial F(x_l, W_l)}{\partial x_l}$$

其中 I 为单位矩阵，来自恒等路径 xₗ 的导数。于是第 l 层的梯度通过链式法则回传：

$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} \cdot \frac{\partial x_{l+1}}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} \left( I + \frac{\partial F(x_l, W_l)}{\partial x_l} \right)$$

从第 l 层一路展开到第 L 层，得到连乘形式

$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L} \prod_{i=l}^{L-1} \left( I + \frac{\partial F(x_i, W_i)}{\partial x_i} \right)$$

**与深层梯度的关联（这正是残差结构的核心价值）：**

- 梯度 ∂L/∂xₗ 与深层梯度 ∂L/∂x_{l+1} 之间只差一个"单位矩阵 + 残差雅可比"的因子；
- 即使每个残差分支的雅可比 ∂F/∂xᵢ 很小，连乘式中始终**保底含有一个单位矩阵 I**（因为 ∏(I + 小量) 展开后必含 I 项），梯度几乎以无损方式沿恒等路径直达浅层，不会消失；
- 作为对照，普通网络是连乘 ∂F/∂xᵢ，各因子都小于 1 时梯度指数衰减（梯度消失），都大于 1 时指数爆炸（梯度爆炸）；
- 训练初期残差分支尚未学好，∂F/∂xᵢ 很小，此时近似有

$$\frac{\partial L}{\partial x_l} \approx \frac{\partial L}{\partial x_{l+1}}$$

即**深层梯度几乎可以原样回传到任意浅层**——这就是为什么上千层的 ResNet 仍能稳定训练，以及为什么残差块常被理解为"让网络只需要学习对恒等映射的修正"。

---

## 2. 思考与对比

### 2.1 Highway Network 与 ResNet 在快捷连接实现上的异同

Highway Network（Srivastava 等，2015，几乎与 ResNet 同期）的每一层定义为

$$y = H(x, W_H) \cdot T(x, W_T) + x \cdot C(x, W_C)$$

其中 H 是普通的变换层，T ∈ (0,1) 是**变换门（Transform Gate）**，C 是**携带门（Carry Gate）**，通常取 C = 1 − T。门控都是可学习的、随输入数据动态变化的。

**相同点：**

- 两者都受"LSTM 门控思想"启发，在层与层之间引入"直通通路"，让梯度有一条不被权重矩阵反复缩放的回传路径，从而能训练上百层的深层网络；
- 两者都在 ImageNet 等任务上证明了"以前认为训不动的超深网络"可以训练；
- 数学形式上，当 T = 1、C = 0 时 Highway 层退化为普通层，当 T = 0、C = 1 时退化为纯恒等映射——**ResNet 可看作"门被固定、始终半开"的 Highway 特例**。

**不同点：**

| 维度 | Highway Network | ResNet |
|---|---|---|
| 快捷连接形式 | 门控加权：y = H·T + x·C | 纯加法：y = x + F(x) |
| 是否有额外参数 | 有（两个门控需额外学习） | 无（恒等路径零参数） |
| 通路是否始终开放 | 否，门可学习着关闭通路 | 是，恒等路径永远存在 |
| 恒等映射是否被"稀释" | 可能，x 被门 C 缩放 | 不会，x 原样保留 |
| 结构复杂度 | 每层多两个门控分支 | 仅需一个加法 |

**简评：** Highway 的门控更灵活（网络可自行决定每层"改多少、留多少"），但灵活性是有代价的——额外参数、门饱和风险、实现更复杂。ResNet 用"无条件的恒等捷径"换来了极简与稳定，实践证明这种"不加门"的简化反而效果更好、更易推广，因此成为后续深度网络的主流构件。

### 2.2 DenseNet 与 ResNet 连接方式的本质不同及其优缺点

ResNet 的连接是**加法式（additive）**：第 l+1 层直接在上层输出上叠加一个修正量，特征图尺寸不变，信息以"扰动叠加"的方式传递。

DenseNet（Huang 等，2017）的连接是**拼接式（concatenative）**：每一层都接收之前**所有层**的输出在通道维上拼接作为输入

$$x_{l+1} = H_l \left( [\, x_0,\ x_1,\ \dots,\ x_l \,] \right)$$

其中 [·] 表示通道拼接，Hₗ 是第 l 层的变换（Conv–BN–ReLU，常为 Bottleneck 结构）。每个 Dense Block 内，第 l 层向之后的所有层引出连接，一个 L 层的 Block 共有 L(L+1)/2 条连接。

**本质不同一句话概括：** ResNet 让特征通过"相加"被逐层修正，DenseNet 让特征通过"拼接"被逐层复用——前者每层只看得见"上一层的结果"，后者每层都能直接利用"之前所有层的原始输出"。

**优点：**

- **特征复用（Feature Reuse）**：浅层提取的边缘、纹理等基础特征可被深层直接访问，不必反复重新学习；
- **参数效率极高**：由于能复用已有特征，每层只需学习少量新特征（DenseNet-121 以约 800 万参数即在 ImageNet 上取得与 ResNet-50 相当的表现）；
- **梯度流通畅**：每个浅层都有一条到损失的无衰减通路（类似隐式的深度监督），深层网络更容易优化；
- **正则化效应强**：密集的短连接降低了过拟合倾向，在小数据集（CIFAR、SVHN）上表现尤其突出。

**缺点：**

- **显存占用大**：反向传播需要保存 Block 内所有层的中间特征图以备拼接回传，训练显存显著高于同深度 ResNet；
- **通道数膨胀**：第 l 层输入通道为 k₀ + k(l−1)（k 为增长率），深层输入非常宽，必须配合 1×1 Bottleneck 卷积控制计算量；
- **计算与工程效率**：大量小卷积核与频繁的内存读写使实际训练速度偏慢，在工业界大规模部署、超大 batch 训练中不及 ResNet 直接；
- **可扩展性**：网络加深时参数与显存增长更快，超深配置的收益递减。

**总结对比表：**

| 维度 | ResNet | DenseNet |
|---|---|---|
| 连接运算 | 加法 x + F(x) | 通道拼接 [x₀, …, xₗ] |
| 每层输入来源 | 仅上一层 | 之前所有层 |
| 特征图尺寸 | 不变 | 通道逐层增长 |
| 参数量 | 较多 | 更少 |
| 训练显存 | 较小 | 较大 |
| 典型优势场景 | 大规模训练、工业部署 | 中小数据集、参数受限场景 |

---

## 参考文献

1. He, K., et al. (2015). *Deep Residual Learning for Image Recognition*. CVPR.（ResNet 与残差连接）
2. Srivastava, R. K., et al. (2015). *Highway Networks*. ICML Workshop / JMLR.（门控快捷连接）
3. Huang, G., et al. (2017). *Densely Connected Convolutional Networks*. CVPR.（DenseNet）
4. Goodfellow, I., et al. *Deep Learning*, Chapter 8.（梯度消失/爆炸与优化）
