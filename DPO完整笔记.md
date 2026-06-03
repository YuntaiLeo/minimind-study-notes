# DPO 完整笔记（原理 · 推导 · 直觉 · 变体）

> minimind pipeline 学习笔记 · 对齐 (Alignment) 阶段
> 配套：Pretrain → SFT → **DPO** → GRPO → Agentic RL → 评测/部署

---

## 第一部分：DPO 是什么

### 1. DPO 在解决什么问题

SFT 之后，模型**学会了对话格式和指令跟随**，但它只是在模仿训练集里的"标准答案"，并不知道**人更喜欢哪种回答**。同一个问题两个回答都通顺，但一个更有帮助、更礼貌、更安全——SFT 没法表达这种"偏好"。

**对齐(alignment)** 这一步就是把"人类偏好"注入模型。数据形式是**成对偏好**：同一个 prompt $x$ ，给一个**更好的回答 $y_w$ (chosen/win)** 和一个**更差的回答 $y_l$ (rejected/lose)**。minimind 里就是 `dpo.jsonl`。

### 2. 经典做法：RLHF + PPO（DPO 取代了什么）

OpenAI 那套经典 RLHF 分两步：

1. **训练奖励模型(RM)**：用偏好对训一个打分模型 $r(x,y)$ ，让 $r(x,y_w) > r(x,y_l)$ 。
2. **用 PPO 做强化学习**：让策略模型生成回答，RM 打分当奖励，用 PPO 最大化奖励，同时加 **KL 惩罚** 拉住它别偏离 SFT 模型太远。

目标函数：

$$ \max_{\pi}\ \mathbb{E}\big[\,r(x,y)\,\big] \;-\; \beta \cdot \mathrm{KL}\big(\pi(\cdot \mid x)\,\|\,\pi_{\text{ref}}(\cdot \mid x)\big) $$

**痛点**：要额外训 RM、训练时要在线采样、PPO 超参多又不稳、显存要同时塞 4 个模型(策略/参考/RM/Critic)。又贵又难调。

### 3. DPO 的核心思想：把 RL 问题变回"分类"问题

DPO(Direct Preference Optimization) 的洞见：上面那个带 KL 约束的 RL 目标**有闭式最优解**，这个解能让你把奖励 $r$ 用策略本身表达出来——于是**奖励模型和 PPO 循环全都可以扔掉**，直接在偏好对上做一个类似分类的监督训练。

> 一句话：**DPO = 不训奖励模型、不做在线采样，直接用一个简单 loss 拉开 chosen 和 rejected 的概率差。**

### 4. DPO 的 Loss（逐项拆解）

$$ \mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,\,y_w,\,y_l)}\Big[\,\log \sigma\big(\beta\,(s_w - s_l)\big)\,\Big] $$

其中：

$$ s_w = \log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x) $$

$$ s_l = \log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x) $$

$\sigma$ 为 sigmoid， $\beta$ 为温度系数（控制偏离参考模型的力度，minimind 里一般取 $0.1$ ）。

**关于 $\mathbb{E}$ （数学期望）**：它就是期望算子。下标 $(x, y_w, y_l)$ 表示"对谁求期望"——对偏好数据集 $\mathcal{D}$ （即 `dpo.jsonl`）中抽出的三元组（prompt、chosen、rejected）求期望，写全是 $\mathbb{E}_{(x,\,y_w,\,y_l)\sim \mathcal{D}}[\,\cdot\,]$ 。实践中没有真实分布、只有有限样本，所以用 **batch 内的样本均值**近似它（蒙特卡洛估计）：

$$ \mathcal{L}_{\text{DPO}} \approx -\frac{1}{N}\sum_{i=1}^{N} \log\sigma\big(\beta\,(s_w^{(i)} - s_l^{(i)})\big) $$

其中 $N$ 是 batch 内样本数， $i$ 是第几条。代码里 `train_dpo.py` 的那个 `.mean()` 就是这个 $\mathbb{E}$ 的工程实现。

> 通用约定：几乎所有深度学习的 loss 写成 $\mathbb{E}[\cdots]$ 都是这个意思——**理论上是对数据分布求期望，实现上是对 mini-batch 求平均**。不止 DPO。

逐项理解：

- **$\pi_\theta$**：正在训练的策略模型，从 `full_sft` 权重初始化。
- **$\pi_{\text{ref}}$**：参考模型，**冻结的 SFT 模型副本**（训练全程不更新）。
- **$s_w$**：chosen 在"新模型 vs 老模型"下的对数概率变化，希望它**变大**。
- **$s_l$**：rejected 的同样量，希望它**变小**。
- **$\beta(s_w - s_l)$**：本质是 chosen 相对 rejected 的"隐式奖励差"。套上 $-\log \sigma(\cdot)$ 就是一个**让 chosen 隐式奖励高于 rejected 的二分类交叉熵**。

> 把 $\hat{r}(x,y) = \beta\,\big(\log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x)\big)$ 看作"**隐式奖励**"，DPO loss 就是经典 Bradley-Terry 偏好模型 $P(y_w \succ y_l) = \sigma(\hat{r}_w - \hat{r}_l)$ 的最大似然——只不过奖励不再来自单独的 RM，而是**策略模型自己算出来的**。

> 这个 loss 不是拍脑袋来的——它能从第二部分的 RLHF 目标**严格推导**出来。

---

## 第二部分：为什么这样成立（完整推导）

> 目标：讲透两个最容易卡住的点 —— ① 为什么能"绕过奖励模型" ② 为什么配分函数 $Z(x)$ 能消掉。

### 先抓住一句话（全篇的灵魂）

> **奖励 $r$ 和"最优策略 $\pi^{\ast}$"是一一对应的——知道其中一个，就等价于知道另一个。**

经典 RLHF 是：先训出奖励 $r$ ，再用 RL 求出对应的最优策略 $\pi$ 。
DPO 的洞见是：既然 $r$ 和 $\pi^{\ast}$ 一一对应，那我**直接去求 $\pi^{\ast}$ 就行了，根本不用先经过 $r$ 这个中间商**。

下面四步就是把这句话变成公式。

### Step 1：带 KL 约束的目标，最优策略长什么样

RLHF 的目标（对某个固定的 $x$ ）：

$$ \max_{\pi}\ \mathbb{E}_{y\sim\pi}\big[\,r(x,y)\,\big] \;-\; \beta\cdot\mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big) $$

前一项 $\mathbb{E}[r]$ 要**奖励高**；后一项（KL 惩罚）要求**别离 SFT 模型 $\pi_{\text{ref}}$ 太远**。

这个"奖励最大化 + KL 拉住"的优化问题，有**已知的闭式解**（可用拉格朗日乘子推，结论记住即可）：

$$ \pi^{\ast}(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\Big(\frac{r(x,y)}{\beta}\Big) $$

**直觉**：最优策略 = 在 SFT 模型 $\pi_{\text{ref}}$ 的基础上，**按奖励高低做重新加权**——奖励高的回答 $\exp(r/\beta)$ 这个因子大、概率被放大；奖励低的被压低。

$Z(x)$ 是**配分函数**，就是个归一化分母：

$$ Z(x) = \sum_{y} \pi_{\text{ref}}(y\mid x)\,\exp\!\Big(\frac{r(x,y)}{\beta}\Big) $$

它的作用只是保证 $\pi^{\ast}(y\mid x)$ 加起来等于 $1$ （是个合法概率分布）。**注意：它要对"所有可能的回答 $y$"求和，根本算不出来**——这是 RLHF 难算的根源之一。先记住：**$Z(x)$ 只跟 $x$ 有关，跟具体回答 $y$ 无关。**

### Step 2：把式子反过来，用策略表达奖励

上面那个 $\pi^{\ast}$ 的式子，两边取 log：

$$ \log \pi^{\ast}(y\mid x) = \log \pi_{\text{ref}}(y\mid x) + \frac{r(x,y)}{\beta} - \log Z(x) $$

移项解出 $r$ ：

$$ r(x,y) = \beta\,\log\frac{\pi^{\ast}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x) $$

**这一步就是"灵魂那句话"的数学形式**：奖励 $r$ 完全可以用"最优策略 $\pi^{\ast}$ 与参考策略 $\pi_{\text{ref}}$ 的对数比"表达出来。唯一碍眼的就是尾巴上那个 $\beta\log Z(x)$ （算不出来的那一项）。

### Step 3：代进偏好模型， $Z(x)$ 自动消失 ✨

人类偏好用 **Bradley-Terry 模型**描述—— " $y_w$ 比 $y_l$ 更受偏好"的概率，**只取决于两者奖励之差**：

$$ P(y_w \succ y_l) = \sigma\big(r(x,y_w) - r(x,y_l)\big) $$

关键来了：把 Step 2 的 $r$ 代进这个**差**里：

$$
\begin{aligned}
r(x,y_w) - r(x,y_l)
&= \Big[\beta\log\tfrac{\pi^{\ast}(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} + \beta\log Z(x)\Big] \\
&\quad - \Big[\beta\log\tfrac{\pi^{\ast}(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)} + \beta\log Z(x)\Big] \\
&= \beta\log\frac{\pi^{\ast}(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi^{\ast}(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\end{aligned}
$$

**为什么能消？** 因为 $Z(x)$ 只跟 $x$ 有关。 $y_w$ 和 $y_l$ 是**同一个问题 $x$** 的两个回答，所以它俩的 $\beta\log Z(x)$ 是**完全相同的同一个数**，一减就抵消了。

> 🎯 **类比**： $Z(x)$ 像是"这道题所有人统一加的分"。你要比较甲乙两个学生在**同一张卷子**上谁更好，这个统一加的分对两人都一样，做差时自然消掉——它根本不影响"谁更好"的判断。这正是为什么我们能甩掉那个算不出来的 $Z(x)$ 。

消掉后：

$$ P(y_w \succ y_l) = \sigma\Big(\beta\log\frac{\pi^{\ast}(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi^{\ast}(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big) $$

整个式子里**只剩 $\pi^{\ast}$ 和 $\pi_{\text{ref}}$ ，奖励 $r$ 和 $Z(x)$ 都没了**。

### Step 4：最后一跃 —— 把要训的模型 $\pi_\theta$ 当作 $\pi^{\ast}$

上式里的 $\pi^{\ast}$ 是"理论最优策略"，我们不知道它是谁。DPO 的做法：**就用正在训练的模型 $\pi_\theta$ 去充当这个 $\pi^{\ast}$**，然后在偏好数据上做**最大似然**（让模型给"chosen 优于 rejected"这个事实尽量高的概率）。取负对数似然，就是 DPO loss：

$$ \mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,\,y_w,\,y_l)}\Big[\log \sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big] $$

这正是第一部分给出的 loss（把 $s_w,\,s_l$ 的定义代回即得，只是这里写成了展开形式）。

**这就是魔法所在**：当你用这个 loss 把 $\pi_\theta$ 训到能满足所有偏好对时，数学上保证了—— $\pi_\theta$ 就是那个"经典 RLHF 要费尽周折（训 RM + 跑 PPO）才能得到的最优策略 $\pi^{\ast}$"。中间的奖励模型被"折叠"进了策略本身，**一步到位，不用 RL**。

### 一句话收尾

- **经典 RLHF**： 偏好 $\to$ 训奖励模型 $r$ $\to$ 跑 RL 求 $\pi$ （三步，绕）。
- **DPO**：利用" $r$ 和 $\pi^{\ast}$ 一一对应"，把这条链**短路**成 偏好 $\to$ 直接训 $\pi$ （一步）；而"奖励差"这个形式让只依赖 $x$ 的 $Z(x)$ 天然对消，于是那个算不出来的归一化项压根不用管。

---

## 第三部分：如何直观理解 loss 里的"差"

DPO loss 的核心是这个差 $\beta(s_w - s_l)$ 。拆开看就很清楚。

### 第 1 层：单项 $s$ —— "相对 SFT 的概率涨跌"

$$ s = \log \pi_\theta(y\mid x) - \log \pi_{\text{ref}}(y\mid x) = \log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)} $$

它衡量：新模型 $\pi_\theta$ 把这个回答的概率，相对老模型 $\pi_{\text{ref}}$ (SFT) 调高了还是调低了。

- $s > 0$ ：新模型**抬高**了这个回答（比 SFT 更愿意说它）
- $s < 0$ ：新模型**压低**了它
- $s = 0$ ：没动

可以理解成模型对这个回答的**“投票变化量”**。

### 第 2 层：差 $s_w - s_l$ —— "相对 margin"

它衡量：**模型有没有把 chosen 抬得比 rejected 更多。**

- 差 $> 0$ ：好，往“偏向 chosen”动了 → $\sigma(\cdot)\to 1$ → loss 小
- 差 $< 0$ ：坏，反而更偏 rejected → loss 大，被狠狠惩罚

一句话：**DPO 就是把 chosen 概率往上推、rejected 往下压，直到这个差足够正。**

### 为什么要"相对 $\pi_{\text{ref}}$"，不直接比 $\pi_\theta(y_w) > \pi_\theta(y_l)$ ？

因为有些回答**天生概率就高**（短、常见说法、套话），这跟“它好不好”无关。直接比绝对概率会被这种“天然似然”干扰。减去 $\log \pi_{\text{ref}}$ 就是**减掉这个天然 baseline**，只留下“**我们到底主动改了多少**”。

> 类比：不看学生的**裸分**（有人天生底子好），而看他相对自己基线的**进步幅度**——chosen 要比 rejected **进步得更多**。
> 顺带，这个“对 ref 的比”也正是隐式的 KL 锚，防止模型为讨好偏好把整个分布带跑偏。

### 为什么是"差"，不是各自的绝对值？

因为偏好数据本身就是**相对的**——它只告诉你“A 比 B 好”，从没给过 A、B 的绝对分数。既然信息只有“谁 > 谁”，能学的也只有两者的**差距 (gap)**。这跟 Bradley-Terry 一致：偏好概率只由**奖励之差**决定，绝对值不可辨识（整体加减一个常数，排序不变）——和上文 $Z(x)$ 能消掉是同一个道理。

### 梯度直觉（最精髓：自适应用力）

$$ \nabla_\theta \mathcal{L}_{\text{DPO}} \;\propto\; -\,\sigma\big(\beta(s_l - s_w)\big)\,\Big[\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\Big] $$

- 方括号那部分（配合负号与梯度下降）的实际效果：**抬高 chosen、压低 rejected**。
- 前面的权重 $\sigma\big(\beta(s_l - s_w)\big)$ = 模型当前的**“错误程度”**：
  - 还在错误地偏向 rejected（ $s_l > s_w$ ）→ 权重大 → **使劲改**
  - 已经正确且拉开很大（ $s_w \gg s_l$ ）→ 权重趋近 $0$ → **几乎不动**（已满足，别过度优化）

**优雅之处**：DPO 自动把劲使在“还没学对”的样本上，对“早就学对”的松手，无需手动调。

### 小数字例子（ $\beta = 0.1$ ）

| | $\pi_{\text{ref}}$ | $\pi_\theta$ | $s = \log(\pi_\theta/\pi_{\text{ref}})$ |
|---|---|---|---|
| chosen $y_w$ | 0.10 | 0.20 | $\log 2 \approx +0.69$ （概率翻倍） |
| rejected $y_l$ | 0.10 | 0.05 | $\log 0.5 \approx -0.69$ （概率减半） |

$$ s_w - s_l = 0.69 - (-0.69) = 1.38 \quad (>0) $$

差为正，方向对（模型确实更偏向 chosen）。

$$ \beta(s_w - s_l) = 0.1 \times 1.38 = 0.138, \qquad \mathcal{L} = -\log \sigma(0.138) \approx 0.63 $$

反过来若模型错误地抬高 rejected，差变负， $\sigma$ 跌破 $0.5$ ，loss 飙升，梯度把它使劲拽回来。

### 一句话总结

> 那个差 = **chosen 相对 SFT 的概率提升量** 减去 **rejected 的提升量** = “模型偏向 chosen 的净程度”。DPO 在最大化它；因偏好只给相对信息，所以只能、也只需要学这个**差**。

---

## 第四部分：训练与局限（实战视角）

### 训练 `train_dpo.py` 时盯什么

- **loss 下降**：正常该平稳下降。
- **reward margin $= s_w - s_l$ （隐式奖励差）应上升**：说明 chosen 和 rejected 被拉开。
- **reward accuracy**：满足 $\hat{r}_w > \hat{r}_l$ 的样本比例，应 $> 0.5$ 且走高。
- **⚠️ 经典坑**：有时 margin 在涨，但 **chosen 的绝对 log 概率也在跌**（只是 rejected 跌得更快），会让生成质量下降。盯住 chosen logprob 别塌。缓解：调小 lr、调 $\beta$ 、换更干净的偏好数据。

### DPO 的局限

- **离线方法**：只在固定偏好对上学，不采样、不探索，天花板被数据质量锁死。
- **对 $\beta$ 和数据分布敏感**：偏好数据有噪声/分布偏移时易学歪。
- **只优化相对差**：可能压低 chosen 绝对概率。
- 显存上要同时持有 $\pi_\theta$ 和冻结的 $\pi_{\text{ref}}$ （但 ref 不回传，开销可控）。

这些局限催生了两条改进路线：① **变体**（IPO / KTO / SimPO，见第五部分）对其中几条对症下药；② **在线 RL**（GRPO，见第六部分）从根上解决“离线、不探索”的问题。

---

## 第五部分：DPO 的主流变体（IPO / KTO / SimPO）

> 记忆法：**每个变体 = 针对 DPO 的一个毛病开的药。**
>
> DPO 的三个主要毛病：① 容易**过拟合偏好**（loss 无饱和点，把 log 比无限拉大，KL 约束失效）；② 必须**成对数据**；③ 必须挂**参考模型 $\pi_{\text{ref}}$**（占显存 + 长度偏置）。

### IPO —— 治"过拟合"

**Identity Preference Optimization**（DeepMind, 2023）

- **痛点**：偏好确定性时，DPO 的 $-\log \sigma$ 会把 margin 推到无穷大，**等于忽略 KL 正则** → 过拟合。
- **改动**：把 $-\log \sigma(\cdot)$ 换成**平方损失**，让 margin **回归固定目标 $\tfrac{1}{2\beta}$**：

$$ \mathcal{L}_{\text{IPO}} = \Big((s_w - s_l) - \tfrac{1}{2\beta}\Big)^2 $$

- **效果**：margin 有"刹车"，KL 约束重新生效，**抗过拟合更强**。仍需成对数据 + ref。

### KTO —— 治"必须成对"

**Kahneman-Tversky Optimization**（ContextualAI, 2024）

- **痛点**：成对偏好数据贵、难收集。
- **改动**：借鉴**前景理论**（人对得失感受不对称）。KTO **只需每条样本一个二元标签"好/坏"，不需配对**。点赞点踩就能用。
- **机制**：以 ref 模型的 KL 为基准点定义价值函数，对"好/坏"样本用不同权重，**最大化效用**而非偏好似然。
- **效果**：**数据极易获取**、容忍正负样本不均衡，效果常与 DPO 相当甚至更好。仍需 ref。

### SimPO —— 治"要参考模型 + 长度偏置"

**Simple Preference Optimization**（Princeton, 2024）

- **痛点**：ref 模型占资源；DPO 隐式奖励（对 ref 的 log 比）与推理时的"平均对数似然"对不上；偏向长回答。
- **改动**：
  1. **完全去掉 ref 模型**，直接用**长度归一化的平均对数概率**当隐式奖励（与推理生成度量一致）；
  2. 加一个**目标 margin $\gamma$**，要求 chosen 超出 rejected 至少 $\gamma$ ：

$$ \mathcal{L}_{\text{SimPO}} = -\log \sigma\!\Big(\frac{\beta}{|y_w|}\log \pi(y_w \mid x) - \frac{\beta}{|y_l|}\log \pi(y_l \mid x) - \gamma\Big) $$

- **效果**：**无 ref → 更省显存更简单**，长度归一化**缓解长度偏置**，多个 benchmark 超过 DPO。代价：少了 KL 锚，**对 $\beta$ 、 $\gamma$ 更敏感**，调不好可能退化。

> 另有 **ORPO**：更激进，**连 SFT 和偏好对齐合成一步**（无 ref、无独立 SFT），适合想省流程的场景。

### 对照表

| 方法 | 治 DPO 的什么病 | 核心改动 | 要成对? | 要 ref 模型? |
|---|---|---|---|---|
| **DPO** | — | 基准 | ✅ | ✅ |
| **IPO** | 过拟合 | sigmoid→平方损失，margin 回归定值 | ✅ | ✅ |
| **KTO** | 数据要成对 | 只需"好/坏"二元标签 | ❌ | ✅ |
| **SimPO** | 要 ref + 长度偏置 | 去 ref、长度归一化、加目标 margin $\gamma$ | ✅ | ❌ |

### 与项目的连接（方向 A：受控消融）

minimind 模型小、训练几分钟一轮，可在**同一份偏好数据**上分别跑 **DPO vs IPO vs KTO vs SimPO**，出一张"四种对齐方法在 26M 模型上的对照表"——**干净的受控对比**正是作品集里最显功底的东西，大模型上没人舍得这么烧钱做。

---

## 第六部分：DPO 与对比学习（通往 GRPO 的桥）

DPO 本质上**就是一种对比学习**——学界也有方法直接叫 Contrastive Preference Learning (CPL)，SimPO/DPO 常被描述成 contrastive 式目标。对应关系几乎一一对得上：

| 对比学习 | DPO |
|---|---|
| 正样本 positive | chosen $y_w$ |
| 负样本 negative | rejected $y_l$ |
| 拉近正样本、推远负样本 | 抬高 $\pi_\theta(y_w \mid x)$ 、压低 $\pi_\theta(y_l \mid x)$ |
| 用“相似度差”算 loss | 用“隐式奖励差” $s_w - s_l$ 算 loss |
| 只在乎相对（谁更近），不锚绝对值 | 只在乎相对（谁更好），只能学 gap |
| 常见 logistic / softmax 形式 | $-\log\sigma(\cdot)$ 也是 logistic 形式 |

尤其 $-\log\sigma(s_w - s_l)$ ，本质就是一个**成对的二分类 / 排序 loss**——和对比学习里“一正一负”的 pairwise logistic loss 同形。

### 但别完全画等号（4 个关键区别）

1. **作用空间不同**：经典对比学习（SimCLR / InfoNCE）在**表示 / embedding 空间**拉近推远，用向量相似度（点积 / cosine）；DPO 在**生成概率（序列对数似然）**上操作，score 是 $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ 。
2. **正负样本来源不同**：对比学习里正样本常是“同一张图的增广”；DPO 的 chosen/rejected 是**人类偏好标注**。
3. **DPO 多了个参考锚 $\pi_{\text{ref}}$**：要求别偏离 SFT 太远（隐式 KL）。普通对比学习没有这种“别跑偏”约束。
4. **负样本数量**：DPO 是**一正一负（pairwise）**；InfoNCE 通常**一正多负**（softmax over 多个）。

### 桥：从第 4 点通向 GRPO

如果把 DPO 从“一正一负”推广到“**一组里一正多负、做 softmax**”，就更像标准 InfoNCE 了——而这**恰恰是 GRPO 的味道**：对同一 prompt 采样一组答案、组内按奖励相对比较。

> 所以可以这样串：**GRPO ≈ 把 pairwise 偏好对比，升级成「组内多负样本对比 + 在线采样」。** 沿着“对比学习”这条线往下走，正好顺势理解下一站 GRPO。

> **DPO ≈ 在“生成概率空间”里、带参考锚的、pairwise 对比学习。** 把它和对比学习、GRPO 串起来，整条对齐 / RL 线就成体系了。

---

## 面试八股 Q&A

**Q1：DPO 相比 PPO-RLHF 最本质的区别？**
A：DPO 把"训奖励模型 + 在线 RL"两步合一，用闭式解把奖励用策略自身表达，变成**离线、无需采样、无需 RM 的监督式训练**。更稳更省，但失去在线探索。

**Q2：DPO 推导的起点和核心 trick 是什么？**
A：起点是 KL 约束下的 RLHF 目标；核心 trick 是它的闭式最优解给出了"奖励 $\leftrightarrow$ 最优策略"的一一对应（reparametrization），于是可以用策略反表达奖励，把"求奖励再做 RL"短路成"直接拟合策略"。

**Q3：DPO 为什么不需要奖励模型？（"奖励模型被折叠进策略"是什么意思）**
A：由 $r(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)$ ，策略本身就隐式定义了一个奖励—— $\pi_\theta$ 自己就是隐式奖励模型，所以无需再单独训练 RM。这就是"奖励模型被折叠进了策略"。

**Q4：DPO loss 里为什么是 $\pi_\theta$ 而不是 $\pi^{\ast}$ ？**
A： $\pi^{\ast}$ 是未知的理论最优策略。DPO 用可训练的 $\pi_\theta$ 去参数化（充当） $\pi^{\ast}$ ，在偏好数据上做最大似然；训练收敛时 $\pi_\theta$ 即逼近 $\pi^{\ast}$ 。

**Q5：配分函数 $Z(x)$ 为什么能消掉 / 不用计算？**
A： $Z(x)$ 需对所有可能回答求和，不可解；但它只依赖 $x$ 。偏好建模只看同一 $x$ 下两回答的**奖励差**，做差时两个相同的 $\beta\log Z(x)$ 对消，所以最终 loss 既不含、也无需计算 $Z(x)$ 。

**Q6：loss 里 $\beta$ 起什么作用？**
A：控制偏离参考模型的强度（隐式 KL 约束）。 $\beta$ 大→更贴近 SFT、改动保守； $\beta$ 小→偏好信号权重大、改动激进但易跑偏。

**Q7：参考模型 $\pi_{\text{ref}}$ 是什么？为什么冻结？**
A：训练起点 SFT 模型的冻结副本。提供"锚"，通过 log 概率比实现隐式 KL 约束，防止策略坍塌到只会输出 chosen 而丧失通用能力。

**Q8：loss 里为什么用"对 $\pi_{\text{ref}}$ 的对数比"，而不是直接比 $\pi_\theta$ 的绝对概率？**
A：绝对概率含“天然似然”偏置（短句/套话天生概率高），与好坏无关。减去 $\log\pi_{\text{ref}}$ 把这个 baseline 去掉，只留“相对 SFT 主动改了多少”；同时这个对数比就是隐式 KL 锚，防止分布跑偏。

**Q9：DPO 的梯度为什么是"自适应"的？**
A：梯度权重为 $\sigma\big(\beta(s_l - s_w)\big)$ ，即模型当前的错误程度——错得越狠（仍偏向 rejected）权重越大、更新越猛；已学对并拉开则权重趋近 $0$ 、几乎不动，从而把算力集中在难样本上。

**Q10：DPO 的已知失效模式？**
A：只优化相对 margin，可能在拉开差距的同时压低 chosen 绝对似然，导致生成退化；对噪声偏好数据敏感。缓解：调小 lr/调 $\beta$ 、清洗数据，或用变体(IPO/KTO/SimPO)。

**Q11：SimPO 为什么能去掉参考模型？**
A：把隐式奖励从"对 ref 的对数概率比"改成**长度归一化的平均对数似然**，该量自身能比较好坏、且与推理生成度量一致，于是不再需要 ref 当锚；靠目标 margin $\gamma$ 防坍塌。

**Q12：KTO 最大的工程优势？**
A：**不需要成对数据**，只要每条样本一个"好/坏"标签，数据采集成本骤降，还能处理正负样本不均衡。

**Q13：IPO 和 DPO 的根本差别？**
A：损失函数形状——DPO 用 log-sigmoid（可被推向无穷、易过拟合），IPO 用平方损失把 margin 拉回固定目标，**保住了 KL 正则**。
