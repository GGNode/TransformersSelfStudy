# Transformer 从零实现学习项目

本项目旨在通过从零开始手写一个完整的 Transformer 模型，来深入理解其内部工作原理，并提升 Python 和 PyTorch 的编程能力。

## 项目目标

实现一个基于 PyTorch 的 "玩具级" Transformer 模型，用于完成一个简单的机器翻译任务（例如：英译中）。

## 目录结构

```
.
├── data/                # 存放原始数据、处理后的数据和词汇表
├── notebooks/           # 用于探索性编程和可视化的 Jupyter Notebooks
├── README.md            # 项目说明
├── src/                 # 存放核心源代码
│   ├── transformer/     # Transformer 模型的核心模块
│   │   ├── __init__.py
│   │   ├── attention.py       # 注意力机制模块
│   │   ├── encoder.py         # 编码器模块
│   │   ├── decoder.py         # 解码器模块
│   │   ├── feed_forward.py    # 前馈网络模块
│   │   ├── model.py           # 完整的 Transformer 模型组装
│   │   └── utils.py           # 工具函数，如位置编码、masking等
│   ├── data_loader.py   # 数据加载和预处理
│   ├── train.py         # 模型训练脚本
│   └── translate.py     # 使用训练好的模型进行翻译的脚本
└── tests/               # 存放单元测试
    ├── test_attention.py
    └── ...              # 其他模块的测试
```

## 学习与实现步骤

我们建议按照以下顺序，分步实现和学习。每完成一个模块，最好在 `tests/` 目录下为其编写简单的单元测试，以确保其正确性。

### 第 0 步：环境准备

1.  **安装 Python**: 确保您已安装 Python 3.8 或更高版本。
2.  **安装 PyTorch**: 这是我们唯一的核心依赖。访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适合您系统的安装命令。
3.  **安装其他工具**:
    ```bash
    pip install numpy jupyterlab
    ```

### 第 1 步：理解并实现核心模块 (`src/transformer/`)

这是整个项目的核心。您需要逐一理解并实现以下组件：

1.  **位置编码 (Positional Encoding)** - `utils.py`
    *   **任务**: Transformer 无法像 RNN 那样捕捉序列的顺序信息，因此需要明确地将位置信息注入到输入中。请实现 `PositionalEncoding` 类。
    *   **关键点**: 理解为何使用 `sin` 和 `cos` 函数，以及它们如何为每个位置生成独特的编码。

2.  **自注意力机制 (Self-Attention)** - `attention.py`
    *   **任务**: 实现 `MultiHeadAttention` (多头注意力) 模块。这是 Transformer 最核心的创新。
    *   **关键点**:
        *   理解 Q (Query), K (Key), V (Value) 的概念。
        *   实现 Scaled Dot-Product Attention (缩放点积注意力)。
        *   理解为何需要 "Scaled" (缩放)。
        *   将多个注意力头的结果拼接起来，实现 "Multi-Head"。

3.  **Masking (掩码)** - `utils.py`
    *   **任务**: 实现两种掩码：
        *   **Padding Mask**: 忽略输入序列中 "padding" 的部分。
        *   **Sequence Mask (Look-ahead Mask)**: 在解码器中使用，防止在预测当前词时 "看到" 未来的词。
    *   **关键点**: 理解这两种 mask 在模型中的应用位置和原因。

4.  **前馈神经网络 (Feed-Forward Network)** - `feed_forward.py`
    *   **任务**: 实现 `PositionwiseFeedForward` 模块。它在每个注意力层之后被应用。
    *   **关键点**: 这就是一个简单的两层全连接网络，但它是在序列的“每个位置”上独立应用的。

5.  **编码器层 (Encoder Layer)** - `encoder.py`
    *   **任务**: 将 "多头注意力" 和 "前馈网络" 组装起来，构成一个 `EncoderLayer`。
    *   **关键点**: 注意残差连接 (Residual Connection) 和层归一化 (Layer Normalization) 的应用。这是稳定训练的关键。

6.  **解码器层 (Decoder Layer)** - `decoder.py`
    *   **任务**: 组装成一个 `DecoderLayer`。
    *   **关键点**:
        *   解码器有两个注意力层：一个是对解码器输入的 "Masked Multi-Head Attention"，另一个是 "Encoder-Decoder Attention"（Q 来自解码器，K 和 V 来自编码器的输出）。
        *   同样，注意残差连接和层归一化。

### 第 2 步：组装完整的 Transformer 模型 (`src/transformer/model.py`)

1.  **任务**:
    *   将多个 `EncoderLayer` 堆叠成一个完整的 `Encoder`。
    *   将多个 `DecoderLayer` 堆叠成一个完整的 `Decoder`。
    *   将 `Encoder` 和 `Decoder` 组合，再加上输入嵌入 (Input Embedding)、位置编码和最后的输出线性层，构成完整的 `Transformer` 模型。

### 第 3 步：数据处理和训练

1.  **数据加载** - `data_loader.py`
    *   **任务**: 编写一个 `DataLoader` 来读取简单的翻译语料（可以从网上找一些小的英中平行语料库），并进行以下处理：
        *   构建词汇表 (Vocabulary)。
        *   将文本句子转换成数字 ID 序列 (Tokenization)。
        *   添加特殊符号，如 `<s>` (start), `</s>` (end), `<pad>` (padding)。
        *   创建 PyTorch 的 `Dataset` 和 `DataLoader`。

2.  **训练循环** - `train.py`
    *   **任务**: 编写训练脚本。
    *   **关键点**:
        *   实例化模型、损失函数 (Cross-Entropy Loss) 和优化器 (Adam)。
        *   编写标准的 PyTorch 训练循环：前向传播、计算损失、反向传播、更新权重。
        *   **重要**: 在将目标序列喂给解码器时，需要向右移动一位 (shifted right)，并在开头添加 `<s>` 标志，以用于 "teacher forcing"。

### 第 4 步：推理

1.  **翻译脚本** - `translate.py`
    *   **任务**: 编写一个函数，输入一个英文句子，输出翻译后的中文句子。
    *   **关键点**:
        *   这是一个自回归 (auto-regressive) 的过程。
        *   首先，将英文句子喂给编码器。
        *   然后，解码器从 `<s>` 开始，一次生成一个词。
        *   将上一步生成的词，拼接到解码器的输入中，再预测下一个词，直到生成 `</s>` 或达到最大长度。

## 项目管理与工程实践指南

这个部分将指导您如何像一个专业工程师一样管理这个项目。这不仅关乎代码，更关乎习惯。

### 1. 版本控制 (Git)

我们已经为您初始化了 Git 仓库。请遵循以下工作流：

*   **主分支 (`main`)**: 这是您的“生产”分支，只存放可以稳定运行、经过测试的代码。
*   **特性分支 (`feature/`)**: 每当您开始一个新任务（例如，实现“注意力机制”），都应该从 `main` 分支创建一个新的特性分支。例如：
    ```bash
    # 切换到主分支并确保它是最新的
    git checkout main
    # git pull origin main # (如果是远程仓库)

    # 为新功能创建分支
    git checkout -b feature/implement-attention
    ```
*   **提交 (Commit)**: 当您完成了一个小部分的工作（例如，`attention.py` 的一个函数写完了），就进行一次提交。写下清晰的提交信息。
    ```bash
    # 将更改添加到暂存区
    git add src/transformer/attention.py tests/test_attention.py

    # 提交更改
    git commit -m "feat(attention): Implement scaled dot-product attention"
    ```
    *   **提交信息规范**: 推荐使用 `类型(范围): 描述` 的格式，例如 `feat(attention): ...`, `fix(encoder): ...`, `docs(readme): ...`。这会让您的提交历史非常清晰。
*   **合并 (Merge)**: 当一个特性（例如“注意力机制”）完全开发和测试完毕后，就将其合并回 `main` 分支。
    ```bash
    # 切换回主分支
    git checkout main

    # 将特性分支合并进来
    git merge feature/implement-attention
    ```

### 2. 依赖管理

我们创建了 `requirements.txt` 文件。当您需要安装所有依赖时，只需运行：

```bash
# 建议在虚拟环境中执行
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 3. 测试驱动开发 (TDD) - 简化版

对于每个核心模块，我们都提倡“先写测试（或至少是测试的框架），再写实现”。

*   **流程**: 当您要实现 `attention.py` 时：
    1.  打开 `tests/test_attention.py`。
    2.  构思一下，一个正确的注意力模块应该具备什么行为？例如，输入一个特定维度的张量，输出的张量维度应该是什么？
    3.  编写一个测试函数，用虚拟数据（例如 `torch.randn(...)`）来断言（assert）这个行为。
    4.  运行测试，此时它应该是失败的。
    5.  现在，去 `src/transformer/attention.py` 中编写代码，直到测试通过。
*   **好处**: 这会迫使您在写代码之前，就想清楚“代码要达成什么目标”，极大地提升代码质量和开发效率。

### 4. 代码风格与质量

*   **Linter**: 强烈建议安装并使用代码检查工具，如 `ruff` 或 `flake8`。它们可以自动检查出代码中的风格问题和潜在错误。
    ```bash
    pip install ruff

    # 在项目根目录运行检查
    ruff check .
    ```
*   **Formatter**: 使用 `black` 或 `yapf` 自动格式化您的代码，保持风格一致。
    ```bash
    pip install black

    # 格式化整个 src 目录
    black src/
    ```

将这些工具集成到您的工作流中，是迈向专业开发的关键一步。

---

## 如何使用这个项目

1.  从 **第 0 步** 开始，搭建好您的开发环境。
2.  打开 `notebooks/` 目录，创建一个新的 notebook，例如 `01_Attention_Mechanism.ipynb`。
3.  在 notebook 中，尝试实现 `attention.py` 中的 `MultiHeadAttention` 模块。您可以一边查阅资料，一边编写代码，并用一些虚拟的数据来测试它的输入和输出维度是否正确。
4.  当您觉得模块实现得差不多了，就将代码整理到 `src/transformer/attention.py` 文件中。
5.  (可选，但强烈推荐) 在 `tests/test_attention.py` 中编写一个简单的测试函数，确保您的实现是正确的。
6.  按照 **学习步骤**，依次完成其他模块。
7.  祝您学习愉快！
