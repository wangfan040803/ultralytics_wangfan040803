# 在 Ultralytics 中添加“库里没有的”模块/模型：零基础教程

适用场景：你想在 Ultralytics YOLO 的 YAML 里直接写一个新模块名（例如 `DinoV2Patches`），然后框架在构建模型时能自动识别它、实例化它，并把它拼到整个网络里。

这份教程会用你现在做的 `DinoV2Patches` 作为例子，解释“为什么这样做就能用”，以及完整的标准操作步骤。

---

## 一句话原理

Ultralytics 构建模型的过程是：

1. 读取模型 YAML（里面的模块名是字符串，比如 `Conv`、`C2f`、`DinoV2Patches`）
2. 在 `parse_model()` 里把这个字符串“解析成真正的 Python 类”
3. 调用这个类（就像 `m = DinoV2Patches(...)`）得到 PyTorch 模块
4. 把模块按顺序组装成 `nn.Sequential` / 复杂结构，最终得到整个模型

所以你要做的事情就两件：

- **让 Python 能 import 到你的类**
- **让 `parse_model()` 能从字符串找到这个类**

---

## 目录

- A. 最小可用的接入方式（推荐）
- B. “对外导出”方式（可选，但更规范）
- C. 让 YAML 真能解析到：`tasks.py` 注册点
- D. 写 YAML 的示例
- E. 常见坑（尤其是权重/依赖）

---

## A. 最小可用的接入方式（推荐）

目标：只要做到这一步，你就能在 `tasks.py` 里 import，并且 YAML 可以解析到。

### 1）把模块文件放进包路径

把你的文件放到：

- `ultralytics/nn/modules/pretrained_vit.py`

并确保里面定义了你要用的模块类：

```python
import torch
import torch.nn as nn

class DinoV2Patches(nn.Module):
    def __init__(self, in_chanels=3, out_channels=768, size="base"):
        super().__init__()
        ...

    def forward(self, x):
        ...
        return x
```

只要路径对 + 类名对，Python 就能 `import`。

---

## B. “对外导出”方式（可选，但更规范）

这个步骤不是 YAML 解析的必要条件，但能让你用更舒服的导入方式：

- 允许 `from ultralytics.nn.modules import DinoV2Patches`

### 2）在 `ultralytics/nn/modules/__init__.py` 导入它

在：

- `ultralytics/nn/modules/__init__.py`

加入：

```python
from .pretrained_vit import DinoV2Patches
```

### 3）在 `__all__` 里注册（可选但推荐）

仍然在 `ultralytics/nn/modules/__init__.py` 里，`__all__` 增加：

```python
"DinoV2Patches",
```

`__all__` 的意义：主要影响 `from ultralytics.nn.modules import *` 这种写法，以及“这个包官方对外承诺提供哪些名字”。

---

## C. 让 YAML 真能解析到：`tasks.py` 注册点（关键）

Ultralytics 的 YAML 里写的模块名（例如 `DinoV2Patches`）本质上是**字符串**。框架必须把字符串变成真正的类。

在你当前版本里，这个“字符串 → 类”的逻辑在：

- `ultralytics/nn/tasks.py` 的 `parse_model()`

### 4）确保 `tasks.py` 顶部能 import 到你的类

在 `ultralytics/nn/tasks.py` 顶部加入：

```python
from ultralytics.nn.modules.pretrained_vit import DinoV2Patches
```

为什么这一步重要？

- `parse_model()` 里会用 `globals()[m]` 这种方式找模块
- `globals()` 返回的是“当前这个文件（tasks.py）里所有全局变量/类/函数的字典”
- 你把 `DinoV2Patches` import 进来后，它就成了 `tasks.py` 的一个全局名字
- YAML 里写 `DinoV2Patches` 时，解析器就能通过 `globals()["DinoV2Patches"]` 找到它

### 5）把它加入 `base_modules`（通常需要）

在 `parse_model()` 里会有类似：

```python
base_modules = frozenset({ Conv, C2f, ..., DinoV2Patches })
```

加入 `base_modules` 的意义：

- 告诉解析器“这个模块属于需要通道/宽度倍率等逻辑处理的基础模块集合”
- 这样 `parse_model()` 才会用它的规则去推导/调整参数（比如输入通道 `c1`、输出通道 `c2`）

### 6）是否需要加入 `repeat_modules`？

只有当你的模块 YAML 写法需要类似 `[-1, n, YourModule, [...]]` 并且 `n>1` 表示“重复堆叠”时，才需要把它加入 `repeat_modules`。

大多数“自定义 backbone 模块”并不需要 repeat。

---

## D. 在 YAML 里怎么写（示例）

假设你想把 `DinoV2Patches` 当做 backbone 的第一层（举例）：

```yaml
backbone:
  - [ -1, 1, DinoV2Patches, [768, "base"] ]
```

注意：`args` 会按 `parse_model()` 的规则被传给构造函数。

你现在的实现：

```python
def __init__(self, in_chanels=3, out_channels=768, size="base")
```

所以你要确认 YAML 里传参的位置和 `parse_model()` 对该模块的“参数重排逻辑”是匹配的。

经验法则：

- 如果你把模块加入了 `base_modules`，`parse_model()` 往往会把 `args` 改成 `[c1, c2, ...]` 这种形式
- 这意味着你的 `__init__` 最好接收 `(c1, c2, ...)` 作为前两个参数

如果你不想适配这套通道推导规则，也可以：

- 不把它加入 `base_modules`
- 让它走“普通模块”路径（但你需要确保 YAML 传入的参数完全能满足构造函数）

在你当前代码里，你已经选择了“加入 `base_modules`”这条路。

---

## E. 常见坑（非常常见）

### 1）在 `__init__()` 里立刻下载/加载大权重

你的 `DinoV2Patches` 里使用了：

- `torch.hub.load(... pretrained=True / weights=...)`

这会导致一个现象：

- **只要你一构建模型（即解析 YAML），就会立刻加载权重**
- 如果权重文件路径不对/没放好/本地 hub 源不存在，就会在“建图阶段”直接报错

建议（按需选择）：

- 把权重路径做成显式参数
- 或者在外部先下载好权重再加载
- 或者提供 `pretrained=False` 的选项，训练时再决定是否加载

### 2）`no_grad()` 会阻断训练

如果你希望这个模块参与训练（需要反向传播），不要在 `forward()` 外层包 `torch.no_grad()`。

你现在的写法更像“固定特征提取器”（冻结 backbone），这没问题，但要明确：

- 这样训练时不会更新该模块参数

### 3）尺寸对齐 / patch size

你的 `transform()` 用的是 16 的倍数裁剪，但注释写 14；此外你也有不同 backbone 的 patch size。

要确保：

- 你的裁剪/reshape 与实际 patch size 一致
- `out_channels` 与 backbone 输出 embedding size 一致（否则 reshape 会报错）

### 4）依赖包：torchvision

你在 `__init__` 里 import 了 `torchvision.transforms`。

- 环境缺 `torchvision` 会直接失败

---

## 快速自检清单（你照这个对一遍就行）

- [ ] `ultralytics/nn/modules/pretrained_vit.py` 能被 Python import
- [ ] `ultralytics/nn/tasks.py` 顶部 import 了 `DinoV2Patches`
- [ ] `parse_model()` 里 `base_modules`（必要时 `repeat_modules`）包含 `DinoV2Patches`
- [ ] YAML 中模块名拼写与类名完全一致
- [ ] YAML 的 `args` 与 `__init__`（以及 parse_model 的参数处理规则）一致
- [ ] 权重文件/torch hub source/torchvision 等依赖在运行环境可用

---

## 你现在这套做法“为什么能行”（对应到代码）

- 放到 `ultralytics/nn/modules/`：Python 能找到模块文件
- 在 `tasks.py` import：让 `DinoV2Patches` 出现在 `tasks.py` 的 `globals()`
- `parse_model()` 用 `globals()[m]`：YAML 字符串模块名能映射到真正的类
- 加入 `base_modules`：解析器会按 Ultralytics 的规则处理通道与参数

---

如果你把你实际在用的模型 YAML 那一行（包含 `DinoV2Patches` 的那行）贴出来，我可以帮你确认：

- 解析后传入 `DinoV2Patches.__init__()` 的参数到底是什么顺序
- 需不需要改成更“Ultralytics 风格”的签名（通常是 `__init__(self, c1, c2, ...)`）
