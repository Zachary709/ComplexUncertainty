# ComplexUncertainty

基于大语言模型的不确定性分析工具，专注于分析模型生成过程中的Hidden States变化和概率分布。

## 项目结构

```
ComplexUncertainty/
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── util/                     # 工具函数模块
│   │   ├── __init__.py
│   │   ├── similarity.py         # 相似度计算
│   │   ├── probability.py        # 概率计算
│   │   └── vector.py             # 向量处理
│   ├── server/                   # 服务器模块
│   │   ├── __init__.py
│   │   └── model_server.py       # 模型API服务
│   └── analysis/                 # 分析模块
│       ├── __init__.py
│       ├── hidden_states.py      # Hidden States分析
│       ├── client.py             # API客户端
│       └── questions.py          # 题目加载
├── test/                         # 测试目录
│   ├── __init__.py
│   ├── test_util.py              # 工具函数测试
│   └── test_hidden_states_analysis.py  # Hidden States分析测试
├── config/                       # 配置文件目录
│   ├── model_server.yaml         # 模型服务器配置
│   └── hidden_states_analysis.yaml  # 分析脚本配置
├── data/                         # 数据目录
│   └── questions/                # 题目数据
│       ├── aime2024_questions.txt
│       └── aime2024_answers.txt
├── output/                       # 输出目录
│   └── hidden_states/            # Hidden States分析结果
├── image/                        # 图片输出目录
├── requirements.txt              # 项目依赖
├── .gitignore                    # Git忽略文件
└── README.md                     # 项目说明
```

## 安装

### 1. 克隆项目

```bash
git clone <repository-url>
cd ComplexUncertainty
```

### 2. 创建虚拟环境（推荐）

```bash
conda create -n uncertainty python=3.10
conda activate uncertainty
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置模型路径

编辑 `config/model_server.yaml`，设置模型路径：

```yaml
model:
  path: "/path/to/your/model"
```

## 使用方法

### 1. 启动模型服务器

```bash
python -m src.server.model_server
```

服务器将在 `http://localhost:8000` 启动。

### 2. 运行Hidden States分析

```bash
# 使用默认配置
python -m test.test_hidden_states_analysis

# 指定题目
python -m test.test_hidden_states_analysis --qid 2024-I-1

# 列出所有可用题目
python -m test.test_hidden_states_analysis --list

# 自定义配置文件
python -m test.test_hidden_states_analysis --config config/custom.yaml
```

### 3. 运行单元测试

```bash
pytest test/
```

## API接口

### 健康检查

```
GET /health
```

### 生成文本并获取Hidden States

```
POST /generate_with_full_hidden_states
```

请求体：
```json
{
    "prompt": "你的问题",
    "system_prompt": "系统提示词",
    "enable_thinking": true,
    "max_new_tokens": 100,
    "temperature": 1.0,
    "top_p": 1.0
}
```

## 主要功能

### 工具函数 (src/util/)

- **相似度计算**: 余弦相似度、相似度矩阵、平均相似度
- **概率计算**: softmax、log概率转换、熵计算、KL散度
- **向量处理**: numpy转换、向量归一化

### Hidden States分析 (src/analysis/)

分析生成过程中每个token的Hidden States变化：

- `h_t_norm`: 当前Hidden State的模长
- `delta_norm`: 与前一个Hidden State的差异模长
- `cos_delta_hprev`: 变化向量与前一个Hidden State的余弦相似度
- `cos_ht_hprev`: 当前与前一个Hidden State的余弦相似度
- `kl_divergence`: KL散度
- `tangent_proj`: 切向投影长度
- `normal_norm`: 法向分量模长
- `relative_change`: 相对变化率

## 配置说明

### model_server.yaml

```yaml
server:
  host: "0.0.0.0"
  port: 8000

model:
  path: "/path/to/model"
  torch_dtype: "bfloat16"
  device_map: "auto"
```

### hidden_states_analysis.yaml

```yaml
api:
  url: "http://localhost:8000"

questions:
  questions_file: "data/questions/aime2024_questions.txt"
  answers_file: "data/questions/aime2024_answers.txt"

defaults:
  qid: "2024-I-1"
  max_tokens: 32768

generation:
  system_prompt: "你是一个数学专家"
  enable_thinking: false
  temperature: 0.6
```

## 许可证

MIT License
