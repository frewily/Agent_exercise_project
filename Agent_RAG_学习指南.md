# Agent + RAG 知识体系学习指南

> 基于「智能扫地机器人客服系统」项目深度剖析

---

## 目录

- [第一部分：项目全景概览](#第一部分项目全景概览)
- [第二部分：Agent 智能体知识体系](#第二部分agent-智能体知识体系)
- [第三部分：RAG 检索增强生成知识体系](#第三部分rag-检索增强生成知识体系)
- [第四部分：Agent 与 RAG 如何协作](#第四部分agent-与-rag-如何协作)
- [第五部分：进阶知识点](#第五部分进阶知识点)

---

## 第一部分：项目全景概览

### 1.1 这个项目做了什么？

这是一个**智能扫地机器人客服系统**，用户可以像跟真人客服聊天一样提问，背后由一个 AI Agent 自动判断用户意图、调用各种工具获取信息，并生成专业回答。

整体架构：

```
用户浏览器 (Streamlit UI)
        │
        ▼
   ReactAgent（大脑）
   ├── ReAct 思考循环（思考→行动→观察）
   ├── 8 个工具（查知识库、查天气、查用户记录等）
   ├── 3 个中间件（监控、日志、动态提示词切换）
   │
   ├── rag_summarize ──► RAG 总结服务 ──► Chroma 向量数据库
   ├── fetch_external_data ──► CSV 外部数据
   └── fill_context_for_report ──► 动态切换为「报告写手」模式
```

### 1.2 核心文件速览

| 文件 | 作用 | 对应知识点 |
|------|------|-----------|
| `app.py` | Web 界面，用户交互入口 | Streamlit、会话状态管理、流式输出 |
| `agent/react_agent.py` | Agent 主体，创建并运行 Agent | ReAct 模式、Agent 创建、流式输出 |
| `agent/tools/agent_tools.py` | 定义 8 个工具函数 | 工具定义、Tool Calling |
| `agent/tools/middleware.py` | 3 个中间件 | 中间件模式、动态提示词切换 |
| `rag/rag_service.py` | RAG 总结服务 | 检索增强生成完整流程 |
| `rag/vector_store.py` | 向量数据库服务 | 文档加载→分片→向量化→存储→检索 |
| `model/factory.py` | 模型工厂 | 工厂模式、聊天模型 vs 嵌入模型 |
| `config/*.yml` | 配置文件 | 配置管理、关注点分离 |
| `prompts/*.txt` | 提示词模板 | Prompt Engineering |

---

## 第二部分：Agent 智能体知识体系

### 2.1 什么是 Agent（智能体）？

**核心概念**：Agent = 大语言模型 + 工具调用能力 + 自主决策能力。

传统的 LLM 只能根据训练数据生成文本，但 Agent 可以：
- **自主判断**当前是否需要外部信息
- **主动调用工具**（查数据库、调 API、查天气等）
- **根据工具返回结果**调整下一步行动

---

### 2.2 ReAct 模式（Reasoning + Acting）

这是本项目最核心的 Agent 工作模式。

**概念**：ReAct = **思考（Thought）→ 行动（Action）→ 观察（Observation）→ 再思考** 的循环。

#### 项目中的体现

在 [main_prompt.txt](file:///Users/frewily/PycharmProjects/PythonProject/prompts/main_prompt.txt) 中明确定义了这个流程：

```text
严格遵循「思考→行动→观察→再思考」的流程回答用户问题
```

**举个例子**，用户问："扫地机器人在我所在的地区气温下如何保养？"

Agent 的 ReAct 循环可能是这样的：

```
第 1 轮：
  Thought（思考）: 用户想知道在当前地区如何保养机器人，但我不知道他在哪个城市，
                   也不知道当地气温。需要先获取用户城市，再查天气。
  Action（行动）:  调用 get_user_city()
  Observation（观察）: "北京"

第 2 轮：
  Thought（思考）: 获取到用户在北京，接下来需要查北京天气。
  Action（行动）:  调用 get_weather(city="北京")
  Observation（观察）: "城市北京的天气是晴天，气温26摄氏度，南风3级，最近一周多云"

第 3 轮：
  Thought（思考）: 知道了天气信息，但还缺乏关于"高温下如何保养"的专业知识。
                   需要从知识库检索。
  Action（行动）:  调用 rag_summarize(query="高温天气扫地机器人保养")
  Observation（观察）: "夏季高温时，建议避免在阳光直射下充电，电池温度不超过40°C..."

第 4 轮：
  Thought（思考）: 现在有了天气信息（北京26°C晴天）和专业保养知识，
                   可以给出完整回答了。
  Final Answer:    根据北京当前26°C晴天的天气，结合保养知识给出建议...
```

> **关键理解**：Agent 不是一次性调用所有工具，而是**一步一步地思考、行动、观察、再决策**。这正是 ReAct 的精髓。

---

### 2.3 Agent 的创建

**关键代码**：[react_agent.py](file:///Users/frewily/PycharmProjects/PythonProject/agent/react_agent.py#L8-L28)

```python
class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,          # 大脑：通义千问 qwen3-max
            tools=[...],               # 手和脚：8 个工具
            system_prompt=...,         # 行为准则：System Prompt
            middleware=[...],          # 辅助系统：3 个中间件
        )
```

**知识解析**：

| 参数 | 作用 | 类比 |
|------|------|------|
| `model` | 大语言模型，Agent 的「大脑」 | 人的大脑，负责思考和决策 |
| `tools` | Agent 可调用的工具列表 | 人的手和脚，执行具体操作 |
| `system_prompt` | 系统提示词，定义 Agent 的行为准则 | 人的职业培训手册 |
| `middleware` | 中间件，在 Agent 运行的不同阶段插入逻辑 | 工作中的监督和辅助系统 |

`create_agent` 是 LangChain 提供的高级 API，它底层基于 LangGraph 构建了一个 Agent 运行图（Graph），自动处理 ReAct 循环的状态管理。

---

### 2.4 工具定义（Tool Definition）

**关键代码**：[agent_tools.py](file:///Users/frewily/PycharmProjects/PythonProject/agent/tools/agent_tools.py#L16-L84)

这是 Agent 的**能力边界**。每个工具就是一个 Agent 可以调用的函数。

#### 2.4.1 有参工具

```python
@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)

@tool(description="获取指定城市天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    return f"城市{city}的天气是晴天，气温26摄氏度，南风3级，最近一周多云"
```

#### 2.4.2 无参工具

```python
@tool(description="获取用户所在城市名称，以纯字符串的形式返回")
def get_user_city() -> str:
    return random.choice(["北京", "上海", "广州", "深圳", "杭州"])
```

**知识解析**：

`@tool` 装饰器是 LangChain 提供的工具定义方式。它的核心价值在于：

1. **description 是给 Agent 看的** — Agent 通过 description 理解工具的用途，决定何时调用。写好 description 是工具定义的关键！
2. **类型注解用于参数校验** — `city: str` 帮助 Agent 知道该传什么类型的参数
3. **返回值描述要明确** — Agent 需要知道工具会返回什么格式的数据，才能正确解析

> **实践要点**：description 写得好不好，直接决定 Agent 能否正确使用工具。如果 description 模糊，Agent 可能不知道该什么时候调用这个工具。

#### 2.4.3 特殊工具：fill_context_for_report

```python
@tool(description="无入参，无返回值，调用后触发中间件为报告自动生成的场景注入上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"
```

这个工具本身不做任何实质性操作，它的价值在于**触发中间件逻辑**（在 2.5 节详述）。这是一种"信号工具"的设计模式。

#### 2.4.4 外部数据工具：fetch_external_data

```python
@tool(description="从外部系统中获取指定用户在指定月份的使用记录")
def fetch_external_data(user_id: int, month: str) -> str:
    generate_external_data()
    try:
        record = external_data[str(user_id)][month]
        return str(record)
    except KeyError:
        return ""
```

这个工具从 CSV 文件读取数据。CSV 格式如下（推测）：

```
user_id,feature,efficiency,consumables,comparison,time
1001,全屋清扫,95%,滤网需更换,较上月提升5%,2025-06
```

**知识解析**：工具可以把**任何外部系统**的数据接入 Agent。这个例子中的"外部系统"是一个 CSV 文件，但也可以是数据库、API、Excel 等。这是 Agent 打破 LLM 知识边界的关键方式。

---

### 2.5 中间件（Middleware）

**关键代码**：[middleware.py](file:///Users/frewily/PycharmProjects/PythonProject/agent/tools/middleware.py)

中间件是 Agent 运行过程中的**钩子（Hook）**，可以在不同阶段插入自定义逻辑。

#### 2.5.1 三种中间件类型

| 中间件类型 | 触发时机 | 本项目中的使用 |
|-----------|---------|--------------|
| `@wrap_tool_call` | 工具调用前后 | `monitor_tool` — 监控工具执行 |
| `@before_model` | 模型调用前 | `log_before_model` — 记录日志 |
| `@dynamic_prompt` | 每次生成提示词前 | `report_prompt_switch` — 动态切换提示词 |

#### 2.5.2 monitor_tool — 工具调用监控

```python
@wrap_tool_call
def monitor_tool(request, handler):
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")

    if request.tool_call['name'] == "fill_context_for_report":
        request.runtime.context["report"] = True   # 设置上下文标记

    return handler(request)  # 继续执行原工具
```

**知识解析**：

- `request.tool_call` 包含当前被调用的工具名称和参数
- `handler(request)` 是实际执行工具的闭包，调用它才能让工具真正运行
- **关键设计**：当检测到 `fill_context_for_report` 被调用时，在运行时上下文中设置 `report = True` 标记，这个标记会在下一个中间件中被读取

#### 2.5.3 log_before_model — 模型调用前日志

```python
@before_model
def log_before_model(state, runtime):
    logger.info(f"即将调用模型，带有{len(state['messages'])}条消息")
    return None
```

这个中间件很简单：每次 LLM 被调用前，记录当前对话有多少条消息。

**知识解析**：
- `state['messages']` 是整个 Agent 的状态，包含完整的对话历史
- `len(state['messages'])` 可以监控对话长度，防止超过模型的上下文窗口

#### 2.5.4 report_prompt_switch — 动态提示词切换（重点！）

```python
@dynamic_prompt
def report_prompt_switch(request):
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompts()   # 报告写手模式
    return load_system_prompts()       # 普通客服模式
```

**知识解析**：这是本项目最巧妙的设计之一。

整个切换流程：

```
用户说"生成我的使用报告"
    → Agent 思考后决定调用 fill_context_for_report()
    → monitor_tool 中间件拦截到，设置 context["report"] = True
    → 下一轮 Agent 调用模型前，report_prompt_switch 检查 context["report"]
    → 发现为 True，加载 report_prompt.txt 替换 system prompt
    → Agent 的行为模式从「客服」切换为「报告写手」
```

对比两种模式的 System Prompt：

| 方面 | 客服模式 (main_prompt.txt) | 报告模式 (report_prompt.txt) |
|------|--------------------------|---------------------------|
| 角色 | 专业智能客服 | 专业报告写手 |
| 工作流程 | 自由 ReAct 循环 | 固定四步流程 |
| 输出目标 | 回答问题、给建议 | 生成 Markdown 格式报告 |
| 可用工具 | 全部 8 个 | 4 个（精简版） |

> **核心理解**：动态提示词切换让同一个 Agent 在不同场景下表现出完全不同的行为。这类似于一个人在不同工作场景下切换不同的"身份"。

---

### 2.6 Agent 状态管理

Agent 在运行过程中需要维护状态，包括：

1. **对话历史（Messages）**：用户问题和 AI 回复的完整记录
2. **运行时上下文（Runtime Context）**：跨工具调用的临时数据

#### 项目中的体现

在 [react_agent.py](file:///Users/frewily/PycharmProjects/PythonProject/agent/react_agent.py#L31-L43) 中：

```python
def execute_stream(self, query: str):
    input_dict = {
        "messages": [{"role": "user", "content": query}]
    }

    for chunk in self.agent.stream(
        input_dict,
        stream_mode="values",
        context={"report": False}   # 初始上下文：非报告模式
    ):
        latest_message = chunk["messages"][-1]
        if latest_message.content:
            yield latest_message.content.strip() + "\n"
```

**知识解析**：

- `context={"report": False}` 是在 Agent 启动时传入的初始上下文
- 在运行过程中，middleware 可以修改这个上下文（如 `request.runtime.context["report"] = True`）
- `stream_mode="values"` 表示每次状态更新时都输出完整的当前状态

---

### 2.7 流式输出（Streaming）

**关键代码**：[react_agent.py](file:///Users/frewily/PycharmProjects/PythonProject/agent/react_agent.py#L40-L43)

```python
for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        yield latest_message.content.strip() + "\n"
```

**前端接收**：[app.py](file:///Users/frewily/PycharmProjects/PythonProject/app.py#L35-L48)

```python
res_stream = st.session_state["agent"].execute_stream(prompt)
st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
```

**知识解析**：

- `yield` 关键字将函数变为生成器，实现逐块输出
- Streamlit 的 `write_stream` 支持逐字显示，实现 ChatGPT 那样的打字效果
- 流式输出对用户体验至关重要 — 用户不需要等完整回答生成完才能看到内容

---

### 2.8 会话状态管理

**关键代码**：[app.py](file:///Users/frewily/PycharmProjects/PythonProject/app.py#L10-L14)

```python
if "agent" not in st.session_state:
    st.session_state.agent = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []
```

**知识解析**：

- `st.session_state` 是 Streamlit 的会话状态机制，数据在用户会话期间持久化
- Agent 实例只创建一次，避免了每次用户输入都重新初始化向量库
- `message` 列表维护完整的对话历史，实现多轮对话

---

## 第三部分：RAG 检索增强生成知识体系

### 3.1 什么是 RAG？

**RAG = Retrieval-Augmented Generation（检索增强生成）**

核心思想：**先检索相关信息，再让 LLM 基于检索结果生成回答**。

为什么需要 RAG？
- LLM 的知识有截止日期，不知道训练后的新信息
- LLM 可能产生幻觉（编造不存在的事实）
- 企业有私有知识库，LLM 无法直接访问

RAG 解决了这些问题：让 LLM **先查资料，再回答**。

---

### 3.2 RAG 的完整流程

本项目完整实现了一个 RAG 系统，五个步骤：

```
文档加载 → 文本分片 → 向量化 → 存储检索 → 生成回答
  ①          ②          ③        ④          ⑤
```

---

### 3.3 第一步：文档加载（Document Loading）

**关键代码**：[file_handler.py](file:///Users/frewily/PycharmProjects/PythonProject/utils/file_handler.py#L46-L50)

```python
def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()

def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath).load()
```

**项目中的知识库文件**：
- `扫地机器人100问.pdf` — PDF 格式
- `扫地机器人100问2.txt` — TXT 格式
- `扫拖一体机器人100问.txt`
- `故障排除.txt`
- `维护保养.txt`
- `选购指南.txt`

**知识解析**：
- LangChain 提供了丰富的 Document Loader：PyPDFLoader、TextLoader、CSVLoader、UnstructuredMarkdownLoader 等
- 加载后的结果是 `Document` 对象，包含 `page_content`（文本内容）和 `metadata`（元数据）

---

### 3.4 第二步：文本分片（Text Splitting / Chunking）

**关键代码**：[vector_store.py](file:///Users/frewily/PycharmProjects/PythonProject/rag/vector_store.py#L24-L29)

```python
self.splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # 每个分片 200 个字符
    chunk_overlap=20,     # 相邻分片重叠 20 个字符
    separators=['.', '?', '!', ';', ':', ','],  # 优先在这些符号处分割
    length_function=len
)
```

**知识解析**：

**为什么要分片？**
1. 嵌入模型有最大输入长度限制
2. 小块文本的语义更聚焦，检索精度更高
3. LLM 的上下文窗口有限，小块文本更易处理

**核心参数解析**：

| 参数 | 值 | 作用 |
|------|-----|------|
| `chunk_size` | 200 | 每个分片最多 200 字符 |
| `chunk_overlap` | 20 | 相邻分片重叠 20 字符，防止关键信息被切断 |
| `separators` | `.?!;:,` | 优先在句子结束处分割，保持语义完整 |

```
示例：一段 500 字的文本被分割
┌─────────────────────────┐
│ 分片1: 字符 0-200       │
│              重叠区      │
│            ┌────────────┤
│            │ 分片2: 180-380
│            │    重叠区    │
│            │  ┌──────────┤
│            │  │ 分片3: 360-500
└────────────┴──┴──────────┘
```

---

### 3.5 第三步：向量化（Embedding）

**关键代码**：[factory.py](file:///Users/frewily/PycharmProjects/PythonProject/model/factory.py#L21-L23)

```python
class EmbeddingModelFactory(BaseModelFactory):
    def generator(self):
        return DashScopeEmbeddings(model="text-embedding-v4")
```

**知识解析**：

Embedding（嵌入）是将文本转换为数值向量的过程。

```
文本: "扫地机器人如何保养电池"
  ↓ Embedding Model
向量: [0.023, -0.451, 0.789, ..., 0.312]  (如 1024 维)
```

**关键理解**：
- 语义相近的文本，向量距离也近
- "扫地机器人保养" 和 "扫地机维护" 的向量会很接近
- "扫地机器人保养" 和 "今天天气真好" 的向量距离很远

这是相似度检索的数学基础。

**聊天模型 vs 嵌入模型的区别**：

| | 聊天模型 (Chat Model) | 嵌入模型 (Embedding Model) |
|------|------|------|
| 本项目 | `ChatTongyi(qwen3-max)` | `DashScopeEmbeddings(text-embedding-v4)` |
| 输入 | 对话消息 | 文本字符串 |
| 输出 | 自然语言文本 | 数值向量 |
| 用途 | 思考、决策、生成回答 | 文本向量化用于相似度计算 |

---

### 3.6 第四步：向量存储与检索（Vector Store & Retrieval）

**关键代码**：[vector_store.py](file:///Users/frewily/PycharmProjects/PythonProject/rag/vector_store.py#L18-L47)

#### 3.6.1 向量数据库初始化

```python
self.vector_store = Chroma(
    persist_directory="chroma_d",      # 持久化目录
    embedding_function=embed_model,    # 嵌入模型
    collection_name="agent"            # 集合名称
)
```

**知识解析**：

Chroma 是一个轻量级向量数据库。关键参数：
- `persist_directory`：向量数据持久化到磁盘，重启后不丢失
- `embedding_function`：用于将文本转为向量的模型
- `collection_name`：集合名，类似关系数据库中的"表"

#### 3.6.2 文档去重（MD5 机制）

```python
def load_document(self):
    for path in allowed_files_path:
        md5_hex = get_file_md5_hex(path)
        if check_md5_hex(md5_hex):
            continue   # 已处理过，跳过

        documents = get_file_documents(path)
        split_document = self.splitter.split_documents(documents)
        self.vector_store.add_documents(split_document)
        save_md5_hex(md5_hex)
```

**知识解析**：

通过计算文件的 MD5 哈希值来判断文件是否已经被处理过，避免重复向量化。这是一个实用的工程实践：
- 每次启动时自动扫描 data 目录
- 新文件自动入库，旧文件跳过
- MD5 值存储在 `md5.text` 文件中

#### 3.6.3 相似度检索

```python
def get_retriever(self):
    return self.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}   # 返回最相似的 3 个分片
    )
```

**知识解析**：

- `search_type="similarity"`：基于余弦相似度的检索
- `k=3`：返回 Top-3 最相似的文档分片

检索过程：
```
用户问题: "小型户适合哪些扫地机器人?"
    ↓ 向量化
查询向量: [0.12, 0.34, ...]
    ↓ 与向量库中所有分片计算相似度
    ↓ 返回 Top-3
结果: [
  分片A: "小户型推荐选择轻薄款扫地机器人..." (相似度: 0.92)
  分片B: "50平米以下建议选购带激光导航的..." (相似度: 0.87)
  分片C: "小户型选购要点：尺寸、噪音..." (相似度: 0.81)
]
```

---

### 3.7 第五步：生成回答（Generation）

**关键代码**：[rag_service.py](file:///Users/frewily/PycharmProjects/PythonProject/rag/rag_service.py#L14-L52)

#### 3.7.1 RAG Prompt 模板

**关键代码**：[rag_summarize.txt](file:///Users/frewily/PycharmProjects/PythonProject/prompts/rag_summarize.txt)

```text
你是专注于"基于参考资料总结"的AI助手。

### 输入信息
1. 用户提问：{input}
2. 参考资料：{context}

### 严格遵守以下约束
1. 事实准确：回答必须完全基于参考资料中的信息，不编造
2. 聚焦提问：严格围绕用户原始提问总结
```

**知识解析**：

`{input}` 和 `{context}` 是两个占位符，在运行时被替换：
- `{input}` → 用户的原始问题，例如 "小型户适合哪些扫地机器人?"
- `{context}` → 从向量库检索到的参考资料

**这个 Prompt 的核心约束**：
- "回答必须完全基于参考资料" — 防止模型幻觉，确保回答有据可查
- "不编造、不添加未提及的内容" — 即使模型"知道"相关知识，如果参考资料里没有，也不能说

#### 3.7.2 RAG Chain（LCEL 链式调用）

```python
def _init_chain(self):
    chain = self.prompt_template | self.model | StrOutputParser()
    return chain
```

**知识解析**：这是 LangChain 的 LCEL（LangChain Expression Language）表达式。

```
prompt_template | model | StrOutputParser
      ↓              ↓           ↓
  组装提示词      调用LLM    解析输出为纯字符串
```

数据流向：
```
{input: "小型户适合哪些扫地机器人?", context: "参考资料内容..."}
  → PromptTemplate 渲染成完整 Prompt
  → ChatTongyi(qwen3-max) 生成回答
  → StrOutputParser 提取纯文本
  → "小户型建议选择轻薄款扫地机器人，推荐..."
```

#### 3.7.3 完整调用流程

```python
def rag_summarize(self, query: str) -> str:
    # ① 检索
    context_docs = self.retriever_docs(query)

    # ② 拼接参考资料
    context = ""
    for doc in context_docs:
        context += f"【参考资料】：{doc.page_content} | 元数据：{doc.metadata}\n"

    # ③ 调用 chain 生成回答
    return self.chain.invoke({
        "input": query,
        "context": context
    })
```

---

## 第四部分：Agent 与 RAG 如何协作

### 4.1 RAG 作为 Agent 的一个工具

在本项目中，RAG 不是独立运行的系统，而是作为 Agent 的一个工具 `rag_summarize` 存在。

```
用户: "我的扫地机器人应该怎么保养？"
         │
         ▼
    Agent (ReAct 循环)
         │
         ├── 思考: 需要专业知识 → 调用 rag_summarize("扫地机器人保养")
         │         │
         │         ▼
         │    RAG 系统
         │    检索 → 总结 → 返回保养知识
         │         │
         ├── 观察: 获取到了保养知识，信息足够
         │
         └── 最终回答: 基于 RAG 结果生成专业回复
```

**这是 Agent + RAG 最经典的合作模式**：Agent 负责决策"何时需要检索知识"，RAG 负责"检索并提供知识"。

### 4.2 多工具协作场景

更复杂的场景中，Agent 会组合多个工具：

```
用户: "帮我生成6月份的使用报告"

Agent ReAct 循环：
  第1轮: 思考→获取用户ID→调用 get_user_id()→返回 1003
  第2轮: 思考→获取月份→调用 get_calendar("6月")→返回 "2025-06"
  第3轮: 思考→标记报告模式→调用 fill_context_for_report()
         → monitor_tool 设置 context["report"]=True
  第4轮: 思考→获取数据→调用 fetch_external_data(1003, "2025-06")
         → 返回使用记录
  第5轮: report_prompt_switch 检测到 report=True
         → 切换到报告写手模式
         → 生成 Markdown 格式报告

最终输出:
  # 黑马程序员扫地机器人使用情况报告与保养建议
  ## 清洁效率分析
  ...
```

### 4.3 知识边界的分层

Agent + RAG 的组合实现了三层知识体系：

```
┌──────────────────────────────────┐
│ 第1层: LLM 内置知识              │  ← 模型训练时学到的通用知识
│   例: "什么是扫地机器人"         │
├──────────────────────────────────┤
│ 第2层: RAG 知识库                │  ← 企业私有知识文档
│   例: "XX型号的保养周期是3个月" │
├──────────────────────────────────┤
│ 第3层: 外部系统实时数据          │  ← 数据库/API/CSV
│   例: "用户1003在6月的使用记录"  │
└──────────────────────────────────┘
```

Agent 作为调度中心，按需从不同层级获取信息。

---

## 第五部分：进阶知识点

### 5.1 工厂模式（Factory Pattern）

**关键代码**：[factory.py](file:///Users/frewily/PycharmProjects/PythonProject/model/factory.py)

```python
class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self):
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self):
        return ChatTongyi(model="qwen3-max")

class EmbeddingModelFactory(BaseModelFactory):
    def generator(self):
        return DashScopeEmbeddings(model="text-embedding-v4")

chat_model = ChatModelFactory().generator()
embed_model = EmbeddingModelFactory().generator()
```

**知识解析**：
- 使用工厂模式解耦模型创建逻辑
- 好处：如果以后要换模型（如换成 OpenAI），只需要修改工厂类
- `chat_model` 和 `embed_model` 是模块级单例，全局复用避免重复初始化

### 5.2 配置管理

**关键代码**：[config_handler.py](file:///Users/frewily/PycharmProjects/PythonProject/utils/config_handler.py)

项目将配置信息放在 YAML 文件中：
- `rag.yml` — 模型名称
- `chroma.yml` — 向量库参数（chunk_size, k值, 文件类型等）
- `prompts.yml` — 提示词文件路径
- `agent.yml` — 外部数据路径

**知识解析**：
- **配置与代码分离**：修改参数不需要改代码
- YAML 格式：人类可读，结构清晰
- 例如想调整检索返回数量：只需改 `chroma.yml` 的 `k: 3` 为 `k: 5`

### 5.3 日志系统

**关键代码**：[logger_handler.py](file:///Users/frewily/PycharmProjects/PythonProject/utils/logger_handler.py)

```python
logger = get_logger(
    name="agent",
    console_level="INFO",   # 控制台只显示 INFO 及以上
    file_level="DEBUG"      # 文件记录所有 DEBUG 及以上
)
```

**知识解析**：
- 双通道日志：控制台简洁（INFO），文件详细（DEBUG）
- 按日期自动分割日志文件：`agent_2025-06-15.log`
- 防止重复添加 handler：`if logger.hasHandlers(): return logger`

### 5.4 Prompt Engineering（提示词工程）

本项目的三个提示词展示了不同层次的 Prompt Engineering：

#### 5.4.1 客服 System Prompt（main_prompt.txt）

特点：
- 明确的角色定义：**你是扫地机器人的专业智能客服**
- 行为准则：**ReAct 思考准则**（1-4条）
- 工具说明：详细描述每个工具的能力和调用规则
- 输出规则：定义什么时候该输出什么
- 带约束的输出：最多5次工具调用，超过则回复"不知道"

> **这是典型的 Agent System Prompt 写法**：角色 + 行为准则 + 工具说明 + 输出约束

#### 5.4.2 RAG 总结 Prompt（rag_summarize.txt）

特点：
- 明确的占位符：`{input}` 和 `{context}`
- 严格的事实约束：不编造、基于参考资料
- 输出格式约束：纯文本、不封装为 JSON

#### 5.4.3 报告生成 Prompt（report_prompt.txt）

特点：
- 角色切换：从客服变为**报告写手**
- 工具精简：不需要天气、城市等工具
- 输出格式指定：Markdown 格式，固定标题

### 5.5 关键技术对比总结

| 概念 | 一句话解释 | 项目中的对应 |
|------|----------|------------|
| **Agent** | LLM + 工具 + 自主决策 | `ReactAgent` 类 |
| **ReAct** | Think → Act → Observe 循环 | `main_prompt.txt` 中的流程定义 |
| **Tool** | Agent 可调用的函数 | `@tool` 装饰的 8 个函数 |
| **Middleware** | Agent 生命周期的钩子 | `monitor_tool`, `log_before_model`, `report_prompt_switch` |
| **RAG** | 检索 + 生成，先查后答 | `RagSummarizeService` |
| **Embedding** | 文本→向量 | `DashScopeEmbeddings` |
| **Vector Store** | 存储和检索向量 | `Chroma` |
| **Chunking** | 长文本切成小块 | `RecursiveCharacterTextSplitter` |
| **System Prompt** | Agent 的行为准则 | `main_prompt.txt` |
| **Dynamic Prompt** | 运行时切换提示词 | `report_prompt_switch` 中间件 |
| **Streaming** | 逐字输出 | `yield` + `write_stream` |
| **LCEL** | LangChain 的链式调用语法 | `prompt | model | parser` |

---

## 学习路径建议

1. **先理解 Agent 独立运行**（不看 RAG 部分）
   - 运行 `agent/react_agent.py` 的 `__main__` 部分
   - 观察 Agent 的思考→行动→观察循环
   - 理解工具调用的时机和方式

2. **再理解 RAG 独立运行**（不看 Agent 部分）
   - 运行 `rag/rag_service.py` 的 `__main__` 部分
   - 理解文档加载→分片→向量化→检索→生成的全流程

3. **最后理解 Agent + RAG 协作**
   - Agent 把 RAG 当工具调用
   - 理解中间件如何实现动态提示词切换
   - 跟踪一次完整的"生成报告"流程

4. **动手实践**
   - 尝试添加一个新工具（比如让 Agent 能查询产品价格）
   - 尝试添加新的知识文档到 data 目录
   - 尝试修改 chunk_size 或 k 值，观察效果变化

---

> 本文档基于 `智能扫地机器人客服系统` 项目深度剖析生成，涵盖了 Agent 智能体和 RAG 检索增强生成的核心概念及实战应用。建议配合项目源码反复阅读，理论与代码对照理解效果最佳。
