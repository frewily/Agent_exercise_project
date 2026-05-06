# 智能扫地机器人客服系统

基于 LangChain + LangGraph + Streamlit 构建的 ReAct 智能客服，支持 RAG 知识检索、外部数据查询和动态报告生成。
（建议搭配Agent_RAG_学习指南.md阅读源码）

---

## 项目结构

```
PythonProject/
├── app.py                        # Streamlit Web 入口
├── agent/
│   ├── react_agent.py            # ReAct 智能体
│   └── tools/
│       ├── agent_tools.py        # 工具定义
│       └── middleware.py         # 中间件（监控、日志、动态提示词）
├── rag/
│   ├── rag_service.py            # RAG 总结服务
│   └── vector_store.py           # 向量存储服务
├── model/
│   └── factory.py                # 模型工厂（聊天模型 / 嵌入模型）
├── utils/
│   ├── config_handler.py         # 配置加载
│   ├── file_handler.py           # 文件处理（PDF / TXT）
│   ├── logger_handler.py         # 日志
│   ├── path_tool.py              # 绝对路径工具
│   └── prompt_loader.py          # 提示词加载
├── config/
│   ├── agent.yml                 # 外部数据路径配置
│   ├── chroma.yml                # 向量库配置
│   ├── prompts.yml               # 提示词文件路径配置
│   └── rag.yml                   # 模型名称配置
├── prompts/
│   ├── main_prompt.txt           # 客服系统提示词
│   ├── rag_summarize.txt         # RAG 总结提示词
│   └── report_prompt.txt         # 报告生成提示词
├── data/
│   ├── external/records.csv      # 用户使用记录（外部数据）
│   ├── 扫地机器人100问.pdf
│   ├── 扫地机器人100问2.txt
│   ├── 扫拖一体机器人100问.txt
│   ├── 故障排除.txt
│   ├── 维护保养.txt
│   └── 选购指南.txt
└── chroma_d/                     # 向量库持久化目录（自动生成）
```

---

## 架构说明

```
用户输入
   │
   ▼
Streamlit UI (app.py)
   │
   ▼
ReactAgent (agent/react_agent.py)
   │  ┌─────────────────────────────────┐
   │  │ Middleware                       │
   │  │  - monitor_tool   工具调用监控   │
   │  │  - log_before_model  模型前日志  │
   │  │  - report_prompt_switch 动态提示词│
   │  └─────────────────────────────────┘
   │
   ├── rag_summarize ──► RagSummarizeService ──► VectorStoreService (Chroma)
   ├── get_weather
   ├── get_user_city
   ├── get_user_id
   ├── get_current_month
   ├── get_calendar
   ├── fetch_external_data ──► data/external/records.csv
   └── fill_context_for_report ──► 触发提示词切换为报告模式
```

### 报告生成流程

调用 `fill_context_for_report` 后，`report_prompt_switch` 中间件检测到 `context["report"] = True`，将系统提示词从客服模式切换为报告写手模式，后续模型调用使用 `report_prompt.txt`。

固定执行顺序：`get_user_id` → `get_current_month`（或用户指定月份）→ `fill_context_for_report` → `fetch_external_data`

---

## 环境要求

| 依赖 | 版本 |
|------|------|
| Python | 3.10+ |
| langchain | 1.2.16 |
| langchain-community | 0.4.1 |
| langchain-chroma | 1.1.0 |
| langgraph | 1.1.10 |
| streamlit | 1.57.0 |
| PyYAML | 6.0.3 |
| langchain-text-splitters | 1.1.2 |

---

## 配置

### 环境变量

需要配置阿里云 DashScope API Key：

```bash
export DASHSCOPE_API_KEY=your_api_key_here
```

### config/rag.yml

```yaml
chat_model_name: qwen3-max
embedding_model_name: text-embedding-v4
```

### config/chroma.yml

```yaml
collection_name: agent
persist_directory: chroma_d   # 相对于项目根目录
k: 3                          # 检索返回数量
data_path: data
md5_hex_store: md5.text
allow_knowledge_file_type: [pdf, txt]
chunk_size: 200
chunk_overlap: 20
separators: ['.', '?', '!', ';', ':', ',']
```

---

## 启动

```bash
streamlit run app.py
```

首次启动时，`VectorStoreService.load_document()` 会自动扫描 `data/` 目录，将 PDF 和 TXT 文件向量化并写入 `chroma_d/`。已处理的文件通过 MD5 去重，不会重复写入。

---

## 工具说明

| 工具 | 入参 | 说明 |
|------|------|------|
| `rag_summarize` | `query: str` | 从向量库检索知识并总结 |
| `get_weather` | `city: str` | 获取指定城市天气（模拟数据） |
| `get_user_city` | 无 | 获取用户所在城市（模拟数据） |
| `get_user_id` | 无 | 获取用户 ID（模拟数据） |
| `get_current_month` | 无 | 获取系统当前月份，格式 `YYYY-MM` |
| `get_calendar` | `month: str` | 返回指定月份字符串 |
| `fetch_external_data` | `user_id: int, month: str` | 查询用户指定月份使用记录 |
| `fill_context_for_report` | 无 | 触发报告模式提示词切换 |

---

## 日志

日志文件自动生成于 `logs/` 目录，按日期命名（`agent_YYYY-MM-DD.log`）。

- 控制台：INFO 级别
- 文件：DEBUG 级别（包含完整工具调用参数和 RAG 检索内容）

---

## 知识库更新

将新的 PDF 或 TXT 文件放入 `data/` 目录，重启应用后自动向量化入库。无需手动操作，MD5 机制保证不重复处理。
