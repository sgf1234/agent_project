# Streaming ReAct Agent（Tool Calling，支持并行工具调用）
手写ReAct，不依赖LangChain/LangGraph，原生处理OpenAI兼容流式ToolCall分片拼接

## 功能
1. 流式输出，手动增量拼接tool_call的arguments
2. 支持**并行多工具调用**（asyncio.gather）
3. 同步/异步工具统一适配器 wrap_tool，同步函数用to_thread防阻塞事件循环
4. 内置3个工具：沙盒计算器、时间查询、aiohttp异步联网天气查询
5. 边界保护：最大思考轮次、工具连续失败熔断、工具名校验、JSON参数校验
6. 日志记录每一轮思考、工具入参、返回结果

## 技术栈
Python 3.10+ | asyncio | aiohttp | Pydantic | openai sdk
支持OpenAI兼容接口：DeepSeek、通义千问等，切换base_url和模型名即可

##  安装依赖
```bash
pip install -r requirements.txt
