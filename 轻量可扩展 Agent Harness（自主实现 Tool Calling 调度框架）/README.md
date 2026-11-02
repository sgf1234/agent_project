# Agent Harness：轻量可扩展 Tool Calling 调度框架
自研轻量Agent调度内核，面向DeepSeek/Qwen/Ollama等兼容OpenAI协议的大模型，
聚焦ToolCall调度、工具权限管控、会话隔离、同步/异步工具统一适配。
> 定位：不是业务Agent Demo，是一套可以快速批量搭建工具Agent的底座；
> 适合简单~中等复杂度工具调用场景，和LangGraph形成互补。

## 核心亮点
1. **职责解耦模块化设计**
    - 权限管理器：`ALLOW` 直接执行 / `ASK` 人工审批 / `DENY` 禁止执行，生产级安全策略
    - 工具注册表：统一注册工具，自动生成OpenAI规范tools schema，隐藏函数实现
    - 会话上下文管理器：多session隔离，独立消息历史，预留token窗口、持久化扩展
    - LLM适配器：厂商隔离，切换DeepSeek、通义千问、Ollama只改适配器层，不改动调度核心
    - Runtime调度器：ReAct循环，自动调用工具、回填结果，上限循环防止死循环
2. **工具兼容**：自动识别同步 / async异步工具，统一封装执行
3. **安全机制**：高危工具人工审批拦截，禁止工具直接禁用
4. **协议标准**：完全遵循OpenAI tool call消息格式，兼容DeepSeek、Qwen等主流API