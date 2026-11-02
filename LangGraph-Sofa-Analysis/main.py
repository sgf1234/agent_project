from dotenv import load_dotenv
load_dotenv()
import os
import json
from typing import TypedDict, Annotated, Sequence
import operator
import pandas as pd
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

# -------------------------- 1. 模拟沙发销售数据集（业务数据） --------------------------
def get_sofa_data() -> pd.DataFrame:
    data = [
        {"model":"SF-001","material":"布艺","price":3200,"cost":1400,"channel":"线下门店","sale_date":"2026-01","sales":128,"region":"浙江"},
        {"model":"SF-002","material":"真皮","price":8900,"cost":4200,"channel":"线下门店","sale_date":"2026-01","sales":46,"region":"浙江"},
        {"model":"SF-003","material":"科技布","price":4500,"cost":2100,"channel":"电商抖音","sale_date":"2026-01","sales":215,"region":"江苏"},
        {"model":"SF-001","material":"布艺","price":3200,"cost":1400,"channel":"电商淘宝","sale_date":"2026-02","sales":163,"region":"上海"},
        {"model":"SF-002","material":"真皮","price":8900,"cost":4200,"channel":"线下门店","sale_date":"2026-02","sales":52,"region":"江苏"},
        {"model":"SF-004","material":"猫抓布","price":3800,"cost":1700,"channel":"电商抖音","sale_date":"2026-02","sales":189,"region":"浙江"},
        {"model":"SF-003","material":"科技布","price":4500,"cost":2100,"channel":"电商淘宝","sale_date":"2026-03","sales":176,"region":"上海"},
        {"model":"SF-004","material":"猫抓布","price":3800,"cost":1700,"channel":"线下门店","sale_date":"2026-03","sales":94,"region":"江苏"},
    ]
    return pd.DataFrame(data)

df = get_sofa_data()

# -------------------------- 2. 定义Agent状态 State（核心！面试重点） --------------------------
class AgentState(TypedDict):
    question: str                  # 用户原始自然语言问题
    code: str                      # 生成的python分析代码
    code_result: str               # 代码执行结果
    error_msg: str                 # 报错信息
    retry_count: int               # 重试次数，防止无限循环
    final_answer: str              # 最终整理后的自然语言答案

# -------------------------- 3. LLM初始化（复用DeepSeek） --------------------------
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0
)

# -------------------------- 4. 节点1：生成数据分析代码 --------------------------
def generate_code(state: AgentState):
    prompt = f"""
你是沙发公司数据分析专家，只能基于给定pandas dataframe df做分析。
df字段：model(型号),material(材质:布艺/真皮/科技布/猫抓布),price(售价),cost(成本),channel(渠道),sale_date(年月),sales(销量),region(地区)
要求：
1. 只输出可直接运行的Python代码，不要解释、不要markdown```
2. 结果用print输出清晰汇总，不要绘图
3. 禁止写文件、网络请求、删除数据等高危操作
4. 有计算利润：利润= (price-cost)*sales
用户问题：{state['question']}
如果上一轮代码报错：{state.get('error_msg','')}，你要修正错误
"""
    resp = llm.invoke(prompt)
    return {"code": resp.content.strip(), "retry_count": state["retry_count"] + 1}

# -------------------------- 5. 节点2：安全沙箱执行代码（简化沙箱，加安全限制） --------------------------
def safe_execute(state: AgentState):
    code = state["code"]
    # 安全黑名单：禁止高危函数
    black_list = ["import os","subprocess","eval","exec","open(", "write", "rm", "requests"]
    for bad in black_list:
        if bad in code:
            return {"code_result": "", "error_msg": "检测到高危代码，禁止执行"}
    try:
        # 隔离执行环境，只传入df
        local_env = {"df":df,"pd":pd}
        exec(code, {}, local_env)
        return {"code_result": str(local_env.get("output", "代码执行完成无输出")), "error_msg": ""}
    except Exception as e:
        return {"code_result": "", "error_msg": str(e)}

# -------------------------- 6. 节点3：反思&总结，生成最终答案 --------------------------
def reflect_and_answer(state: AgentState):
    if state["error_msg"]:
        answer = f"多次尝试分析失败：{state['error_msg']}"
    else:
        prompt = f"""
根据代码执行结果，用自然语言给业务人员总结沙发销售分析结论，简洁易懂。
用户问题：{state['question']}
执行结果：{state['code_result']}
"""
        resp = llm.invoke(prompt)
        answer = resp.content
    return {"final_answer": answer}

# -------------------------- 7. 路由函数：判断是否重试 --------------------------
def should_retry(state: AgentState):
    # 最多重试2次，避免死循环
    if state["error_msg"] and state["retry_count"] <= 2:
        return "retry"
    else:
        return "finish"

# -------------------------- 8. 构建LangGraph图 + 开启Checkpoint持久化 --------------------------
builder = StateGraph(AgentState)
# 注册节点
builder.add_node("generate_code", generate_code)
builder.add_node("safe_execute", safe_execute)
builder.add_node("reflect_and_answer", reflect_and_answer)
# 入口
builder.set_entry_point("generate_code")
# 边
builder.add_edge("generate_code", "safe_execute")
# 分支路由
builder.add_conditional_edges(
    "safe_execute",
    should_retry,
    {
        "retry": "generate_code",
        "finish": "reflect_and_answer"
    }
)
builder.add_edge("reflect_and_answer", END)

# 内存checkpoint：支持对话中断、恢复（企业核心特性）
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# -------------------------- 9. 测试入口 --------------------------
if __name__ == "__main__":
    # thread_id 会话ID，同一个id可以中断恢复
    config = {"configurable": {"thread_id": "sofa-biz-001"}}
    user_question = "2026年布艺沙发总销量是多少？"
    res = graph.invoke(
        {"question": user_question, "retry_count": 0, "code": "", "code_result": "", "error_msg": "", "final_answer": ""},
        config=config
    )
    print("=====最终分析结果=====")
    print(res["final_answer"])
