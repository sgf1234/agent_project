import re
import inspect
import logging
from pydantic import BaseModel, Field
from datetime import datetime
import aiohttp
import asyncio
import json
from openai import OpenAI

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ReAct-Agent")

# ===================== Pydantic 参数模型 =====================
class CalculatorInput(BaseModel):
    expression: str = Field(
        description="需要计算的数学表达式，支持+-*/和括号，例如（100+23）*5/2"
    )

class TimeInput(BaseModel):
    pass

class WeatherInput(BaseModel):
    city: str = Field(description="要查询天气的城市名称，例如：杭州、北京、上海")

# ===================== 工具函数 =====================
def cal(expression: str):
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        return f"【计算失败】表达式包含非法字符：{expression}，仅支持数字、+ - * / ()"
    try:
        # 沙盒隔离，清空内置函数，降低eval风险
        result = eval(expression, {"__builtins__": {}}, {})
        return f"【计算结果】{expression} = {result}"
    except ZeroDivisionError:
        return f"【计算失败】表达式 {expression} 出现除零错误"
    except SyntaxError:
        return f"【计算失败】表达式 {expression} 语法错误，请检查括号和运算符"
    except Exception as e:
        return f"【计算失败】未知错误：{str(e)}"

async def get_current_time():
    now = datetime.now()
    return f"【当前时间】{now.strftime('%Y-%m-%d %H:%M:%S')}，星期{now.weekday()+1}"

async def get_weather(city: str):
    url = f"https://wttr.in/{city}?format=3"
    try:
        timeout = aiohttp.ClientTimeout(total=8)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return f"【天气查询失败】接口返回状态码 {resp.status}"
                text = await resp.text()
                return f"【天气结果】{text.strip()}"
    except asyncio.TimeoutError:
        return "【天气查询失败】请求超时"
    except Exception as e:
        return f"【天气查询失败】{str(e)}"

# ===================== Tool 定义 =====================
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "用于数学四则运算，当需要计算数值、算术公式的时候调用这个工具。只处理简单数学表达式，不要传入文字。",
            "parameters": CalculatorInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的本地日期、时间，当用户询问现在几点、今天几号、星期几的时候调用",
            "parameters": TimeInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气，用户问某个地方天气怎么样、温度多少的时候调用",
            "parameters": WeatherInput.model_json_schema()
        }
    }
]

tool_map = {
    "calculator": cal,
    "get_current_time": get_current_time,
    "get_weather": get_weather,
}

# ===================== 工具包装器：统一同步/异步工具 =====================
async def wrap_tool(func, **kwargs):
    """自动识别同步/异步函数，同步函数丢到线程池，不阻塞事件循环"""
    if inspect.iscoroutinefunction(func):
        return await func(**kwargs)
    else:
        return await asyncio.to_thread(func, **kwargs)

async def ReAcT_stream(
    user_query,
    client: OpenAI,
    model_name: str,
    max_round: int = 5,        # 最大思考轮次，防止死循环
    max_tool_fail: int = 2     # 连续工具失败阈值
):
    messages = [
        {"role": "system", "content": "你是一个集计算、获取时间、获取天气于一身的助手，缺少数据就调用工具"},
        {"role": "user", "content": user_query}
    ]
    fail_count = 0
    round_idx = 0

    while round_idx < max_round:
        round_idx += 1
        logger.info(f"===== 第 {round_idx} 轮思考 =====")

        try:
            stream = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                temperature=0.1,
                stream=True
            )
        except Exception as e:
            logger.error(f"LLM 请求失败：{e}")
            return "模型接口调用异常，请稍后重试。"

        full_content = ""
        full_tool_calls = None

        print("\n🤖 输出：", end="", flush=True)
        for chunk in stream:
            delta = chunk.choices[0].delta
            # 流式文本实时打印
            if delta.content:
                print(delta.content, end="", flush=True)
                full_content += delta.content
            # 增量拼接tool call参数
            if delta.tool_calls:
                if full_tool_calls is None:
                    full_tool_calls = delta.tool_calls
                else:
                    for new_tc in delta.tool_calls:
                        idx = new_tc.index
                        full_tool_calls[idx].function.arguments += new_tc.function.arguments
        print()

        # 没有工具调用，直接结束
        if not full_tool_calls:
            logger.info(f"本轮无工具调用，直接输出答案：{full_content}")
            return full_content

        logger.info(f"本轮准备调用工具数量：{len(full_tool_calls)}")

        # 组装assistant消息存入上下文
        assistant_msg = {
            "role": "assistant",
            "content": full_content,
            "tool_calls": [tc.model_dump() for tc in full_tool_calls]
        }
        messages.append(assistant_msg)

        # 并行执行工具
        tasks = []
        tc_id_list = []
        tool_info_list = []
        valid = True

        for tc in full_tool_calls:
            func_name = tc.function.name
            raw_args = tc.function.arguments
            tc_id_list.append(tc.id)

            # 校验工具是否存在
            if func_name not in tool_map:
                logger.error(f"模型幻觉，不存在工具：{func_name}")
                valid = False
                break
            # 校验参数json合法性
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                logger.error(f"工具参数JSON解析失败，func={func_name}, args={raw_args}")
                valid = False
                break

            func = tool_map[func_name]
            tool_info_list.append({"name": func_name, "args": args})
            tasks.append(wrap_tool(func,** args))

        if not valid:
            fail_count += 1
            logger.warning(f"工具参数校验失败，当前连续失败次数：{fail_count}")
            if fail_count >= max_tool_fail:
                return "工具调用多次参数异常，无法继续回答。"
            # 追加一条错误tool结果，让模型重新思考
            for tid in tc_id_list:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": "工具调用参数错误或者工具不存在，请重新生成调用参数"
                })
            continue

        # 日志打印本次调用信息
        for info in tool_info_list:
            logger.info(f"调用工具 -> {info['name']}, 参数: {info['args']}")

        results = await asyncio.gather(*tasks)

        # 追加tool返回消息
        for tid, res in zip(tc_id_list, results):
            logger.info(f"工具返回结果: {res}")
            tool_msg = {
                "role": "tool",
                "tool_call_id": tid,
                "content": res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
            }
            messages.append(tool_msg)

        fail_count = 0  # 成功调用，清空失败计数

    # 达到最大轮次
    logger.warning(f"达到最大思考轮次 {max_round}，停止推理")
    return "已达到最大思考轮数，停止工具调用，当前信息不足以回答问题。"

# ===================== 测试入口 =====================
if __name__ == "__main__":
    client = OpenAI(
        api_key="sk-5932882ceb2c49b6ab8db24f7ea8b89b",
        base_url="https://api.deepseek.com"
    )
    ret = asyncio.run(ReAcT_stream(
        "杭州天气怎么样，帮我算 111+222*3",
        client,
        model_name="deepseek-chat"
    ))
    print("\n==== 完整结果 ====")
    print(ret)
