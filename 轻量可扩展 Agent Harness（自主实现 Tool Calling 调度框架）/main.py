from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Callable, Dict, Any, Optional, List
import json
import inspect
from enum import StrEnum

# =====================================================
# 【模块1：权限枚举 + 权限管理器（Harness安全层）】
# =====================================================
class Permission(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"

class PermissionManager:
    def __init__(self):
        self.tool_policy: Dict[str, Permission] = {}

    def set_tool_permission(self, tool_name: str, perm: Permission):
        self.tool_policy[tool_name] = perm

    def check(self, tool_name: str) -> Permission:
        return self.tool_policy.get(tool_name, Permission.DENY)


# =====================================================
# 【模块2：工具注册表 ToolRegistry（Harness工具舱）】
# =====================================================
class ToolSpec(BaseModel):
    name: str
    description: str
    params_schema: dict
    func: Callable
    permission: Permission

class ToolRegistry:
    def __init__(self, perm_mgr: PermissionManager):
        self.tools: Dict[str, ToolSpec] = {}
        self.perm_mgr = perm_mgr

    def register(self, name: str, description: str, params_schema: dict, func: Callable, permission: Permission):
        self.tools[name] = ToolSpec(
            name=name,
            description=description,
            params_schema=params_schema,
            func=func,
            permission=permission
        )
        self.perm_mgr.set_tool_permission(name, permission)

    def get_openai_tools_list(self) -> List[dict]:
        out = []
        for t in self.tools.values():
            out.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.params_schema
                }
            })
        return out

    async def execute(self, tool_name: str, args: dict, user_approve_hook) -> str:
        spec = self.tools.get(tool_name)
        if not spec:
            return f"错误：不存在工具 {tool_name}"

        perm = self.perm_mgr.check(tool_name)
        if perm == Permission.DENY:
            return f"安全拦截：工具 {tool_name} 被禁止"
        if perm == Permission.ASK:
            ok = await user_approve_hook(tool_name, args)
            if not ok:
                return f"用户拒绝执行工具 {tool_name}"
        try:
            if inspect.iscoroutinefunction(spec.func):
                result = await spec.func(**args)
            else:
                result = spec.func(**args)
            return str(result)
        except Exception as e:
            return f"工具执行异常：{repr(e)}"


# =====================================================
# 【模块3：会话上下文管理器 ContextManager】
# =====================================================
class SessionContext:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Dict[str, Any]] = []

    def add_msg(self, role: str, content: Optional[str] = None, tool_calls: Optional[list] = None, tool_result: Optional[dict] = None):
        if tool_calls:
            self.messages.append({"role": "assistant", "tool_calls": tool_calls})
        elif tool_result:
            self.messages.append({"role": "tool", **tool_result})
        else:
            self.messages.append({"role": role, "content": content})

    def get_messages(self):
        return self.messages

class ContextManager:
    def __init__(self):
        self.sessions: Dict[str, SessionContext] = {}

    def get_or_create_session(self, session_id: str) -> SessionContext:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionContext(session_id)
        return self.sessions[session_id]


# =====================================================
# 【模块4：模型适配器 LLMAdapter】
# =====================================================
class LLMAdapter:
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    async def chat(self, messages: list, tools: list):
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        choice = resp.choices[0]
        return choice.message


# =====================================================
# 【模块5：Harness 主循环 Runtime】
# =====================================================
class AgentHarness:
    def __init__(
        self,
        llm: LLMAdapter,
        tool_reg: ToolRegistry,
        ctx_mgr: ContextManager,
        max_loop: int = 5
    ):
        self.llm = llm
        self.tool_reg = tool_reg
        self.ctx_mgr = ctx_mgr
        self.max_loop = max_loop

    async def run(self, session_id: str, user_query: str, approve_hook):
        sess = self.ctx_mgr.get_or_create_session(session_id)
        sess.add_msg(role="user", content=user_query)

        for loop in range(self.max_loop):
            msg = await self.llm.chat(sess.get_messages(), self.tool_reg.get_openai_tools_list())

            if msg.tool_calls:
                sess.add_msg(role="assistant", tool_calls=msg.tool_calls)
                for tc in msg.tool_calls:
                    func_name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    tool_out = await self.tool_reg.execute(func_name, args, approve_hook)
                    sess.add_msg(tool_result={
                        "tool_call_id": tc.id,
                        "content": tool_out
                    })
            else:
                sess.add_msg(role="assistant", content=msg.content)
                return msg.content
        return "达到最大循环次数，任务终止"


# =====================================================
# ========== 演示代码 ==========
# =====================================================
async def demo_approve(tool_name, args):
    print(f"\n【需要审批】准备执行工具：{tool_name} 参数={args}，y确认/n拒绝")
    inp = input("> ").strip().lower()
    return inp == "y"

perm_mgr = PermissionManager()
ctx_mgr = ContextManager()
tool_reg = ToolRegistry(perm_mgr)

# 工具1：加法
def add(a:int, b:int) -> int:
    return a + b
tool_reg.register(
    name="add",
    description="两个数字相加",
    params_schema={
        "type":"object",
        "properties":{"a":{"type":"number"},"b":{"type":"number"}},
        "required":["a","b"]
    },
    func=add,
    permission=Permission.ALLOW
)

# 工具2：删除文件，人工审批
def delete_file(path:str) -> str:
    return f"【模拟】已删除 {path}"
tool_reg.register(
    name="delete_file",
    description="删除指定路径文件",
    params_schema={
        "type":"object",
        "properties":{"path":{"type":"string"}},
        "required":["path"]
    },
    func=delete_file,
    permission=Permission.ASK
)

# 工具3：关机，禁止
def shutdown() -> str:
    return "系统关机"
tool_reg.register(
    name="shutdown",
    description="关闭服务器",
    params_schema={"type":"object","properties":{}},
    func=shutdown,
    permission=Permission.DENY
)

llm = LLMAdapter(
    base_url="https://api.deepseek.com/v1",
    api_key="你的key",
    model_name="deepseek-chat"
)

harness = AgentHarness(llm, tool_reg, ctx_mgr, max_loop=5)

import asyncio
async def main():
    sid = "user_001"
    res = await harness.run(
        session_id=sid,
        user_query="帮我算 123 + 456，然后删除 /tmp/log.txt，最后关机",
        approve_hook=demo_approve
    )
    print("\n====最终回答====\n", res)

asyncio.run(main())
