import os
import json
import logging
from typing import List, Optional
from pathlib import Path
from loguru import logger
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from bge_reranker import BGEReranker
import faiss
from pypdf import PdfReader


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-5932882ceb2c49b6ab8db24f7ea8b89b")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
RERANK_TOP = 3
# =================================================================

app = FastAPI(title="医美合规知识库Agentic RAG助手")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 加载模型
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
reranker = BGEReranker(RERANK_MODEL_NAME)

# 全局向量库 & 文档缓存（内存版，重启丢失）
index = faiss.IndexFlatL2(embed_model.get_sentence_embedding_dimension())
chunk_store: List[dict] = []  # {"text": str, "source": str, "page": int}

# 日志配置
logger.add("rag_log.log", rotation="10MB", level="INFO")

# 请求体
class QueryRequest(BaseModel):
    question: str
    use_tool: bool = True

# 工具定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "价格、优惠、分期金额等数学计算，支持 +-*/()",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "数学表达式"}},
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前时间，用于查询营业时间、活动截止时间",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

# ---------------------- 工具函数 ----------------------
def calculator(expression: str):
    import re
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        return f"计算失败：表达式包含非法字符 {expression}"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算报错：{str(e)}"

def get_time():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tool_map = {
    "calculator": calculator,
    "get_time": get_time
}

# ---------------------- 文档加载 & 分块 ----------------------
def load_pdf(file_path: str) -> List[dict]:
    chunks = []
    reader = PdfReader(file_path)
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = text[i:i+CHUNK_SIZE]
            chunks.append({
                "text": chunk_text,
                "source": Path(file_path).name,
                "page": page_idx + 1
            })
    return chunks

def load_md(file_path: str) -> List[dict]:
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_text = text[i:i+CHUNK_SIZE]
        chunks.append({
            "text": chunk_text,
            "source": Path(file_path).name,
            "page": 0
        })
    return chunks

# ---------------------- 向量入库 ----------------------
def add_chunks_to_vector(chunks: List[dict]):
    global index, chunk_store
    texts = [c["text"] for c in chunks]
    embeds = embed_model.encode(texts, normalize_embeddings=True)
    index.add(embeds)
    chunk_store.extend(chunks)
    logger.info(f"入库 {len(chunks)} 个chunk，当前总chunk: {len(chunk_store)}")

# ---------------------- 检索 + Rerank ----------------------
def retrieve_and_rerank(question: str) -> List[dict]:
    if len(chunk_store) == 0:
        return []
    q_emb = embed_model.encode(question, normalize_embeddings=True).reshape(1,-1)
    _, idxs = index.search(q_emb, TOP_K)
    candidates = [chunk_store[i] for i in idxs[0]]
    pairs = [[question, c["text"]] for c in candidates]
    scores = reranker.compute_score(pairs)
    for c, s in zip(candidates, scores):
        c["score"] = s
    candidates.sort(key=lambda x:x["score"], reverse=True)
    return candidates[:RERANK_TOP]

# ---------------------- Agentic RAG主逻辑【医美专属Prompt】 ----------------------
def agentic_rag_flow(question: str, use_tool=True):
    logger.info(f"用户问题：{question}")
    retrieved = retrieve_and_rerank(question)
    source_text = ""
    source_ref = []
    for item in retrieved:
        source_text += f"\n【文档片段】{item['text']}"
        source_ref.append(f"文档：{item['source']}, 页码：{item['page']}")

    system_prompt = """
你是医美机构合规知识库助手，面向客服、咨询师使用。
✅ 规则：
1. 仅基于检索到的知识库内容回答，知识库内容包含：项目介绍、禁忌症、术前术后注意事项、宣传口径、活动规则、收费标准。
2. 严禁输出医疗诊断、疗效承诺、保证效果、替代医师面诊建议！如果用户问能不能做、适不适合，统一回复：需要到院由专业医师面诊评估。
3. 如果检索片段没有对应信息，直接回复【知识库暂无相关合规资料】，禁止编造医美相关内容。
4. 仅在需要计算价格、优惠、查看当前时间时，调用工具。
5. 回答末尾附上引用来源文档和页码。
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"用户问题：{question}\n检索上下文：{source_text if retrieved else '无检索内容'}"}
    ]

    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        tools=tools if use_tool else None,
        temperature=0.1
    )
    msg = resp.choices[0].message

    # 工具调用分支
    if msg.tool_calls:
        logger.info(f"触发工具调用: {msg.tool_calls[0].function.name}")
        tc = msg.tool_calls[0]
        func_name = tc.function.name
        args = json.loads(tc.function.arguments)
        obs = tool_map[func_name](**args)
        messages.append(msg)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})
        resp2 = client.chat.completions.create(model=DEEPSEEK_MODEL, messages=messages, temperature=0.1)
        final_ans = resp2.choices[0].message.content
    else:
        final_ans = msg.content

    out = {
        "answer": final_ans,
        "retrieval_source": source_ref,
        "has_tool_call": bool(msg.tool_calls)
    }
    logger.info(f"返回结果：{json.dumps(out, ensure_ascii=False)}")
    return out

# ===================== API接口 =====================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传医美合规文档（PDF / Markdown），自动切分入库"""
    save_path = f"./upload/{file.filename}"
    os.makedirs("upload", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    if file.filename.lower().endswith(".pdf"):
        chunks = load_pdf(save_path)
    elif file.filename.lower().endswith(".md"):
        chunks = load_md(save_path)
    else:
        return {"code": 400, "msg": "仅支持pdf、md格式文档"}
    add_chunks_to_vector(chunks)
    return {"code": 200, "msg": "上传成功", "chunk_count": len(chunks)}

@app.post("/query")
async def query_rag(req: QueryRequest):
    """提问接口，Agentic RAG医美问答"""
    res = agentic_rag_flow(req.question, req.use_tool)
    return {"code":200, "data": res}

@app.get("/clear")
async def clear_kb():
    """清空知识库向量库"""
    global index, chunk_store
    dim = embed_model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dim)
    chunk_store = []
    logger.info("医美知识库已清空")
    return {"code":200, "msg":"知识库清空完成"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
