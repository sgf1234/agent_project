# Agentic RAG 医美合规知识库问答助手
> 项目定位：面向医美机构客服/咨询师的内部合规问答系统，解决咨询师记忆大量项目规则、宣传口径、禁忌事项的痛点，**严格规避医美宣传违规风险**。
> 技术：DeepSeek API + FastAPI + FAISS + BGE Embedding + BGE-Reranker + Agent Tool Calling

## 功能亮点
1. 文档接入：支持PDF、Markdown医美合规文档上传入库
2. 文本分块、向量化、FAISS粗召回，BGE-Reranker细重排，提升检索精准度
3. Agent联动工具：支持价格优惠计算、查询当前时间
4. 强幻觉抑制 + 医美合规约束：禁止生成医疗诊断、疗效承诺，无资料直接告知
5. 溯源引用：回答附带文档名称+页码，方便核验合规性
6. 完整日志，记录每一轮用户提问、检索片段、工具调用、返回结果
7. 接口化封装，可直接对接内部客服系统

## 技术栈
Python, FastAPI, FAISS, BGE embedding/reranker, DeepSeek OpenAI兼容SDK, PyPDF

## 知识库示例文档（仓库自带测试素材）
- 光子嫩肤项目合规说明.md
- 玻尿酸填充禁忌症&术后注意事项.md
- 暑期活动价目表.pdf

##  启动方式
```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY="key"
python main.py
