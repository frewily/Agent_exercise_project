"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提供给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.logger_handler import logger

def print_prompt(prompt):
    print("=" * 20)
    print( prompt.to_string())
    print("=" * 20)
    return  prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.vector_store.load_document()  # 加载文档到向量存储
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    # 检索文档
    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:

        # 获取的参考资料
        context_docs = self.retriever_docs(query)
        logger.info(f"[RAG] 检索到 {len(context_docs)} 条相关文档")
        
        # 拼接参考资料
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】：参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
            logger.debug(f"[RAG] 文档{counter}: {doc.page_content[:100]}...")

        # 获取模型回复
        return self.chain.invoke(
            {
                "input": query,
                "context": context
            }
        )

if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小型户适合哪些扫地机器人?"))