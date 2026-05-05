"""
向量存储
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
import os

class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            persist_directory=chroma_conf["persist_directory"],
            embedding_function=embed_model,
            collection_name=chroma_conf["collection_name"]
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
        从数据文件夹中读取数据文件，转为向量存入向量库
        要计算文件的md5做去重
        :return: None
        """

        def check_md5_hex(hd5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                #创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                md5_hex_list = f.readlines()
                for md5_hex in md5_hex_list:
                    if md5_hex.strip() == hd5_for_check:
                        return True

                return  False

        def save_md5_hex(md5_hex: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_hex)
                f.write("\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return  []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"{path} has been processed")
                continue
            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.info(f"{path} is empty")
                    continue

                split_document: list[Document] = self.splitter.split_documents(documents)

                if not split_document:
                    logger.info(f"{path}分片后无有效内容")
                    continue

                # 存入向量库
                self.vector_store.add_documents(split_document)

                # 保存md5
                save_md5_hex(md5_hex)

                logger.info(f"{path} has been processed")
            except Exception as e:
                logger.error(e)

if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*20)

