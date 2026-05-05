import hashlib
import os

from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def get_file_md5_hex(filepath: str):

    if not os.path.exists(filepath):
        logger.error(f"File {filepath} does not exist")
        return

    if not os.path.isfile(filepath):
        logger.error(f"{filepath} is not a file")
        return

    md5_obj = hashlib.md5()

    chunk_size = 4096  # 4KB,避免内存溢出
    try:
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                md5_obj.update(chunk)
        return md5_obj.hexdigest()
    except Exception as e:
        logger.error(e)

def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):  #返回文件夹内的文件列表（允许指定文件类型）
    file = []

    if not os.path.exists(path):
        logger.error(f"Path {path} does not exist")
        return []

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            file.append(os.path.join(path, f))

    return tuple(file)

def pdf_loader(filepath: str,passwd = None) -> list[Document]:
    return PyPDFLoader(filepath,passwd).load()

def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath).load()