"""
为整个工程提供统一的绝对路径
"""
import os

def get_project_root() -> str:
    """
    获取工程所在根目录
    :return: 字符串根目录
    """
    # 获取当前文件所在目录
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在目录的父目录
    parent_path = os.path.dirname(current_file_path)
    # 获取项目根目录
    project_root = os.path.dirname(parent_path)
    return project_root

def get_abs_path(relative_path: str) -> str:
    """
    获取绝对路径
    :param relative_path: 相对路径
    :return: 绝对路径
    """
    project_root = get_project_root()
    abs_path = os.path.join(project_root, relative_path)
    return abs_path

if __name__ == '__main__':
    print(get_abs_path('data/data.csv'))