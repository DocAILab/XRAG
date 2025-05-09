# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys; sys.setrecursionlimit(1500)
project = 'XRAG'
copyright = '2024, xrag'
author = 'XRAG'
release = '1.0'


import os
import sys

sys.path.insert(0, os.path.abspath('../../src')) 
print(sys.path)

# 手动 Mock Config 类或提供 config.toml 路径
from xrag.config import Config

config_path = os.path.abspath('../../config.toml')
Config(config_file_path=config_path)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',        # 自动生成文档
    'sphinx.ext.viewcode',       # 添加 "view source" 链接
    'sphinx.ext.napoleon',       # 支持 Google 和 NumPy 风格 docstring
    'sphinx_autodoc_typehints',  # 支持 Python 类型提示
    'myst_parser',
]
myst_enable_extensions = [
    "dollarmath",    # 支持美元符号包裹的数学公式
    "amsmath",       # 支持更高级的数学公式
    "deflist",       # 支持定义列表
    "html_admonition", # 支持 HTML 风格的警告框
    "html_image",    # 支持 HTML 风格的图片插入
    # 其他 MyST 扩展...
]
# 指定文档源文件的扩展名
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_book_theme'

html_static_path = ['_static']
# autodoc_mock_imports = ['xrag.config', 'xrag.llms']

