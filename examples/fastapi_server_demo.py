# -*- coding: utf-8 -*-
"""

首先，代码导入了所需要的一些Python库，其中包括：
- `os` 和 `sys` 为操作文件与文件路径提供了接口；
- `numpy` 用于处理大规模的数据数组；
- `torch` 是一个开源机器学习框架；
- `argparse` 是Python内置的命令行参数解析包；
- `uvicorn` 是一个基于ASGI规范的服务器，用于运行FastAPI或其它ASGI应用；
- `FastAPI` 是一个用于构建API的框架；
- `CORSMiddleware` 是FastAPI的一个中间件，用于处理跨源请求；
- `BaseModel` 和 `Field` 是Pydantic的模块，用于数据验证；
- `SentenceModel` 是一个用于生成句子嵌入的模型。
"""

import argparse
import uvicorn
import sys
import os
from fastapi import FastAPI, Query
from starlette.middleware.cors import CORSMiddleware
import torch
from loguru import logger
from typing import List
from pydantic import BaseModel, Field
import numpy as np


sys.path.append('..')
from text2vec import SentenceModel

class Item(BaseModel):
    input: str = Field(..., max_length=512)

pwd_path = os.path.abspath(os.path.dirname(__file__))
use_cuda = torch.cuda.is_available()
logger.info(f'use_cuda:{use_cuda}')
# Use fine-tuned model
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="shibing624/text2vec-base-chinese",
                    help="Model save dir or model name")
args = parser.parse_args()
s_model = SentenceModel(args.model_name_or_path)

"""
 这个函数 `_normalize_embedding_2D()` 的作用是将输入的二维数组（向量）标准化。这是一种常用的预处理步骤，它能使得向量的长度（或者说模）为1，而方向不变。

具体来说，这个函数的实现过程如下：

1. 首先，通过 `np.ascontiguousarray()` 函数，确保输入的向量在内存中是连续的。这样可以提高后续计算的效率。

2. 接下来，计算输入向量的模（也就是长度）。这是通过先对向量进行点积运算（`.dot(vec)`），然后对结果求平方根（`np.sqrt()`）来完成的。点积运算可以简单理解为向量中对应元素的乘积之和，而整个计算过程其实就是求向量的欧几里得范数（即长度）。

3. 如果求得的模不为零，则通过 `vec /= norm` 将向量每个元素都除以模的大小。这样就能得到一个模为1的单位向量。注意这里对原向量进行了直接修改（in-place operation），会改变原向量的值。

4. 最后，返回这个单位向量。

所以，任何通过这个函数处理的向量，都会被转化成一个单位向量，模为1。
"""
def _normalize_embedding_2D(vec: np.ndarray) -> np.ndarray:
  vec = np.ascontiguousarray(vec)
  norm = np.sqrt(vec.dot(vec))
  if norm != 0.0:
    vec /= norm
  return vec

# define the app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


@app.get('/')
async def index():
    return {"message": "index, docs url: /docs"}


@app.post('/emb')
async def emb(item: Item):
    try:
        print('item====', item)
        embeddings = s_model.encode(item.input)
        embeddings = np.array(embeddings)
        normalized_embeddings = _normalize_embedding_2D(embeddings)
        result_dict = {'emb': normalized_embeddings.tolist()}
        logger.debug(f"Successfully get sentence embeddings, q:{item.input}")
        return result_dict
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': e}, 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8001)

