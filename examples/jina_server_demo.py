# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install jina
"""


# from jina import Flow

# port = 50001
# f = Flow(port=port).add(
#     uses='jinahub://Text2vecEncoder',
#     uses_with={'model_name': 'shibing624/text2vec-base-chinese'},
#     install_requirements = True
# )

# with f:
#     # start server, backend server forever
#     f.block()


from jina import Flow

if __name__ == '__main__':
    port = 50001
    f = Flow(port=port).add(
        uses='jinahub://Text2vecEncoder',
        uses_with={'model_name': 'shibing624/text2vec-base-chinese'},
        install_requirements=True
    )

    with f:
        # start server, backend server forever
        f.block()