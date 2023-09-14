import requests
import json

# 将要发送的数据，通常为字典形式
data = {'input': '一朵美丽的小红花'}
json_data = json.dumps(data)
response = requests.post('http://localhost:8001/emb', data=json_data)

# 输出响应的文本
print(response.text)