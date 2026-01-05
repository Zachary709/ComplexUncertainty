"""
模型API客户端模块
"""

import requests
from typing import Optional


def check_health(api_url: str) -> dict:
    """
    检查服务健康状态

    Args:
        api_url: API服务地址

    Returns:
        健康状态信息
    """
    try:
        response = requests.get(f"{api_url}/health")
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "无法连接到服务，请先启动 model_server.py"}


def generate_with_full_hidden_states(
    api_url: str,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    enable_thinking: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> dict:
    """
    调用API生成文本，并获取完整的hidden_states向量

    Args:
        api_url: API服务地址
        prompt: 输入提示
        system_prompt: 系统提示词
        enable_thinking: 是否启用思考模式
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        top_p: top-p采样参数

    Returns:
        API响应结果
    """
    response = requests.post(
        f"{api_url}/generate_with_full_hidden_states",
        json={
            "prompt": prompt,
            "system_prompt": system_prompt,
            "enable_thinking": enable_thinking,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
    )
    return response.json()
