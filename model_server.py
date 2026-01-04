"""
模型API服务
启动方式: python model_server.py
服务地址: http://localhost:8000
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Union
import uvicorn

# 模型路径
MODEL_PATH = "/home/zhangdw/models/Qwen/Qwen3-8B"

# 全局变量存储模型和tokenizer
model = None
tokenizer = None

app = FastAPI(title="Qwen3-8B Model API")


class Message(BaseModel):
    """对话消息格式"""
    role: str  # "system", "user", "assistant"
    content: str


class GenerateRequest(BaseModel):
    prompt: Union[str, List[Message]]  # 支持字符串或消息列表
    max_new_tokens: int = 100
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    apply_template: bool = True  # 是否应用聊天模板
    system_prompt: str = "You are a helpful assistant."  # 默认system prompt
    enable_thinking: bool = True  # Qwen3思考模式开关，True启用思考，False禁用思考


def apply_chat_template(prompt: Union[str, List[dict]], system_prompt: str = "You are a helpful assistant.", enable_thinking: bool = True) -> str:
    """
    应用模型的聊天模板
    
    Args:
        prompt: 输入内容，可以是:
            - str: 自动构建为 [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            - list: 直接作为消息列表使用，格式为 [{"role": "...", "content": "..."}]
        system_prompt: 当prompt为str时使用的系统提示词
        enable_thinking: Qwen3思考模式开关
            - True: 启用思考模式，模型会先在<think>...</think>中思考再回答
            - False: 禁用思考模式，模型直接回答
    
    Returns:
        应用模板后的文本
    """
    global tokenizer
    
    if isinstance(prompt, str):
        # 字符串输入：自动补全user和system信息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        # 列表输入：转换Message对象为dict（如果需要）
        messages = []
        for msg in prompt:
            if isinstance(msg, dict):
                messages.append(msg)
            else:
                messages.append({"role": msg.role, "content": msg.content})
    
    # 应用tokenizer的chat_template，传入enable_thinking参数
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking  # Qwen3思考模式硬开关
    )
    return text


def load_model():
    """加载模型和tokenizer"""
    global model, tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    print("Model loaded successfully!")


@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    load_model()


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/generate_with_full_hidden_states")
async def generate_with_full_hidden_states(request: GenerateRequest):
    """生成文本并返回完整的hidden_states向量（数据量较大）"""
    global model, tokenizer
    
    # 处理prompt：根据apply_template决定是否应用模板
    if request.apply_template:
        processed_prompt = apply_chat_template(request.prompt, request.system_prompt, request.enable_thinking)
    else:
        if isinstance(request.prompt, str):
            processed_prompt = request.prompt
        else:
            processed_prompt = "\n".join([m.content if hasattr(m, 'content') else m['content'] for m in request.prompt])
    
    inputs = tokenizer(processed_prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample,
            temperature=request.temperature if request.do_sample else 1.0,
            top_p=request.top_p if request.do_sample else 1.0,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )
    
    generated_ids = outputs.sequences[0]
    generated_tokens = generated_ids[input_length:]
    generated_token_ids = generated_tokens.tolist()
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 计算每个token的top-20 logprobs
    token_logprobs = []
    scores = outputs.scores
    for i, (score, token_id) in enumerate(zip(scores, generated_token_ids)):
        log_probs = torch.log_softmax(score[0], dim=-1)
        
        # 获取top-20 logprobs
        top_k = 20
        top_logprobs, top_indices = torch.topk(log_probs, top_k)
        top_logprobs_list = []
        for j in range(top_k):
            tid = top_indices[j].item()
            top_logprobs_list.append({
                "token_id": tid,
                "token_text": tokenizer.decode([tid]),
                "logprob": top_logprobs[j].item()
            })
        
        token_text = tokenizer.decode([token_id])
        token_logprobs.append({
            "index": i,
            "token_id": token_id,
            "token_text": token_text,
            "logprob": log_probs[token_id].item(),  # 实际选中token的logprob
            "top_logprobs": top_logprobs_list  # top-20 logprobs
        })
    
    # 获取完整的hidden_states
    hidden_states_data = []
    hidden_states = outputs.hidden_states
    for i, token_hidden_states in enumerate(hidden_states):
        last_layer_hidden = token_hidden_states[-1]
        token_hidden = last_layer_hidden[0, -1, :].float().cpu().tolist()
        
        token_id = generated_token_ids[i] if i < len(generated_token_ids) else -1
        token_text = tokenizer.decode([token_id]) if token_id != -1 else "N/A"
        
        hidden_states_data.append({
            "index": i,
            "token_id": token_id,
            "token_text": token_text,
            "hidden_state": token_hidden
        })
    
    return {
        "prompt": request.prompt,
        "generated_text": generated_text,
        "input_length": input_length,
        "output_length": len(generated_tokens),
        "token_logprobs": token_logprobs,
        "hidden_states": hidden_states_data
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
