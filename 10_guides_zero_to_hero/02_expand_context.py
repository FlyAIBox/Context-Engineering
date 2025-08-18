#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上下文扩展技巧：从提示到分层上下文
=============================================================

本笔记以实践为导向，演示如何将基础提示逐步演化为“分层、信息密度更高”的上下文结构，以提升大模型表现。重点在于：如何有策略地添加与组织上下文层，并系统化衡量它们对 token 开销与输出质量的影响。

主要涵盖的核心概念：
# 1. 由最小提示过渡到“信息丰富、结构清晰”的上下文
# 2. 上下文分层（layering）与“组合式提示工程”的原则
# 3. 随上下文增长对 token 使用的定量测量
# 4. 对输出质量的定性评估方法
# 5. 上下文迭代式优化与精炼的实践流程

使用方式：
    # 在 Jupyter 或 Colab 中：
    %run 02_expand_context.py
    # 或
    # 逐步运行各代码块，修改上下文层并观察影响

注意：
    - 各小节相对独立，鼓励你通过增删上下文层来实验。
    - 同步跟踪上下文的增加如何改变成本（token 数）与效果（输出质量）。
    - 可作为构建更高级上下文工程协议的实践基础。
"""

# ## Setup and Prerequisites

# 让我们先导入所需库：

# ```python
import os
import json
import time
import tiktoken  # OpenAI 的分词器
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union

# 加载环境变量（请在 .env 中配置你的 API key）
# 例如 OpenAI API key
import dotenv
dotenv.load_dotenv()

# 选择 API 客户端（按需切换）
USE_OPENAI = True  # 若使用其他厂商，请设为 False

if USE_OPENAI:
    from openai import OpenAI
    client = OpenAI()
    MODEL = "gpt-3.5-turbo"  # 可切换为 gpt-4 等
else:
    # 在此添加其他提供商初始化（如 Anthropic、Cohere 等）
    pass

# 分词器设置
tokenizer = tiktoken.encoding_for_model(MODEL) if USE_OPENAI else None

def count_tokens(text: str) -> int:
    """使用对应分词器统计字符串的 token 数。"""
    if tokenizer:
        return len(tokenizer.encode(text))
    # 非 OpenAI 模型的后备近似（粗略）
    return int(len(text.split()) * 1.3)


def measure_latency(func, *args, **kwargs) -> Tuple[Any, float]:
    """测量函数执行时延。"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time
# ```

# ## 1. 理解上下文扩展

# 在上一份笔记（`01_min_prompt.ipynb`/`01_min_prompt.py`）中，我们探讨了原子提示的基础。现在我们将把这些“原子”策略地扩展为“分子”（更丰富的上下文结构）。

# 下面先定义一些用于衡量上下文有效性的工具函数：

# ```python
def calculate_metrics(prompt: str, response: str, latency: float) -> Dict[str, float]:
    """为一组提示-响应计算关键指标。"""
    prompt_tokens = count_tokens(prompt)
    response_tokens = count_tokens(response)
    
    # 简易 token 效率：响应 token / 提示 token
    token_efficiency = response_tokens / prompt_tokens if prompt_tokens > 0 else 0
    
    # 每千 token 的时延
    latency_per_1k = (latency / prompt_tokens) * 1000 if prompt_tokens > 0 else 0
    
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "token_efficiency": token_efficiency,
        "latency": latency,
        "latency_per_1k": latency_per_1k
    }


def generate_response(prompt: str) -> Tuple[str, float]:
    """向 LLM 生成一次响应，并返回响应与时延。"""
    if USE_OPENAI:
        start_time = time.time()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        latency = time.time() - start_time
        return response.choices[0].message.content, latency
    else:
        # 在此添加其他厂商调用
        pass
# ```

## 2. 实验：上下文扩展技巧

# 我们将考察不同扩展方式对基础提示的影响，并测量每种扩展的效果：

# ```python
# 基础提示（原子）
base_prompt = "请写一段关于气候变化的说明文字。"

# 扩展后的多种提示（分子）
expanded_prompts = {
    "base": base_prompt,
    
    "with_role": """你是一名气候系统方向的环境科学家。
请写一段关于气候变化的说明文字。""",

    "with_examples": """请写一段关于气候变化的说明文字。

示例 1：
气候变化指的是温度和天气模式的长期变化。自 19 世纪以来，人类活动一直是气候变化的主要驱动因素，尤其是燃烧煤、石油、天然气等化石燃料，会产生大量热量捕获性气体。

示例 2：
全球气候变化体现在极端天气事件的频率增加、海平面上升以及野生动物种群迁移。科学共识认为人类活动是主要原因。""",

    "with_constraints": """请写一段关于气候变化的说明文字。
- 至少包含一个带数字的科学事实
- 同时提及成因与影响
- 以“行动倡议”结尾
- 保持信息性且通俗易懂的语气""",

    "with_audience": """请为刚开始接触环境科学的高中生，写一段关于气候变化的说明文字，
尽量使用清晰解释与贴近生活的例子。""",

    "comprehensive": """你是一名气候系统方向的环境科学家。

请为刚开始接触环境科学的高中生，写一段关于气候变化的说明文字，
使用清晰解释与贴近生活的例子。

写作指南：
- 至少包含一个带数字的科学事实
- 同时提及成因与影响
- 以“行动倡议”结尾
- 保持信息性且通俗易懂的语气

语气与结构示例：
"海洋酸化发生在海水从大气中吸收 CO2 时，导致 pH 值下降。自工业革命以来，海洋 pH 下降了约 0.1 个单位，酸度增加约 30%。这会影响海洋生物，尤其是贝类与珊瑚，因为这会削弱它们形成外壳与骨骼的能力。如果排放按当前速度持续，到 2100 年海洋酸度可能上升 150%，对海洋生态系统造成严重影响。通过使用公共交通等简单行动减少碳足迹，我们可以帮助保护这些关键的海洋栖息地。"""
}

# 运行实验
results = {}
responses = {}

for name, prompt in expanded_prompts.items():
    print(f"正在测试提示：{name}")
    response, latency = generate_response(prompt)
    responses[name] = response
    metrics = calculate_metrics(prompt, response, latency)
    results[name] = metrics
    print(f"  提示 token 数：{metrics['prompt_tokens']}")
    print(f"  响应 token 数：{metrics['response_tokens']}")
    print(f"  时延：{metrics['latency']:.2f} 秒")
    print("-" * 40)
# ```

## 3. 可视化与分析

# 为可视化准备数据
prompt_types = list(results.keys())
prompt_tokens = [results[k]['prompt_tokens'] for k in prompt_types]
response_tokens = [results[k]['response_tokens'] for k in prompt_types]
latencies = [results[k]['latency'] for k in prompt_types]

# 绘制多子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图 1：Token 使用量
axes[0, 0].bar(prompt_types, prompt_tokens, label='提示 Token', alpha=0.7, color='blue')
axes[0, 0].bar(prompt_types, response_tokens, bottom=prompt_tokens, label='响应 Token', alpha=0.7, color='green')
axes[0, 0].set_title('不同提示类型的 Token 使用量')
axes[0, 0].set_ylabel('Token 数量')
axes[0, 0].legend()
plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha='right')

# 图 2：Token 效率（响应/提示）
token_efficiency = [results[k]['token_efficiency'] for k in prompt_types]
axes[0, 1].bar(prompt_types, token_efficiency, color='purple', alpha=0.7)
axes[0, 1].set_title('Token 效率（响应/提示）')
axes[0, 1].set_ylabel('效率比值')
plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha='right')

# 图 3：响应时延
axes[1, 0].bar(prompt_types, latencies, color='red', alpha=0.7)
axes[1, 0].set_title('响应时延')
axes[1, 0].set_ylabel('秒')
plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')

# 图 4：每千 token 的时延
latency_per_1k = [results[k]['latency_per_1k'] for k in prompt_types]
axes[1, 1].bar(prompt_types, latency_per_1k, color='orange', alpha=0.7)
axes[1, 1].set_title('每千 Token 的时延')
axes[1, 1].set_ylabel('秒/千 Token')
plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()
# ```

## 4. 定性分析

# 让我们查看具体响应，以评估质量差异：

# ```python
for name, response in responses.items():
    print(f"=== 提示 {name} 的响应 ===")
    print(response)
    print("\n" + "=" * 80 + "\n")
# ```

# ## 5. 上下文扩展模式

# 基于上述实验，我们可以归纳一些常见且有效的上下文扩展模式：

# 1. 角色设定（Role Assignment）：明确模型的扮演者身份
# 2. 少样本（Few-Shot Examples）：提供示例以引导格式与质量
# 3. 约束定义（Constraint Definition）：设置边界与要求
# 4. 受众定位（Audience Specification）：明确面向人群
# 5. 组合上下文（Comprehensive Context）：策略性地组合多个元素

# 将这些模式整理为可复用模板：

# ```python
def create_expanded_context(
    base_prompt: str, 
    role: Optional[str] = None,
    examples: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    audience: Optional[str] = None,
    tone: Optional[str] = None,
    output_format: Optional[str] = None
) -> str:
    """
    基于基础提示与可选组件，创建一个“扩展上下文”。

    参数：
        base_prompt: 核心指令或问题
        role: 模型的角色/身份
        examples: 示例输出列表，用于引导模型
        constraints: 约束或边界条件列表
        audience: 面向的受众群体
        tone: 期望的语气/风格
        output_format: 期望的输出格式

    返回：
        生成的扩展上下文字符串
    """
    context_parts = []
    
    # 角色设定
    if role:
        context_parts.append(f"你的身份：{role}。")
    
    # 基础提示
    context_parts.append(base_prompt)
    
    # 受众
    if audience:
        context_parts.append(f"请确保你的回答适合：{audience}。")
    
    # 语气
    if tone:
        context_parts.append(f"请使用 {tone} 的语气进行回答。")
    
    # 输出格式
    if output_format:
        context_parts.append(f"请将输出格式化为：{output_format}。")
    
    # 约束
    if constraints and len(constraints) > 0:
        context_parts.append("要求：")
        for constraint in constraints:
            context_parts.append(f"- {constraint}")
    
    # 示例
    if examples and len(examples) > 0:
        context_parts.append("示例：")
        for i, example in enumerate(examples, 1):
            context_parts.append(f"示例 {i}：\n{example}")
    
    # 拼接各部分
    expanded_context = "\n\n".join(context_parts)
    
    return expanded_context
# ```

# 用新提示测试我们的模板：

# ```python
# 测试模板
new_base_prompt = "请解释光合作用的工作原理。"

new_expanded_context = create_expanded_context(
    base_prompt=new_base_prompt,
    role="一位有 15 年教学经验的生物老师",
    audience="初中学生",
    tone="热情且富有教育性",
    constraints=[
        "使用‘植物工厂’类比来说明",
        "提及叶绿素的作用",
        "解释其对地球生态的重要性",
        "全文不超过 200 字"
    ],
    examples=[
        "光合作用就像植物体内的一家‘小工厂’。就像工厂需要原料、能量和工人来生产产品，植物也需要二氧化碳、水、阳光和叶绿素来制造葡萄糖（糖）和氧气。阳光是能量来源，叶绿素分子是捕获能量的‘工人’，而二氧化碳和水是原料。‘工厂’的产品是葡萄糖（植物用于生长与储能）与氧气（释放到空气中供我们呼吸）。这个过程对地球生命至关重要，因为它为我们提供氧气并帮助消除大气中的二氧化碳。"
    ]
)

print("模板生成的扩展上下文：")
print("-" * 80)
print(new_expanded_context)
print("-" * 80)
print(f"Token 数：{count_tokens(new_expanded_context)}")

# 使用扩展上下文生成响应
response, latency = generate_response(new_expanded_context)
metrics = calculate_metrics(new_expanded_context, response, latency)

print("\n响应：")
print("-" * 80)
print(response)
print("-" * 80)
print(f"响应 token 数：{metrics['response_tokens']}")
print(f"时延：{metrics['latency']:.2f} 秒")
# ```

# ## 6. 进阶：上下层优化（Layer Optimization）

# 实际应用中，需要在上下文“丰富度”与“token 效率”之间寻求平衡。下面用系统性的方法做分层配置的对比测试：

# ```python
def test_layered_contexts(base_prompt: str, context_layers: Dict[str, str]) -> Dict[str, Dict]:
    """
    测试不同上下文层组合，寻找更优配置。

    参数：
        base_prompt: 基础指令
        context_layers: 上下文层名称 -> 内容 的字典

    返回：
        每个配置的指标结果字典
    """
    layer_results = {}
    
    # 仅基础提示
    print("正在测试基础提示……")
    base_response, base_latency = generate_response(base_prompt)
    layer_results["base"] = {
        "prompt": base_prompt,
        "response": base_response,
        **calculate_metrics(base_prompt, base_response, base_latency)
    }
    
    # 基础 + 单层
    for layer_name, layer_content in context_layers.items():
        combined_prompt = f"{base_prompt}\n\n{layer_content}"
        print(f"正在测试：基础 + {layer_name}……")
        response, latency = generate_response(combined_prompt)
        layer_results[f"base+{layer_name}"] = {
            "prompt": combined_prompt,
            "response": response,
            **calculate_metrics(combined_prompt, response, latency)
        }
    
    # 全部层合并
    all_layers = "\n\n".join(context_layers.values())
    full_prompt = f"{base_prompt}\n\n{all_layers}"
    print("正在测试：全部层合并……")
    full_response, full_latency = generate_response(full_prompt)
    layer_results["all_layers"] = {
        "prompt": full_prompt,
        "response": full_response,
        **calculate_metrics(full_prompt, full_response, full_latency)
    }
    
    return layer_results

# 定义基础提示与各层
layer_test_prompt = "请实现一个简单的天气应用。"

context_layers = {
    "role": "你的身份：一名资深全栈工程师，兼具 UI/UX 设计经验。",

    "requirements": """需求：
- 应用需展示当前温度、天气情况与未来 3 天的预报
- 支持通过城市名称搜索天气
- 界面需简洁且自适应
- 需优雅处理错误状态""",

    "tech_stack": """技术规格：
- 使用 HTML、CSS 与原生 JavaScript（不使用框架）
- 使用 OpenWeatherMap API 获取天气数据
- 代码需具备良好注释并遵循最佳实践
- 请同时给出 HTML 结构与 JavaScript 逻辑""",

    "example": """结构示例（可在此基础上改进）：
# ```html
<!DOCTYPE html>
<html>
<head>
    <title>Weather App</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>Weather App</h1>
        <div class="search">
            <input type="text" placeholder="Enter city name">
            <button>Search</button>
        </div>
        <div class="weather-display">
            <!-- Weather data will be displayed here -->
        </div>
    </div>
    <script src="app.js"></script>
</body>
</html>
# ```"""
}

# 运行层优化测试
layer_test_results = test_layered_contexts(layer_test_prompt, context_layers)
# ```

# 让我们将层优化结果可视化：

# ```python
# 准备可视化数据
config_names = list(layer_test_results.keys())
prompt_sizes = [layer_test_results[k]['prompt_tokens'] for k in config_names]
response_sizes = [layer_test_results[k]['response_tokens'] for k in config_names]
efficiencies = [layer_test_results[k]['token_efficiency'] for k in config_names]

# 绘图
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# 图 1：不同配置的 Token 使用量
axes[0].bar(config_names, prompt_sizes, label='提示 Token', alpha=0.7, color='blue')
axes[0].bar(config_names, response_sizes, label='响应 Token', alpha=0.7, color='green')
axes[0].set_title('不同上下文配置的 Token 使用量')
axes[0].set_ylabel('Token 数量')
axes[0].legend()
plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')

# 图 2：不同配置的 Token 效率
axes[1].bar(config_names, efficiencies, color='purple', alpha=0.7)
axes[1].set_title('不同上下文配置的 Token 效率')
axes[1].set_ylabel('效率比值（响应/提示）')
plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# 标注最高效配置
most_efficient = max(config_names, key=lambda x: layer_test_results[x]['token_efficiency'])
print(f"Token 效率最高的配置：{most_efficient}")
print(f"效率比值：{layer_test_results[most_efficient]['token_efficiency']:.2f}")
# ```

## 7. 上下文压缩技巧

# 随着上下文扩展，常需要优化 token 使用。下列技巧可用于上下文压缩：

# ```python
def compress_context(context: str, technique: str = 'summarize') -> str:
    """
    应用不同压缩技巧，在保留关键信息的前提下减少 token。

    参数：
        context: 待压缩的上下文
        technique: 压缩方式（summarize/keywords/bullet）

    返回：
        压缩后的上下文
    """
    if technique == 'summarize':
        # 让 LLM 对上下文做精炼摘要
        prompt = f"""请将以下上下文精炼为简洁版本，保留所有关键信息，
并尽量减少用词。聚焦核心指令与关键细节：

{context}"""
        compressed, _ = generate_response(prompt)
        return compressed
    
    elif technique == 'keywords':
        # 提取关键术语与短语
        prompt = f"""请从以下上下文中抽取最重要的关键词、短语与指令：

{context}

请以逗号分隔的形式输出关键术语与短短语列表。"""
        keywords, _ = generate_response(prompt)
        return keywords
    
    elif technique == 'bullet':
        # 转换为精简要点列表
        prompt = f"""请将以下上下文转换为精炼、结构化的要点列表，
以最少的文字覆盖全部关键信息：

{context}"""
        bullets, _ = generate_response(prompt)
        return bullets
    
    else:
        return context  # 不压缩

# 在综合示例上测试压缩
original_context = expanded_prompts["comprehensive"]
print(f"原始上下文 token 数：{count_tokens(original_context)}")

for technique in ['summarize', 'keywords', 'bullet']:
    compressed = compress_context(original_context, technique)
    compression_ratio = count_tokens(compressed) / count_tokens(original_context)
    print(f"\n{technique.upper()} 压缩：")
    print("-" * 80)
    print(compressed)
    print("-" * 80)
    print(f"压缩后 token 数：{count_tokens(compressed)}")
    print(f"压缩比：{compression_ratio:.2f}（越低越好）")
# ```

# ## 8. 上下文修剪：删除无效层

# 有时增加的上下文层未必有帮助。我们实现一个评估与修剪的方法：

# ```python
def evaluate_response_quality(prompt: str, response: str, criteria: List[str]) -> float:
    """
    使用 LLM 基于给定标准评估响应质量。

    参数：
        prompt: 生成该响应的提示
        response: 待评估的响应
        criteria: 评价标准列表

    返回：
        质量分（0.0 ~ 1.0）
    """
    criteria_list = "\n".join([f"- {c}" for c in criteria])
    eval_prompt = f"""请根据以下标准为响应质量打分。
    
提示：
{prompt}

响应：
{response}

请基于如下标准进行评估：
{criteria_list}

对于每个标准，请给出 0-10 的分数，并简要说明；最后给出“总体评分”，范围 0.0 至 1.0。
请按如下格式输出：

标准 1：[分数] - [简短评语]
标准 2：[分数] - [简短评语]
...
总体评分：[0.0-1.0]
"""
    
    evaluation, _ = generate_response(eval_prompt)
    
    # 提取总体评分
    try:
        import re
        score_match = re.findall(r"总体评分[:：]\s*([0-9]*\.?[0-9]+)", evaluation)
        if score_match:
            return float(score_match[-1])
        else:
            return 0.5  # 解析失败时的默认值
    except:
        return 0.5  # 异常时的默认值


def prune_context_layers(base_prompt: str, layers: Dict[str, str], criteria: List[str]) -> Tuple[str, Dict]:
    """
    系统化测试并修剪无助于质量提升的上下文层。

    参数：
        base_prompt: 基础指令
        layers: 上下文层名称 -> 内容
        criteria: 质量评估标准

    返回：
        (优化后的提示, 结果字典)
    """
    print("正在测试基线……")
    base_response, base_latency = generate_response(base_prompt)
    base_quality = evaluate_response_quality(base_prompt, base_response, criteria)
    
    results = {
        "base": {
            "prompt": base_prompt,
            "response": base_response,
            "quality": base_quality,
            "tokens": count_tokens(base_prompt),
            "latency": base_latency
        }
    }
    
    # 全部层合并
    all_layers_text = "\n\n".join(layers.values())
    full_prompt = f"{base_prompt}\n\n{all_layers_text}"
    print("正在测试全部层……")
    full_response, full_latency = generate_response(full_prompt)
    full_quality = evaluate_response_quality(full_prompt, full_response, criteria)
    
    results["all_layers"] = {
        "prompt": full_prompt,
        "response": full_response,
        "quality": full_quality,
        "tokens": count_tokens(full_prompt),
        "latency": full_latency
    }
    
    # 逐一移除某一层
    best_quality = full_quality
    best_config = "all_layers"
    
    for layer_to_remove in layers.keys():
        remaining_layers = {k: v for k, v in layers.items() if k != layer_to_remove}
        remaining_text = "\n\n".join(remaining_layers.values())
        test_prompt = f"{base_prompt}\n\n{remaining_text}"
        
        print(f"测试移除层：'{layer_to_remove}'……")
        test_response, test_latency = generate_response(test_prompt)
        test_quality = evaluate_response_quality(test_prompt, test_response, criteria)
        
        config_name = f"without_{layer_to_remove}"
        results[config_name] = {
            "prompt": test_prompt,
            "response": test_response,
            "quality": test_quality,
            "tokens": count_tokens(test_prompt),
            "latency": test_latency
        }
        
        # 若移除后质量提升或不降，则更新最佳配置
        if test_quality >= best_quality:
            best_quality = test_quality
            best_config = config_name
    
    # 若最佳为“全部层”，直接返回
    if best_config == "all_layers":
        return full_prompt, results
    
    # 若移除某层有提升，递归继续修剪
    if best_config.startswith("without_"):
        removed_layer = best_config.replace("without_", "")
        remaining_layers = {k: v for k, v in layers.items() if k != removed_layer}
        print(f"层 '{removed_layer}' 可移除，继续尝试修剪……")
        return prune_context_layers(base_prompt, remaining_layers, criteria)
    
    return results[best_config]["prompt"], results

# 示例：上下文修剪测试
pruning_test_prompt = "请写一篇关于如何使用 pandas 进行数据分析的教程。"

pruning_layers = {
    "role": "你的身份：一名拥有 10+ 年教学经验的数据科学讲师。",
    
    "audience": "受众：初学 Python 的编程者，已了解基础语法但尚无数据分析经验。",
    
    "structure": "结构：请包含以下章节——简介、安装、读取数据、基础操作、数据清洗、数据可视化、实践示例。",
    
    "style": "风格：使用友好、对话式的语气；给出带注释的代码片段并解释关键行；将复杂概念拆解为简单易懂的说明。",
    
    "unnecessary": "冗余：包含 pandas 历史与作者背景等与教程目标无关的信息（如 Wes McKinney 及其早期经历）。"
}

evaluation_criteria = [
    "完整性：覆盖所有关键概念",
    "清晰度：解释是否通俗易懂",
    "代码质量：示例是否有用且正确",
    "入门友好：假设读者无 pandas 经验",
    "实用性：包含真实世界的应用场景"
]

# 下述测试运行时间较长，按需启用
# optimized_prompt, pruning_results = prune_context_layers(pruning_test_prompt, pruning_layers, evaluation_criteria)
# 
# print("\n优化后的提示：")
# print("-" * 80)
# print(optimized_prompt)
# print("-" * 80)
# 
# # 展示各配置的质量分
# for config, data in pruning_results.items():
#     print(f"{config}: 质量 = {data['quality']:.2f}, Token = {data['tokens']}")
# ```

# ## 9. 结合检索的上下文扩展（RAG）

# 在真实应用中，常需用外部知识扩展上下文。下面实现一个简单的“检索增强上下文”：

# ```python
def retrieve_relevant_info(query: str, knowledge_base: List[Dict[str, str]]) -> List[str]:
    """
    基于查询在知识库中检索相关信息。

    参数：
        query: 查询语句
        knowledge_base: 由 {title, content} 构成的条目列表

    返回：
        相关信息片段列表
    """
    # 实际应用中可用向量嵌入与相似度检索；此处用简单关键词匹配演示
    relevant_info = []
    
    query_terms = set(query.lower().split())
    
    for item in knowledge_base:
        content = item['content'].lower()
        title = item['title'].lower()
        
        # 统计匹配项
        matches = sum(1 for term in query_terms if term in content or term in title)
        
        if matches > 0:
            relevant_info.append(item['content'])
    
    return relevant_info[:3]  # 返回前三条

# 示例知识库（真实应用中应更大更全）
sample_knowledge_base = [
    {
        "title": "Pandas Introduction",
        "content": "Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language. Key features include DataFrame objects, handling of missing data, and data alignment."
    },
    {
        "title": "Pandas Installation",
        "content": "To install pandas, run: pip install pandas. For Anaconda users, pandas comes pre-installed. You can import pandas with: import pandas as pd"
    },
    {
        "title": "Loading Data in Pandas",
        "content": "Pandas can read data from various sources including CSV, Excel, SQL databases, and JSON. Example: df = pd.read_csv('data.csv')"
    },
    {
        "title": "Data Cleaning with Pandas",
        "content": "Pandas provides functions for handling missing data, such as dropna() and fillna(). It also offers methods for removing duplicates and transforming data."
    },
    {
        "title": "Data Visualization with Pandas",
        "content": "Pandas integrates with matplotlib to provide plotting capabilities. Simple plots can be created with df.plot(). For more complex visualizations, use: import matplotlib.pyplot as plt"
    }
]

def create_rag_context(base_prompt: str, query: str, knowledge_base: List[Dict[str, str]]) -> str:
    """
    将基础提示与检索到的相关信息合并，生成 RAG 上下文。

    参数：
        base_prompt: 基础指令
        query: 用于检索的查询语句
        knowledge_base: 知识库

    返回：
        含检索信息的扩展上下文
    """
    relevant_info = retrieve_relevant_info(query, knowledge_base)
    
    if not relevant_info:
        return base_prompt
    
    # 将检索信息作为上下文补充
    context_block = "相关信息：\n\n" + "\n\n".join(relevant_info)
    
    # 合并
    rag_context = f"{base_prompt}\n\n{context_block}"
    
    return rag_context

# 测试 RAG 上下文扩展
rag_test_prompt = "请简要讲解如何在 pandas 中读取数据并处理缺失值。"
rag_context = create_rag_context(rag_test_prompt, "pandas loading data cleaning", sample_knowledge_base)

print("RAG 上下文：")
print("-" * 80)
print(rag_context)
print("-" * 80)
print(f"Token 数：{count_tokens(rag_context)}")

# 使用 RAG 上下文生成响应
rag_response, rag_latency = generate_response(rag_context)
print("\nRAG 响应：")
print("-" * 80)
print(rag_response)
print("-" * 80)
# ```

# ## 10. 结论：上下文扩展最佳实践

# 基于以上实验，总结若干有效实践：

# 1. 从最小开始：先用最简单可能有效的提示
# 2. 度量影响：为每次扩展跟踪 token、时延与质量指标
# 3. 策略分层：把上下文按层组织，便于逐层测试与回退
# 4. 尽量压缩：用摘要、要点或关键词降低 token
# 5. 坚决修剪：移除无法提升质量的上下文
# 6. 模板化：为常见扩展模式沉淀可复用模板
# 7. 结合检索：在大知识库情境下，用检索动态扩展上下文
# 8. 平衡特异与泛化：更具体的上下文可降幻觉，但也可能限制创造性

# ### 上下文扩展决策模板

# ```
# 1. 明确核心目标
#   ↓
# 2. 构造最小可行提示
#   ↓
# 3. 测量基线表现
#   ↓
# 4. 识别潜在的上下文层
#   │  - 角色设定
#   │  - 少样本示例
#   │  - 约束与要求
#   │  - 受众定位
#   │  - 语气/风格
#   ↓
# 5. 单独测试各层
#   ↓
# 6. 组合表现较好的层
#   ↓
# 7. 评估影响（token/质量/时延）
#   ↓
# 8. 修剪无效层
#   ↓
# 9. 压缩保留层
#   ↓
# 10. 最终优化（token 效率）
# ```

# 请牢记：目标不是堆砌最多的上下文，而是以更少 token 实现更高质量与效率的“最有效上下文”。

# ## 下一步

# 在下一份笔记（`03_control_loops.ipynb`）中，我们将在此基础上构建更复杂的多步交互控制机制。
