#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小提示探索：上下文工程（Context Engineering）的基础
==============================================================

本文件以“最小、原子化提示（Atomic Prompt）”为出发点，帮助初学者理解如何通过微小改动影响大模型的输出与行为，并以可重复的小实验方式掌握上下文工程的核心理念。

主要涵盖的核心概念：
1. 如何构造原子化提示，以获得更清晰、可控的输出
2. 使用近似的“分词计数”观察提示长度与成本、响应的关系
3. 通过快速迭代微调提示，建立反馈回路（改一点、看变化）
4. 观察“上下文漂移（context drift）”与最小提示的边界
5. 为从原子提示扩展到“协议化提示（protocolized shells）”打下基础

注意：
- 本文件中的 LLM 调用使用一个演示用的“假模型接口”，仅用于讲解流程；
  在生产或真实实验中，请替换为具体厂商的 SDK/API，并使用相应的分词器进行精确计数。
- 每个实验都可作为“可操作练习”，建议你在修改提示后记录分词量与主观质量变化。
- 该文件可作为入门“基础样例”，后续可逐步扩展为更复杂的上下文工程工作流。
"""


import os
import time
import json
from typing import Dict, List, Any, Tuple, Optional
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None  # type: ignore
    HAS_MPL = False

# 如果你使用 OpenAI 的 API，请确保设置环境变量 OPENAI_API_KEY
# 并可通过 OPENAI_MODEL 指定模型（可选，默认 gpt-4o-mini）
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if USE_OPENAI:
    try:
        from openai import OpenAI
        _openai_client: Optional[OpenAI] = OpenAI()
    except Exception:
        _openai_client = None
        USE_OPENAI = False
else:
    _openai_client = None

# 如果你使用其他厂商的接口，请相应调整
# 以下为演示用的“假 LLM 类”
class SimpleLLM:
    """面向教学的最小化 LLM 接口（演示版）。

    说明：
    - 该类不调用任何真实的模型服务，仅用于展示“计数”和“接口形状”。
    - 在真实项目中，请使用对应厂商的 SDK（如 OpenAI、Anthropic 等）并实现同名方法。
    - 保留方法命名与参数结构，可以最小成本替换为真实实现。
    """
    
    def __init__(self, model_name: str = "dummy-model"):
        """初始化演示用 LLM 接口。
        
        参数：
        - model_name: 模型名称标识（演示用途，不影响逻辑）
        """
        self.model_name = model_name
        self.total_tokens_used = 0
        self.total_requests = 0
        
    def count_tokens(self, text: str) -> int:
        """
        近似统计文本的“词”数量，作为“分词数”的粗略替代。
        
        严谨说明：
        - 不同模型/厂商的“分词器（tokenizer）”实现不同，实际 token 数会有差异。
        - 此处采用简单的按空白分隔的词数作为近似，便于初学者快速感知提示长度变化。
        - 在生产或严谨实验中，请使用目标模型的官方分词器进行精确统计。
        """
        # 该近似方法仅用于教学演示；生产实践务必替换为精确分词器
        return len(text.split())
    
    def generate(self, prompt: str) -> str:
        """
        根据提示生成文本。
        
        - 若检测到 OPENAI_API_KEY 则调用 OpenAI 的真实模型服务
        - 否则返回占位提示，便于在无密钥环境下演示流程
        """
        if USE_OPENAI and _openai_client is not None:
            try:
                response = _openai_client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=800,
                )
                content = response.choices[0].message.content or ""
                # 可选：若未来接入精确分词器，可在此累计真实 token
                # usage = getattr(response, "usage", None)
                # if usage:
                #     self.total_tokens_used += usage.prompt_tokens + usage.completion_tokens
                # self.total_requests += 1
                return content
            except Exception as e:
                return f"[OpenAI 接口调用失败：{e}]"
        
        # 无密钥或客户端失败时的占位返回
        tokens = self.count_tokens(prompt)
        self.total_tokens_used += tokens
        self.total_requests += 1
        return f"[未检测到 OPENAI_API_KEY，返回占位响应。你的提示约计 {tokens} 个 token。]"
    
    def get_stats(self) -> Dict[str, Any]:
        """返回当前会话的使用统计信息。
        
        包含：
        - total_tokens: 所有请求累计的近似 token 数
        - total_requests: 请求次数
        - avg_tokens_per_request: 平均每次请求的近似 token 数
        """
        return {
            "total_tokens": self.total_tokens_used,
            "total_requests": self.total_requests,
            "avg_tokens_per_request": self.total_tokens_used / max(1, self.total_requests)
        }

# Initialize our LLM interface
llm = SimpleLLM()

# ----- 实验 1：最小原子化提示（Atomic Prompt） -----
print("\n----- 实验 1：最小原子化提示 -----")
print("先从最基本的单条指令开始。")
# 说明：最小原子化提示 = 一个简洁、可执行的单条指令。
# 目标：在尽量少的 token 下，传达清晰意图，观察模型响应的基本形态。
atomic_prompt = "为一家三口（爸爸、妈妈、4 岁小女孩）规划 3 天的日照海边亲子旅游行程。"
tokens = llm.count_tokens(atomic_prompt)

print(f"\n原子提示：'{atomic_prompt}'")
print(f"Token 数：{tokens}")
print("\n正在生成响应...")
response = llm.generate(atomic_prompt)
print(f"\n响应：\n{response}")

# ----- 实验 2：加入约束观察差异 -----
print("\n----- 实验 2：加入约束观察差异 -----")
print("现在为原子提示加入约束，观察输出的差异。")
# 说明：逐步增加约束（长度、格式、词汇），感受 token 开销与输出变化。
# 建议：记录每次修改后的 token 数与主观质量评分，形成自己的“感觉刻度”。
# Let's create three versions with increasing constraints
prompts = [
    "为一家三口（爸爸、妈妈、4 岁小女孩）规划 3 天的日照海边亲子旅游行程。",  # 原始
    "为一家三口（爸爸、妈妈、4 岁小女孩）规划 3 天的日照海边行程，按每日上午/下午/晚间分段安排，并标注路程与时长。",  # 增加结构与时间分段约束
    "为一家三口规划 3 天日照亲子行程：适合 4 岁幼儿；避开高强度活动与长时间排队；包含餐饮与午休；提供交通方式与人均预算区间（总预算 3000–5000 元）。"  # 多维约束
]

# Measure tokens and generate responses
results = []
for i, prompt in enumerate(prompts):
    tokens = llm.count_tokens(prompt)
    print(f"\n提示 {i+1}：'{prompt}'")
    print(f"Token 数：{tokens}")
    
    start_time = time.time()
    response = llm.generate(prompt)
    end_time = time.time()
    
    results.append({
        "prompt": prompt,
        "tokens": tokens,
        "response": response,
        "latency": end_time - start_time
    })
    
    print(f"延迟：{results[-1]['latency']:.4f} 秒")
    print(f"响应：\n{response}")

# ----- 实验 3：ROI 曲线（Token 数 vs. 主观质量） -----
print("\n----- 实验 3：ROI 曲线测量 -----")
print("探索提示复杂度与输出质量之间的关系。")
# 说明：此处用占位的主观质量分数，演示如何绘制“投入（token）- 产出（质量）”关系。
# 实践：将来可替换为你任务的真实质量评估体系（如评分细则、对齐度、事实性等）。
# In a real notebook, you would define subjective quality scores for each response
# For this demo, we'll use placeholder values
quality_scores = [3, 6, 8]  # Placeholder subjective scores on a scale of 1-10

# Plot tokens vs. quality
if HAS_MPL:
    plt.figure(figsize=(10, 6))
    tokens_list = [r["tokens"] for r in results]
    plt.plot(tokens_list, quality_scores, marker='o', linestyle='-', color='blue')
    plt.xlabel('提示词中的 Token 数')
    plt.ylabel('输出质量（1-10）')
    plt.title('Token–质量 ROI 曲线')
    plt.grid(True)

    # Add annotations
    for i, (x, y) in enumerate(zip(tokens_list, quality_scores)):
        plt.annotate(f"提示 {i+1}", (x, y), textcoords="offset points", 
                     xytext=(0, 10), ha='center')

    # Show the plot (in Jupyter this would display inline)
    # plt.show()
    print("[在 Jupyter 环境中将显示图表]")
else:
    tokens_list = [r["tokens"] for r in results]
    print("[未安装 matplotlib，跳过绘图；在 Jupyter 环境中可显示图表]")

# ----- 实验 4：最小上下文增强（Minimal Context Enhancement） -----
print("\n----- 实验 4：最小上下文增强 -----")
print("在尽量少增加 token 的前提下，加入必要上下文以提升输出质量。")
# 说明：在不显著增加 token 的前提下，补充“必要但关键”的背景或定义信息。
# 原则：只提供能显著降低歧义或提高稳定性的最小上下文。
# Let's create a prompt with a small amount of strategic context
enhanced_prompt = """任务：为一家三口（爸爸、妈妈、4 岁小女孩）规划 3 天的日照海边亲子旅游行程。

目标与边界：
- 行程应松弛有度，适合 4 岁幼儿（避免过度奔波、过度日晒与高强度项目）
- 每天按 上午 / 下午 / 晚间 分段列出安排，并注明预计时长
- 覆盖：亲子友好景点、沙滩玩耍、餐饮与午休、室内备选（遇到下雨/高温）
- 说明地面交通（步行/公交/打车/小红书常见攻略中的替代建议）
- 预算：总预算 3000–5000 元，给出人均预算区间与主要花费构成
- 附加：给 4 岁幼儿的安全与防晒注意事项、备品清单（要精简）

请用结构化小标题与有序列表输出，便于家长直接照单执行。"""

tokens = llm.count_tokens(enhanced_prompt)
print(f"\n增强提示：\n'{enhanced_prompt}'")
print(f"Token 数：{tokens}")

response = llm.generate(enhanced_prompt)
print(f"\n响应：\n{response}")

# ----- 实验 5：一致性测试（Consistency） -----
print("\n----- 实验 5：一致性测量 -----")
print("比较最小提示与增强提示在输出上的一致性。")
# 说明：重复多次采样，观察输出一致性。正式实验可使用语义相似度等量化指标。
# Function to generate multiple responses and measure consistency
def measure_consistency(prompt: str, n_samples: int = 3) -> Dict[str, Any]:
    """对同一提示多次采样，衡量一致性（演示）。

    参数：
    - prompt: 待评估的提示
    - n_samples: 采样次数（默认 3）

    返回：
    - prompt: 原始提示
    - responses: 多次生成的响应列表
    - total_tokens: 总计消耗的近似 token 数（按提示近似计数累加）
    - consistency_score: 一致性得分（占位值；真实实验应实现更严谨的度量）
    """
    responses = []
    total_tokens = 0
    
    for _ in range(n_samples):
        response = llm.generate(prompt)
        responses.append(response)
        total_tokens += llm.count_tokens(prompt)
    
    # 正式实验建议：
    # - 使用向量化/嵌入计算响应间的语义相似度
    # - 使用可重复的随机种子与温度设置，控制可变性
    consistency_score = 0.5  # 占位值，仅作流程演示
    
    return {
        "prompt": prompt,
        "responses": responses,
        "total_tokens": total_tokens,
        "consistency_score": consistency_score
    }

# Compare basic vs enhanced prompt
basic_results = measure_consistency(prompts[0])
enhanced_results = measure_consistency(enhanced_prompt)

print(f"\n基础提示一致性得分：{basic_results['consistency_score']}")
print(f"增强提示一致性得分：{enhanced_results['consistency_score']}")

# ----- 结论 -----
print("\n----- 结论 -----")
print("实验要点：")
# 回顾：
# 1) 小幅提示改动可显著影响输出质量与稳定性
# 2) 存在投入-产出（token-质量）的“最优点”，应围绕任务建模与指标定义来寻找
# 3) 以最小但关键的上下文增强，往往能提升一致性与可控性
# 4) 好的提示应当“清晰、简洁、提供恰到好处的上下文”
print("1. 即便极小的提示增补也可能显著提升输出质量")
print("2. token 与质量之间存在 ROI 曲线，应寻求折中最优点")
print("3. 加入最小且关键的上下文往往能提升一致性")
print("4. 最优提示应清晰、简洁，并提供恰到好处的上下文")

print("\n本文件累计近似 token 数：", llm.get_stats()["total_tokens"])

# ----- 下一步建议 -----
print("\n----- 下一步 -----")
# 建议与练习：
# 1) 连接真实 LLM API（OpenAI、Anthropic 等），替换演示接口与分词器
# 2) 针对你的任务定义质量指标与一致性度量，形成自动化的评测脚本
# 3) 探索“分子级提示（molecules）”：将多个原子指令合理组合
# 4) 在上下文窗口中尝试少样本示例（few-shot），比较与零样本的差异
print("1. 使用真实 LLM API 复现实验")
print("2. 实现更严谨的一致性与质量度量")
print("3. 探索“分子级提示”，将多条原子指令有机组合")
print("4. 在上下文窗口中尝试 few-shot 示例并比较差异")

"""
给读者的练习：

1. 将本文件接入真实 LLM API（OpenAI、Anthropic 等），并替换为对应分词器
2. 在不同模型/不同大小上重复上述实验，观察 token-质量曲线的差异
3. 针对你关心的具体任务，绘制自己的 token-质量曲线
4. 为你的场景寻找“最小可行上下文（Minimum Viable Context）”

参阅 02_expand_context.ipynb，学习更进阶的上下文工程技巧！
"""

# If this were a Jupyter notebook, we'd save the results to a file here
# with open('experiment_results.json', 'w') as f:
#     json.dump(results, f, indent=2)
