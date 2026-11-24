"""四档压测场景的定义（照抄比赛规则「二.3 性能标准」那张表）。

改 SLO 或并发阶梯时只改这里：评审组开赛前会按官方基线实测后「最终锁定公示」，
到时以公示值为准。

字段说明：
  input_tokens   平均输入 token 数（规则：智能体/编程是「混合长度，最长 256K」）
  output_tokens  平均输出 token 数
  cache_hit      目标缓存命中率（0 表示不复用前缀）
  concurrency    并发阶梯
  ttft_s         P90 首 token 时延上限（秒）
  tpot_ms        P90 每个后续 token 时延上限（毫秒）
  mixed_length   True = 输入长度按分布抽样（均值 = input_tokens，上限 max_input_tokens）
  natural_text   True = 必须用自然文本（规则明确禁止长文本档用随机乱码测速度，
                 乱码会让投机解码的接受率失真）
"""

SCENARIOS = {
    "agent": {
        "title": "智能体场景",
        "input_tokens": 50_000,
        "output_tokens": 1_000,
        "cache_hit": 0.70,
        "concurrency": [8, 16, 32, 64, 80],
        "ttft_s": 15.0,
        "tpot_ms": 75.0,
        "mixed_length": True,
        "max_input_tokens": 256_000,
        "official_dataset": "评审组下发固化多轮编程轨迹",
    },
    "coding": {
        "title": "编程场景",
        "input_tokens": 100_000,
        "output_tokens": 2_000,
        "cache_hit": 0.90,
        "concurrency": [8, 16, 32, 64, 80],
        "ttft_s": 15.0,
        "tpot_ms": 75.0,
        "mixed_length": True,
        "max_input_tokens": 256_000,
        "official_dataset": "评审组下发仓库复用合成集",
    },
    "longgen": {
        "title": "长文本生成",
        "input_tokens": 1_000,
        "output_tokens": 8_000,
        "cache_hit": 0.0,
        "concurrency": [8, 16, 32, 64, 80],
        "ttft_s": 10.0,
        "tpot_ms": 75.0,
        "mixed_length": False,
        "max_input_tokens": 1_000,
        "official_dataset": "SPEED-Bench 固化 60 条",
        "natural_text": True,
    },
    "stress": {
        "title": "高压稳定验证",
        "input_tokens": 32_000,
        "output_tokens": 1_000,
        "cache_hit": 0.0,
        # 注意：规则里这一档只到 64，没有 80
        "concurrency": [8, 16, 32, 64],
        "ttft_s": 20.0,
        "tpot_ms": 75.0,
        "mixed_length": False,
        "max_input_tokens": 32_000,
        "official_dataset": "评审组下发定长合成集",
    },
}

ORDER = ["agent", "coding", "longgen", "stress"]
