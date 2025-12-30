# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class GRPOConfig:
    num_generations: int = 4
    max_new_tokens: int = 768
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True

    num_policy_updates: int = 1
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.2

    beta_kl: float = 0.0 
    stop_token_ids: Optional[List[int]] = None

    adv_eps: float = 1e-4

    def validate(self):
        assert self.num_generations >= 2, "GRPO requires num_generations>=2 to compute group advantage"
        assert self.max_new_tokens > 0
        if self.do_sample:
            assert self.temperature > 0, "When do_sample=True, temperature must be >0"
