from dataclasses import dataclass
from typing import Optional, List

@dataclass
class GRPOConfig:
    # rollout
    num_generations: int = 4
    max_new_tokens: int = 768
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    do_sample: bool = True

    # optimization
    num_policy_updates: int = 1
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.2

    # KL
    beta_kl: float = 0.0
    stop_token_ids: Optional[List[int]] = None

    # advantage norm
    adv_eps: float = 1e-4

    # loss masking (keep conditioning full, but optimize answer part only)
    loss_on_answer_only: bool = True

    def validate(self):
        assert self.num_generations >= 2, "GRPO 需要 num_generations>=2 才有 group advantage"
        assert self.max_new_tokens > 0
        if self.do_sample:
            assert self.temperature > 0, "do_sample=True 时 temperature 必须 >0"
