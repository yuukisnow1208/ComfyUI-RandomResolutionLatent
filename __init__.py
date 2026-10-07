# -*- coding: utf-8 -*-
"""ComfyUI-RandomResolutionLatent · 🎲 随机分辨率空白 Latent

每次排队执行自动切换一个「合理的空 latent 分辨率」。

解决：同一套提示词批量出图时，想让每张图尺寸都不一样，但又不想手动
一个个改 Empty Latent Image 的宽高 / 反复重排队列。

节点类名仍为 ResolutionScheduler（保持工作流兼容），仅显示名做了更名。

WEB_DIRECTORY 只在这里声明（pyproject 里再写会按模块名和 project.name
各注册一次，JS 被导入两遍）。
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__version__ = "1.2.1"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
