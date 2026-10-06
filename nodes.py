# -*- coding: utf-8 -*-
"""随机分辨率空白 Latent · ComfyUI 节点定义（V1 节点，兼容 ComfyUI 0.3+ / 前端 1.10+）

痛点：同一套提示词批量出图时，希望每张图的尺寸都不一样。

做法：本节点等价于一个「会自动换尺寸的 Empty Latent Image」。
每次按「队列」(Batch count / 重新排队) 执行时，它都会从分辨率池里
取下一个候选，并输出对应的空 latent。

关键实现：重写 IS_CHANGED 返回 NaN —— ComfyUI 会认为该节点每次都变化，
于是强制重新执行；而这个「已变化」标记会沿数据流向上传播到缓存签名里，
所以下游的 KSampler / VAE 也会跟着重新执行，不会命中旧缓存。
"""

from __future__ import annotations

from typing import Any, Dict

import torch

try:  # 在 ComfyUI 内跑时用官方的中间设备 / 精度
    import comfy.model_management as model_management
except Exception:  # pragma: no cover - 独立单元测试时
    model_management = None

from .resolution_pool import (
    DEFAULT_ASPECTS,
    DEFAULT_CUSTOM,
    LATENT_FORMATS,
    PICK_MODES,
    PICK_SEQUENTIAL,
    POOL_SOURCE_PRESET,
    POOL_SOURCES,
    PRESET_GROUPS,
    build_pool,
    describe_pool,
    next_resolution,
)

CATEGORY = "🎲 随机分辨率空白 Latent (Random Resolution Empty Latent)"
PRESET_KEYS = list(PRESET_GROUPS.keys())

# 供「池预览」节点复用的公共输入
_POOL_INPUTS: Dict[str, Any] = {
    "pool_source": (POOL_SOURCES, {
        "default": POOL_SOURCE_PRESET,
        "tooltip": "分辨率从哪里来：内置预设桶 / 自己的列表 / 两者合并 / 按目标像素+宽高比实时计算",
    }),
    "preset_group": (PRESET_KEYS, {
        "default": PRESET_KEYS[0],
        "tooltip": "内置预设桶。全部像素对齐到 64，符合 SDXL / SD1.5 的训练桶，出图最稳",
    }),
    "custom_resolutions": ("STRING", {
        "default": DEFAULT_CUSTOM,
        "multiline": True,
        "dynamicPrompts": False,
        "tooltip": "自定义列表，每行一个 1024x1024。界面上的格子编辑器会读写这个列表；"
                   "也支持 16:9（按目标像素换算）与 # 注释（兼容旧工作流）",
    }),
    "megapixels": ("FLOAT", {
        "default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05,
        "tooltip": "目标像素（百万）。自定义列表里写比例（如 16:9）时、以及「按目标像素生成」模式，都按这个值反推尺寸",
    }),
    "aspect_ratios": ("STRING", {
        "default": DEFAULT_ASPECTS,
        "multiline": False,
        "dynamicPrompts": False,
        "tooltip": "「按目标像素生成」模式使用的宽高比列表，逗号分隔，如 1:1, 3:2, 2:3, 16:9, 9:16, 21:9",
    }),
    "multiple_of": ("INT", {
        "default": 64, "min": 8, "max": 256, "step": 8,
        "tooltip": "自动换算尺寸时的对齐倍数。SDXL 建议 64；SD1.5 低分辨率可用 32/8",
    }),
}


def _make_latent(batch_size: int, channels: int, width: int, height: int, downsample: int):
    shape = [int(batch_size), int(channels), int(height) // downsample, int(width) // downsample]
    kwargs: Dict[str, Any] = {}
    if model_management is not None:
        if hasattr(model_management, "intermediate_device"):
            kwargs["device"] = model_management.intermediate_device()
        if hasattr(model_management, "intermediate_dtype"):
            kwargs["dtype"] = model_management.intermediate_dtype()
    if not kwargs:
        kwargs["dtype"] = torch.float32
    return torch.zeros(shape, **kwargs)


class ResolutionScheduler:
    """🎲 随机分辨率空白 Latent：每次排队自动换一个新分辨率的空 latent。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **_POOL_INPUTS,
                "pick_mode": (PICK_MODES, {
                    "default": PICK_SEQUENTIAL,
                    "tooltip": "挑选方式：顺序循环（一轮走完全部尺寸）/ 随机（可重复）/ 洗牌不重复（每轮内不重复）",
                }),
                "start_index": ("INT", {
                    "default": 0, "min": 0, "max": 4095, "step": 1,
                    "tooltip": "从池里的第几个开始（0 开始计数）。想接着上次的位置继续时用",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64, "step": 1,
                    "tooltip": "同一次执行内一次生成几张同尺寸空 latent（喂给 KSampler 的 batch）。想要「不同尺寸」请用队列的 Batch count",
                }),
                "latent_format": (list(LATENT_FORMATS.keys()), {
                    "default": list(LATENT_FORMATS.keys())[0],
                    "tooltip": "latent 通道数与下采样倍数：SD1.5/SDXL 用 4 通道；SD3/Flux/Qwen 用 16 通道",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": "随机模式的随机源。固定 seed + 递增次数 => 序列可复现；配合控件可每次随机",
                }),
                "advance_each_run": ("BOOLEAN", {
                    "default": True,
                    "label_on": "每次排队都换（开）",
                    "label_off": "锁定当前分辨率（关）",
                    "tooltip": "关掉等于普通 Empty Latent Image：命中缓存，反复排队都用同一个尺寸（用于调试）",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width", "height", "info")
    OUTPUT_TOOLTIPS = (
        "本次选中的空 latent。⚠️ 必须接入采样器（KSampler / KSamplerAdvanced / 自定义采样器等），"
        "不要直接接 VAEDecode——采样器会自动补齐 latent 的通道数与维度",
        "本次宽度（可接到别的节点复用）",
        "本次高度（可接到别的节点复用）",
        "本次调度信息文本",
    )
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "每次排队执行时自动从分辨率池里挑一个新尺寸，输出对应的空 latent，"
        "用于「同一提示词批量产出多种分辨率」。配合队列的 Batch count 使用。\n"
        "⚠️ 输出必须先进采样器再解码：采样器会调用 fix_empty_latent_channels 自动补全"
        "通道数与维度。Qwen-Image / Wan 等 3 维 VAE 模型直接把 4 维 latent 接给 "
        "VAEDecode 会报 shape 越界（tuple index out of range）。"
    )
    SEARCH_ALIASES = [
        "resolution", "size", "width height", "empty latent", "random resolution",
        "batch resolution", "分辨率", "分辨率调度", "空latent", "批量尺寸",
        "随机分辨率", "空白latent", "随机尺寸", "random resolution empty latent",
    ]

    @classmethod
    def IS_CHANGED(cls, advance_each_run: bool = True, **_kwargs):
        # 返回 NaN => ComfyUI 判定本节点「每次都变了」，并且该判定会作为
        # 缓存签名的一部分向上传给下游节点，因此 KSampler 等不会被旧缓存命中。
        return float("NaN") if advance_each_run else "resolution-scheduler-locked"

    def run(
        self,
        pool_source: str,
        preset_group: str,
        custom_resolutions: str,
        megapixels: float,
        aspect_ratios: str,
        multiple_of: int,
        pick_mode: str,
        start_index: int,
        batch_size: int,
        latent_format: str,
        seed: int,
        advance_each_run: bool = True,
        unique_id: Any = None,
    ):
        sizes, notes = build_pool(
            source=pool_source,
            preset_group=preset_group,
            custom_text=custom_resolutions,
            megapixels=megapixels,
            aspect_text=aspect_ratios,
            multiple_of=multiple_of,
        )

        # 关键参数一改就自动重排队列（seed 不入签名：它可能被前端每次随机化）
        signature = (
            pool_source, preset_group, (custom_resolutions or "").strip(),
            round(float(megapixels), 4), (aspect_ratios or "").strip(),
            int(multiple_of), pick_mode, int(start_index), latent_format,
        )
        key = str(unique_id) if unique_id is not None else "resolution_scheduler::global"

        width, height, seq, index = next_resolution(
            key=key,
            signature=signature,
            pick_mode=pick_mode,
            sizes=sizes,
            start_index=start_index,
            seed=seed,
        )

        channels, downsample = LATENT_FORMATS[latent_format]
        latent = _make_latent(batch_size, channels, width, height, downsample)

        info = (
            f"#{seq}  {width}×{height}  {width * height / 1e6:.2f}MP"
            f"  |  池 {len(sizes)} 种 · 第 {index + 1} 个 · {pick_mode.split(' ')[0]}"
        )
        if not advance_each_run:
            info += "\n⚠ 已锁定分辨率（advance_each_run 关闭）"
        if notes:
            info += "\n⚠ " + "\n⚠ ".join(notes)

        return (
            {"samples": latent, "downscale_ratio_spacial": downsample},
            width,
            height,
            info,
            {"ui": {"text": [info]}},
        )


class ResolutionPoolPreview:
    """🔍 分辨率池预览：不参与采样，只是把当前池子列出来核对。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": dict(_POOL_INPUTS)}

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("pool_text", "pool_size")
    OUTPUT_TOOLTIPS = ("分辨率池的可读清单", "池中分辨率数量")
    FUNCTION = "preview"
    CATEGORY = CATEGORY
    DESCRIPTION = "把当前参数组合出来的分辨率池列出来核对（尤其是自定义列表有没有写错）。"
    OUTPUT_NODE = True
    SEARCH_ALIASES = ["resolution pool", "pool preview", "分辨率池", "预览"]

    def preview(
        self,
        pool_source: str,
        preset_group: str,
        custom_resolutions: str,
        megapixels: float,
        aspect_ratios: str,
        multiple_of: int,
    ):
        sizes, notes = build_pool(
            source=pool_source,
            preset_group=preset_group,
            custom_text=custom_resolutions,
            megapixels=megapixels,
            aspect_text=aspect_ratios,
            multiple_of=multiple_of,
        )
        text = describe_pool(sizes, source_label=pool_source)
        if notes:
            text += "\n\n提示：\n" + "\n".join(f"· {n}" for n in notes)
        return (text, len(sizes), {"ui": {"text": [text]}})


NODE_CLASS_MAPPINGS = {
    "ResolutionScheduler": ResolutionScheduler,
    "ResolutionPoolPreview": ResolutionPoolPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResolutionScheduler": "🎲 随机分辨率空白 Latent (Random Resolution Empty Latent)",
    "ResolutionPoolPreview": "🔍 随机分辨率池预览 (Random Resolution Pool Preview)",
}
