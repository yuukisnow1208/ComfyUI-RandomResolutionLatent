# -*- coding: utf-8 -*-
"""随机分辨率空白 Latent · 纯逻辑层

不依赖 ComfyUI / torch，可单独导入测试。负责三件事：

1. 维护内置的「分辨率预设桶」（全部按 64 对齐，贴合 SDXL / SD1.5 训练桶）；
2. 解析用户自定义列表（支持 ``1024x1024`` / ``1024*1024`` / ``1024,1024`` /
   ``1024×1024`` / ``16:9`` 混写，支持 ``#`` 注释）；
3. 维护「每次排队换一个分辨率」的调度状态机
   （顺序循环 / 随机 / 洗牌不重复）。

计数器按节点 ``unique_id`` 隔离；当关键参数发生变化（换预设、改列表、
改模式、改起始序号）时自动归零重排。
"""

from __future__ import annotations

import math
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

MIN_SIDE = 64
MAX_SIDE = 8192
LATENT_STEP = 8  # VAE 下采样倍数，像素必须对齐到 8

# 池来源
POOL_SOURCE_PRESET = "预设桶 (preset)"
POOL_SOURCE_CUSTOM = "自定义列表 (custom)"
POOL_SOURCE_MERGE = "预设 + 自定义 (merge)"
POOL_SOURCE_MP = "按目标像素生成 (megapixel)"
POOL_SOURCES = [
    POOL_SOURCE_PRESET,
    POOL_SOURCE_CUSTOM,
    POOL_SOURCE_MERGE,
    POOL_SOURCE_MP,
]

# 挑选方式
PICK_SEQUENTIAL = "顺序循环 (sequential)"
PICK_RANDOM = "随机 (random)"
PICK_SHUFFLE = "洗牌不重复 (shuffle)"
PICK_MODES = [PICK_SEQUENTIAL, PICK_RANDOM, PICK_SHUFFLE]

# latent 格式：显示名 -> (通道数, 空间下采样倍数)
LATENT_FORMATS: Dict[str, Tuple[int, int]] = {
    "SD1.5 / SDXL (4 通道, ÷8)": (4, 8),
    "SD3 / Flux / Qwen-Image (16 通道, ÷8)": (16, 8),
}

# 内置预设桶：像素全部是 64 的整数倍
PRESET_GROUPS: Dict[str, List[Tuple[int, int]]] = {
    "SDXL 标准桶 · 1MP (9 种)": [
        (1024, 1024),
        (1152, 896), (896, 1152),
        (1216, 832), (832, 1216),
        (1344, 768), (768, 1344),
        (1536, 640), (640, 1536),
    ],
    "横竖方三档 · 1MP (3 种)": [
        (1024, 1024),
        (1216, 832),
        (832, 1216),
    ],
    "SD1.5 标准桶 · 0.4MP (8 种)": [
        (512, 512),
        (512, 768), (768, 512),
        (640, 640),
        (576, 768), (768, 576),
        (448, 768), (768, 448),
    ],
    "16:9 视频帧 · 横竖 (8 种)": [
        (1024, 576), (1344, 768), (1536, 864), (1920, 1088),
        (576, 1024), (768, 1344), (864, 1536), (1088, 1920),
    ],
    "高清 2MP · Flux / Qwen (5 种)": [
        (1408, 1408),
        (1728, 1152), (1152, 1728),
        (1920, 1088), (1088, 1920),
    ],
    "方形多档 · 调试 / 头像 (5 种)": [
        (512, 512), (768, 768), (1024, 1024), (1280, 1280), (1536, 1536),
    ],
}

DEFAULT_ASPECTS = "1:1, 3:2, 2:3, 4:3, 3:4, 16:9, 9:16, 21:9, 9:21"

DEFAULT_CUSTOM = """\
# 每行一个，支持 1024x1024 / 1024*1024 / 1024,1024 / 16:9 混写
1024x1024
1152x896
896x1152
1344x768
768x1344
"""

# 数字 + 分隔符 + 数字：'x * × , ，' 视为宽高，':' '：' 视为宽高比
_TOKEN_RE = re.compile(r"(\d{1,5})\s*([x×X*,，]|[:：])\s*(\d{1,5})")
_SPLIT_RE = re.compile(r"[\s,，;；|]+")


# --------------------------------------------------------------------------- #
# 尺寸工具
# --------------------------------------------------------------------------- #

def clamp_side(value: int, multiple: int = 1) -> int:
    value = int(value)
    if multiple > 1:
        value = int(round(value / multiple)) * multiple
    return max(MIN_SIDE, min(MAX_SIDE, value))


def size_from_aspect(
    aw: float,
    ah: float,
    megapixels: float = 1.0,
    multiple_of: int = 64,
) -> Tuple[int, int]:
    """按「目标像素 + 宽高比」反推一对尺寸，并对齐到 multiple_of。"""
    aw = float(aw) if aw else 1.0
    ah = float(ah) if ah else 1.0
    ratio = max(1e-6, aw / ah)
    total = max(0.01, float(megapixels)) * 1_000_000.0
    width = math.sqrt(total * ratio)
    height = math.sqrt(total / ratio)
    return (
        clamp_side(width, multiple_of),
        clamp_side(height, multiple_of),
    )


def _gcd_ratio(width: int, height: int, limit: int = 30) -> str:
    """把尺寸还原成可读的宽高比，例如 1152x896 -> 9:7。"""
    a, b = int(width), int(height)
    while b:
        a, b = b, a % b
    a = a or 1
    w, h = width // a, height // a
    if w > limit or h > limit:
        return f"{width / height:.2f}:1"
    return f"{w}:{h}"


def describe_size(width: int, height: int) -> str:
    return f"{width}×{height}  {width * height / 1e6:.2f}MP  {_gcd_ratio(width, height)}"


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #

def _normalize_size(
    width: int,
    height: int,
    notes: List[str],
    raw: str,
) -> Optional[Tuple[int, int]]:
    """校验并（必要时）对齐到 8 的倍数。"""
    if width < 16 or height < 16:
        notes.append(f"「{raw}」尺寸过小，已忽略")
        return None
    if width % LATENT_STEP or height % LATENT_STEP:
        snapped = (
            clamp_side(round(width / LATENT_STEP) * LATENT_STEP),
            clamp_side(round(height / LATENT_STEP) * LATENT_STEP),
        )
        notes.append(f"「{raw}」不是 8 的倍数，已对齐为 {snapped[0]}×{snapped[1]}")
        width, height = snapped
    if width > MAX_SIDE or height > MAX_SIDE:
        notes.append(f"「{raw}」超过 {MAX_SIDE}，已截断")
        width, height = clamp_side(width), clamp_side(height)
    return width, height


def parse_custom_list(
    text: str,
    megapixels: float = 1.0,
    multiple_of: int = 64,
) -> Tuple[List[Tuple[int, int]], List[str]]:
    """解析自定义分辨率列表，返回 (尺寸列表, 提示信息)。"""
    sizes: List[Tuple[int, int]] = []
    notes: List[str] = []
    text = (text or "").strip()
    if not text:
        return sizes, ["自定义列表为空"]

    cleaned = "\n".join(line.split("#", 1)[0] for line in text.splitlines())

    for match in _TOKEN_RE.finditer(cleaned):
        first, sep, second = int(match.group(1)), match.group(2), int(match.group(3))
        if sep in ":：":
            size = size_from_aspect(first, second, megapixels, multiple_of)
            size = _normalize_size(size[0], size[1], notes, match.group(0))
        else:
            size = _normalize_size(first, second, notes, match.group(0))
        if size and size not in sizes:
            sizes.append(size)

    leftover = _SPLIT_RE.split(_TOKEN_RE.sub(" ", cleaned))
    leftovers = [t for t in leftover if t]
    if leftovers:
        notes.append("已忽略无法识别的条目：" + "、".join(leftovers[:8]))

    if not sizes:
        notes.append("自定义列表里没有解析到任何可用分辨率")
    return sizes, notes


def parse_aspects(text: str) -> Tuple[List[Tuple[float, float]], List[str]]:
    """解析宽高比列表，支持 ``16:9`` / ``16/9`` / ``1.85``。"""
    ratios: List[Tuple[float, float]] = []
    notes: List[str] = []
    for token in _SPLIT_RE.split(text or ""):
        if not token:
            continue
        if ":" in token or "：" in token or "/" in token:
            parts = re.split(r"[:：/]", token)
            if len(parts) == 2:
                try:
                    aw, ah = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                if aw > 0 and ah > 0:
                    ratios.append((aw, ah))
        else:
            try:
                value = float(token)
            except ValueError:
                continue
            if value > 0:
                ratios.append((value * 100.0, 100.0))
    if not ratios:
        notes.append(f"宽高比列表为空或无法识别，已回退到默认值：{DEFAULT_ASPECTS}")
        ratios = [(1, 1), (3, 2), (2, 3), (16, 9), (9, 16)]
    return ratios, notes


def _canon(value: str, options: Sequence[str], fallback: str) -> str:
    value = str(value or "")
    if value in options:
        return value
    for option in options:
        if value and value.startswith(option.split(" ")[0]):
            return option
    return fallback


# --------------------------------------------------------------------------- #
# 组装分辨率池
# --------------------------------------------------------------------------- #

def build_pool(
    source: str = POOL_SOURCE_PRESET,
    preset_group: str = "",
    custom_text: str = "",
    megapixels: float = 1.0,
    aspect_text: str = DEFAULT_ASPECTS,
    multiple_of: int = 64,
) -> Tuple[List[Tuple[int, int]], List[str]]:
    """按参数组装候选分辨率池，返回 (尺寸列表, 提示信息)。"""
    source = _canon(source, POOL_SOURCES, POOL_SOURCE_PRESET)
    notes: List[str] = []
    sizes: List[Tuple[int, int]] = []

    def add_preset() -> None:
        group = PRESET_GROUPS.get(preset_group)
        if group is None:
            group = next(iter(PRESET_GROUPS.values()))
            notes.append(f"未找到预设「{preset_group}」，已回退到「SDXL 标准桶」")
        sizes.extend(group)

    def add_custom() -> None:
        parsed, custom_notes = parse_custom_list(custom_text, megapixels, multiple_of)
        sizes.extend(parsed)
        notes.extend(custom_notes)

    if source == POOL_SOURCE_PRESET:
        add_preset()
    elif source == POOL_SOURCE_CUSTOM:
        add_custom()
    elif source == POOL_SOURCE_MERGE:
        add_preset()
        add_custom()
    else:  # 按目标像素生成
        ratios, ratio_notes = parse_aspects(aspect_text)
        notes.extend(ratio_notes)
        for aw, ah in ratios:
            size = size_from_aspect(aw, ah, megapixels, multiple_of)
            if size not in sizes:
                sizes.append(size)

    # 去重 + 兜底
    deduped: List[Tuple[int, int]] = []
    for size in sizes:
        if size not in deduped:
            deduped.append(size)
    if not deduped:
        deduped = list(PRESET_GROUPS["SDXL 标准桶 · 1MP (9 种)"])
        notes.append("分辨率池为空，已回退到 SDXL 标准桶")

    return deduped, notes


def describe_pool(
    sizes: Sequence[Tuple[int, int]],
    per_line: int = 3,
    source_label: str = "",
) -> str:
    """把分辨率池渲染成可读文本（用于预览节点）。"""
    if not sizes:
        return "（空）"
    head = f"分辨率池：{len(sizes)} 种"
    if source_label:
        head += f"  ·  来源：{source_label}"
    lines = [head, "-" * 34]
    row: List[str] = []
    for index, (width, height) in enumerate(sizes, 1):
        row.append(f"{index:>2}. {describe_size(width, height)}")
        if len(row) == per_line:
            lines.append("   ".join(row))
            row = []
    if row:
        lines.append("   ".join(row))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 调度状态机
# --------------------------------------------------------------------------- #

_STATES: Dict[str, dict] = {}


def clear_states(key: Optional[str] = None) -> None:
    """清空调度状态；key 为空时清空全部。"""
    if key is None:
        _STATES.clear()
    else:
        _STATES.pop(str(key), None)


def _get_state(key: str, signature) -> dict:
    state = _STATES.get(key)
    if state is None or state.get("signature") != signature:
        state = {
            "signature": signature,
            "count": 0,     # 累计执行次数（从起始值开始）
            "bag": [],      # 洗牌模式的待取序列
            "last": -1,     # 上一次取到的下标，用于避免洗牌边界重复
            "epoch": 0,     # 洗牌轮次
        }
        _STATES[key] = state
    return state


def next_resolution(
    key: str,
    signature,
    pick_mode: str,
    sizes: Sequence[Tuple[int, int]],
    start_index: int = 0,
    seed: int = 0,
) -> Tuple[int, int, int, int]:
    """取出「本次执行」的分辨率。

    返回 ``(width, height, 第几次执行, 池内下标)``。
    """
    if not sizes:
        raise ValueError("分辨率池为空")

    pick_mode = _canon(pick_mode, PICK_MODES, PICK_SEQUENTIAL)
    state = _get_state(str(key), signature)
    count = state["count"]
    size_count = len(sizes)

    if pick_mode == PICK_SEQUENTIAL:
        index = (int(start_index) + count) % size_count
    elif pick_mode == PICK_RANDOM:
        # 用 seed + 执行次数派生随机源：结果可复现，且每次排队都不同
        rng = random.Random(
            (int(seed) * 6364136223846793005 + count * 1442695040888963407) & ((1 << 64) - 1)
        )
        index = rng.randrange(size_count)
    else:  # 洗牌不重复
        if not state["bag"]:
            rng = random.Random(
                (int(seed) * 1000003 + state["epoch"] * 97 + 1) & ((1 << 64) - 1)
            )
            order = list(range(size_count))
            rng.shuffle(order)
            # 让新一轮的第一个尽量不等于上一轮的最后一个
            if size_count > 1 and order[0] == state["last"]:
                order.append(order.pop(0))
            state["bag"] = order
            state["epoch"] += 1
        index = state["bag"].pop(0)
        state["last"] = index

    state["count"] = count + 1
    width, height = sizes[index]
    return int(width), int(height), count + 1, int(index)
