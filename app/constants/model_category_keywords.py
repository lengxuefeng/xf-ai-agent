# -*- coding: utf-8 -*-
"""模型分类关键词常量。"""
from enum import Enum
from typing import Dict, Tuple


class ModelCategory(str, Enum):
    """模型分类枚举。"""

    CHAT = "chat"
    EMBEDDING = "embedding"
    VISION = "vision"
    OTHER = "other"


MODEL_CATEGORY_KEYWORDS: Dict[ModelCategory, Tuple[str, ...]] = {
    ModelCategory.EMBEDDING: (
        "embedding",
        "embed",
        "bge-",
        "text-embedding",
        "e5-",
        "sentence-",
        "all-minilm",
        "all-mpnet",
        "gte-",
        "m3e-",
        "text2vec",
        "simcse",
        "sbert",
        "instructor",
        "multilingual-e5",
        "paraphrase-",
    ),
    ModelCategory.VISION: (
        "dall-e",
        "midjourney",
        "stable-diffusion",
        "image",
        "vision",
        "visual",
        "gpt-4-vision",
        "gpt-4v",
        "gemini-pro-vision",
        "gemini-vision",
        "cogvlm",
        "blip",
        "clip",
        "flamingo",
        "llava",
        "minigpt",
        "instructblip",
        "qwen-vl",
        "internvl",
        "yi-vl",
        "视觉",
        "图像",
        "看图",
        "识图",
    ),
    ModelCategory.CHAT: (
        "gpt",
        "claude",
        "gemini",
        "llama",
        "qwen",
        "chat",
        "pro",
        "turbo",
        "flash",
        "sonnet",
        "haiku",
        "opus",
        "mistral",
        "yi-",
        "baichuan",
        "chatglm",
        "vicuna",
        "alpaca",
        "wizard",
        "orca",
        "falcon",
        "mpt",
        "palm",
        "bard",
        "ernie",
        "wenxin",
        "tongyi",
        "spark",
    ),
}


EMBEDDING_EXACT_NAMES: Tuple[str, ...] = ("engtrng", "engtng1", "333", "zz", "1123")
CLAUDE_VISION_MARKERS: Tuple[str, ...] = ("claude-3", "vision")
DISTILBERT_EMBED_MARKERS: Tuple[str, ...] = ("distilbert", "embed")

