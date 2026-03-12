"""
tools package for AgeMem agent.

This package provides various tools for the agent including:
- Web search and fetch tools
- Corpus/document management tools
- Commodity price tools (Alpha Vantage integration)
"""

from .web_tools import (
    web_search,
    fetch_url,
    fetch_url_tool,
    write_file,
    ingest_document,
    register_conversation_urls,
    validate_url_for_fetch,
    tool_definitions as web_tool_definitions,
)

from .commodity_tool import (
    commodity_price,
    CommodityPriceTool,
    CommodityHistoryTool,
    GMEPriceTool,
    get_commodity_price_tool,
    get_commodity_history_tool,
    get_gme_price_tool,
    get_available_commodities,
    get_commodity_info,
    get_commodity_tool_definitions,
    get_tool_metrics,
    health_check,
)

from .alpha_api_translation import (
    CommodityTagsTranslator,
    TranslationResult,
    TranslationStatus,
    AlphaAPIConfig,
    COMMODITY_TAGS,
    COMMODITY_TAGS_ALPHA,
    get_commodity_translator,
    get_available_tags,
    get_tag_info,
    is_alpha_vantage_available,
)

__all__ = [
    # Web tools
    "web_search",
    "fetch_url",
    "fetch_url_tool",
    "write_file",
    "ingest_document",
    "register_conversation_urls",
    "validate_url_for_fetch",
    "web_tool_definitions",
    # Commodity tools
    "commodity_price",
    "CommodityPriceTool",
    "CommodityHistoryTool",
    "GMEPriceTool",
    "get_commodity_price_tool",
    "get_commodity_history_tool",
    "get_gme_price_tool",
    "get_available_commodities",
    "get_commodity_info",
    "get_commodity_tool_definitions",
    "get_tool_metrics",
    "health_check",
    # Alpha API translation layer
    "CommodityTagsTranslator",
    "TranslationResult",
    "TranslationStatus",
    "AlphaAPIConfig",
    "COMMODITY_TAGS",
    "COMMODITY_TAGS_ALPHA",
    "get_commodity_translator",
    "get_available_tags",
    "get_tag_info",
    "is_alpha_vantage_available",
]
