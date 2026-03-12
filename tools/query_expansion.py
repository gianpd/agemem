"""
tools/query_expansion.py
─────────────────────────
Query expansion for LTM retrieval.

Generates paraphrase variants of a query string using a local LLM
via the existing LLMClient. Falls back to regex-based expansion
if the LLM call fails or returns malformed output.

All expansion is done at query time, not at index time.
Zero schema changes required.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from agents.llm_client import LLMClient

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Generates paraphrase variants of a query string using a local LLM
    via the existing LLMClient. Falls back to regex-based expansion
    if the LLM call fails or returns malformed output.
    
    All expansion is done at query time, not at index time.
    Zero schema changes required.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str,
        n_variants: int = 3,
        use_ner_hints: bool = True,
        fallback_on_error: bool = True,
        timeout_ms: int = 2000,
        fallback_transforms: list[str] | None = None,
        acronym_dict: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the QueryExpander.

        Parameters
        ----------
        llm_client : LLMClient
            Existing LLM client for making expansion calls.
        model : str
            Model name to use for expansion.
        n_variants : int
            Total number of queries including original. Default: 3.
        use_ner_hints : bool
            Whether to inject NER entity types into the prompt.
        fallback_on_error : bool
            Whether to return [original] on LLM failure.
        timeout_ms : int
            Abort expansion if LLM takes longer than this.
        fallback_transforms : list[str] | None
            List of enabled fallback transform names.
        acronym_dict : dict[str, str] | None
            User-supplied acronym expansion dictionary.
        """
        self._llm = llm_client
        self._model = model
        self._n_variants = n_variants
        self._use_ner_hints = use_ner_hints
        self._fallback_on_error = fallback_on_error
        self._timeout_ms = timeout_ms
        self._fallback_transforms = fallback_transforms or ["nominalize", "add_how_to"]
        self._acronym_dict = acronym_dict or {}

    def expand(
        self,
        query: str,
        ner_entities: list[dict] | None = None,
    ) -> list[str]:
        """
        Generate paraphrase variants of the query.

        Returns [original_query, variant_1, ..., variant_N].
        First element is always the unmodified original.
        Never raises — returns [query] on any failure.

        Parameters
        ----------
        query : str
            The original search query.
        ner_entities : list[dict] | None
            Optional GLiNER output for the query.

        Returns
        -------
        list[str]
            List of query variants including the original.
        """
        if not query or not query.strip():
            return [query]

        # Always include the original query first
        variants = [query]

        try:
            # Try LLM-based expansion with timeout
            llm_variants = self._expand_with_llm(query, ner_entities)
            if llm_variants:
                variants.extend(llm_variants)
                return variants[:self._n_variants]
        except Exception as e:
            logger.debug(f"LLM expansion failed: {e}")

        # Fallback to regex-based expansion
        if self._fallback_on_error:
            try:
                fallback_variants = self._expand_with_regex(query)
                variants.extend(fallback_variants)
            except Exception as e:
                logger.debug(f"Fallback expansion failed: {e}")

        return variants[:self._n_variants]

    def _expand_with_llm(
        self,
        query: str,
        ner_entities: list[dict] | None = None,
    ) -> list[str]:
        """
        Generate variants using the LLM.

        Returns list of variants (not including original).
        Raises on failure if fallback_on_error is False.
        """
        # Build NER hints block
        ner_block = ""
        if self._use_ner_hints and ner_entities:
            entity_strs = []
            for entity in ner_entities:
                entity_type = entity.get("label", entity.get("type", "ENTITY"))
                entity_text = entity.get("text", entity.get("word", ""))
                if entity_text:
                    entity_strs.append(f'{entity_type}: "{entity_text}"')
            if entity_strs:
                ner_block = f"\nKnown entities in query: {', '.join(entity_strs)}"

        # Build prompts
        system_prompt = f"""You are a query expansion assistant for a document retrieval system.
Given a search query, generate {self._n_variants - 1} alternative phrasings that express
the same intent using different vocabulary. Focus on:
- Domain synonyms and technical aliases
- Nominalization variants (e.g. "throttle requests" → "request throttling")
- Abbreviation expansion and contraction
- Passive/active voice alternates for action queries

Return ONLY a JSON array of strings. No explanation. No markdown.
Example output: ["variant one", "variant two", "variant three"]"""

        user_prompt = f"""Query: {query}{ner_block}
Generate {self._n_variants - 1} alternative phrasings."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Make LLM call with timeout
        start_time = time.time()
        try:
            response = self._llm.chat(
                messages=messages,
                model=self._model,
                max_tokens=256,
                temperature=0.7,
                timeout=self._timeout_ms / 1000.0,
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms >= self._timeout_ms:
                logger.debug(f"LLM expansion timed out after {elapsed_ms:.0f}ms")
            raise

        # Parse response
        variants = self._parse_llm_response(response)
        return variants

    def _parse_llm_response(self, response: str) -> list[str]:
        """
        Parse LLM response to extract variants.

        Returns list of variant strings.
        Raises JSONParseError if parsing fails.
        """
        from agents.llm_client import extract_json, JSONParseError

        try:
            data = extract_json(response, repair=True)
            if isinstance(data, list):
                # Filter to only strings and deduplicate
                seen = set()
                variants = []
                for v in data:
                    if isinstance(v, str):
                        v_stripped = v.strip()
                        if v_stripped and v_stripped.lower() not in seen:
                            seen.add(v_stripped.lower())
                            variants.append(v_stripped)
                return variants
            elif isinstance(data, dict):
                # Some models return {"variants": [...]} or similar
                for key in ["variants", "phrasings", "alternatives", "queries"]:
                    if key in data and isinstance(data[key], list):
                        seen = set()
                        variants = []
                        for v in data[key]:
                            if isinstance(v, str):
                                v_stripped = v.strip()
                                if v_stripped and v_stripped.lower() not in seen:
                                    seen.add(v_stripped.lower())
                                    variants.append(v_stripped)
                        return variants
            raise JSONParseError(response, "expected JSON array of strings")
        except JSONParseError:
            # Try to extract array from text using regex
            array_match = re.search(r'\[([^\]]+)\]', response)
            if array_match:
                try:
                    # Try to parse the matched array
                    array_str = array_match.group(0)
                    data = json.loads(array_str)
                    if isinstance(data, list):
                        seen = set()
                        variants = []
                        for v in data:
                            if isinstance(v, str):
                                v_stripped = v.strip()
                                if v_stripped and v_stripped.lower() not in seen:
                                    seen.add(v_stripped.lower())
                                    variants.append(v_stripped)
                        return variants
                except json.JSONDecodeError:
                    pass
            raise

    def _expand_with_regex(self, query: str) -> list[str]:
        """
        Generate variants using deterministic regex transformations.

        Returns list of variants (not including original).
        """
        variants = []
        query_lower = query.lower().strip()

        # Nominalize: verb phrase → noun phrase
        if "nominalize" in self._fallback_transforms:
            nominalized = self._nominalize(query_lower)
            if nominalized and nominalized != query_lower:
                variants.append(nominalized)

        # Add "how to" prefix
        if "add_how_to" in self._fallback_transforms:
            if not query_lower.startswith("how to"):
                how_to_variant = f"how to {query_lower}"
                variants.append(how_to_variant)

        # Acronym expansion
        if self._acronym_dict:
            expanded = self._expand_acronyms(query)
            if expanded != query:
                variants.append(expanded)

        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            v_lower = v.lower().strip()
            if v_lower and v_lower not in seen and v_lower != query_lower:
                seen.add(v_lower)
                unique_variants.append(v)

        return unique_variants

    def _nominalize(self, query: str) -> str:
        """
        Convert verb phrases to noun phrases.

        Examples:
        - "authenticate user" → "user authentication"
        - "deploy container" → "container deployment"
        - "limit rate" → "rate limiting"
        """
        # Common verb → noun patterns
        verb_to_noun = {
            "authenticate": "authentication",
            "authorize": "authorization",
            "deploy": "deployment",
            "configure": "configuration",
            "optimize": "optimization",
            "validate": "validation",
            "initialize": "initialization",
            "terminate": "termination",
            "migrate": "migration",
            "integrate": "integration",
            "monitor": "monitoring",
            "scale": "scaling",
            "backup": "backup",
            "restore": "restoration",
            "encrypt": "encryption",
            "decrypt": "decryption",
            "compress": "compression",
            "decompress": "decompression",
            "serialize": "serialization",
            "deserialize": "deserialization",
            "cache": "caching",
            "index": "indexing",
            "search": "searching",
            "filter": "filtering",
            "sort": "sorting",
            "aggregate": "aggregation",
            "transform": "transformation",
            "parse": "parsing",
            "compile": "compilation",
            "execute": "execution",
            "schedule": "scheduling",
            "queue": "queuing",
            "throttle": "throttling",
            "limit": "limiting",
            "balance": "balancing",
            "route": "routing",
            "proxy": "proxying",
            "log": "logging",
            "debug": "debugging",
            "test": "testing",
            "build": "building",
            "release": "releasing",
            "rollback": "rollback",
            "update": "updating",
            "delete": "deletion",
            "create": "creation",
            "read": "reading",
            "write": "writing",
        }

        words = query.split()
        if len(words) < 2:
            return query

        # Check if first word is a verb
        first_word = words[0].lower()
        if first_word in verb_to_noun:
            noun = verb_to_noun[first_word]
            # Reorder: "verb object" → "object noun"
            return f"{' '.join(words[1:])} {noun}"

        # Check if last word is a verb (less common but possible)
        last_word = words[-1].lower()
        if last_word in verb_to_noun:
            noun = verb_to_noun[last_word]
            return f"{noun} {' '.join(words[:-1])}"

        return query

    def _expand_acronyms(self, query: str) -> str:
        """
        Expand acronyms in the query using the provided dictionary.

        Example: "API key" → "application programming interface key"
        """
        result = query
        for acronym, expansion in self._acronym_dict.items():
            # Case-insensitive replacement with word boundaries
            pattern = re.compile(re.escape(acronym), re.IGNORECASE)
            result = pattern.sub(expansion, result)
        return result