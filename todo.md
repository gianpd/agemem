**What the pipeline is doing:** Processing an Italian public procurement code (Codice degli Appalti, D.Lgs. 36/2023) through a 4-stage pipeline: table extraction → entity recognition (GLiNER) → markdown export → index update.

**The core issue — repeated truncation warnings**

The vast majority of these logs are the same warning repeated ~180+ times:

```
UserWarning: Sentence of length X has been truncated to 384
```

GLiNER's underlying transformer model has a hard token limit of 384, but sentences in the document regularly exceed this — some dramatically so (up to 4,870 characters in chunk 173). This means entity extraction is silently dropping the tail of any sentence beyond that limit, which could cause missed entities in long legal provisions.

**What to investigate / fix:**

The most impactful change would be implementing sentence splitting before passing text to GLiNER. Long legal sentences (common in Italian codices) should be broken at semicolons, commas, or conjunctions before NER inference, then entity spans re-mapped to original offsets.

You might also consider switching to a model with a longer context window (e.g., a longformer-based NER), or using a sliding window with overlap over long sentences rather than hard truncation.

**The `Asking to truncate to max_length but no maximum length` warning** is a secondary Hugging Face tokenizer misconfiguration — the tokenizer's `max_length` isn't being set explicitly, so it falls back to no truncation at the tokenizer level while GLiNER enforces its own limit downstream. You should pass `max_length=384` explicitly when initializing the tokenizer.

**The output looks healthy otherwise** — 278 chunks processed, 1,357 sections extracted, all 18 entity types populated (though `autorizzaciones` and `scadenze` only got 1 hit each, which may be worth reviewing for recall).