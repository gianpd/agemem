  ---
  Coherence Analysis: Evaluation Pipeline vs Production Orchestrator

  Executive Summary

  The evaluation pipeline tests only the LTM semantic search component in isolation. It bypasses 11 production subsystems that influence real-world memory behavior. The benchmark results measure raw retrieval quality but do not validate the end-to-end memory lifecycle.

  ---
  1. Module Coverage Matrix

  ┌───────────────────────────────────┬──────────────────────────┬───────────────────────┬─────────────┬───────────┐
  │              Module               │     Production Path      │    Evaluation Path    │ Shared Code │  Status   │
  ├───────────────────────────────────┼──────────────────────────┼───────────────────────┼─────────────┼───────────┤
  │ memory/ltm_store.py               │ ✅ LTMStore              │ ✅ LTMStore           │ ✅          │ COHERENT  │
  ├───────────────────────────────────┼──────────────────────────┼───────────────────────┼─────────────┼───────────┤
  │ memory/embedding.py               │ ✅                       │ ✅                    │ ✅          │ COHERENT  │
  ├───────────────────────────────────┼──────────────────────────┼───────────────────────┼─────────────┼───────────┤
  │ memory/vector_index.py            │ ✅                       │ ✅                    │ ✅          │ COHERENT  │
  ├───────────────────────────────────┼──────────────────────────┼───────────────────────┼─────────────┼───────────┤
  │ core/config.py                    │ ✅ AgememConfig          │ ✅ AgememConfig       │ ✅          │ COHERENT  │
  ├───────────────────────────────────┼──────────────────────────┼───────────────────────┼─────────────┼───────────┤
  │ core/types.py                     │ ✅ MemoryEntry           │ ✅ MemoryEntry        │ ✅          │ COHERENT  │
  ├───────────────────────────────────┼──────────────────────────┼───────────────────────┼─────────────┼───────────┤
  │ tools/query_expansion.py          │ ✅ (if configured)       │ ❌ Not passed         │ ⚠️          │ OMITTED   │
  ├───────────────────────────────────┼──────────────────────────┼───────────────────────┼─────────────┼───────────┤
  │ memory/context_retrieval.py       │ ✅ ContextAwareRetriever │ ❌ Direct search      │ ⚠️          │ BYPASSED  │
  ├───────────────────────────────────┼──────────────────────────┼───────────────────────┼─────────────┼───────────┤
  │ memory/stm_context.py             │ ✅ STMContext            │ ❌                    │ ❌          │ OMITTED   │
  ├───────────────────────────────────┼──────────────────────────┼───────────────────────┼─────────────┼───────────┤
  │ agents/orchestrator.py            │ ✅ Orchestrator          │ ❌                    │ ❌          │ OMITTED   │
  ├───────────────────────────────────┼──────────────────────────┼───────────────────────┼─────────────┼───────────┤
  │ agents/learning_scorer.py         │ ✅                       │ ❌ Uses preset scores │ ❌          │ OMITTED   │
  ├───────────────────────────────────┼──────────────────────────┼───────────────────────┼─────────────┼───────────┤
  │ agents/memory_agent.py            │ ✅                       │ ❌                    │ ❌          │ OMITTED   │
  ├───────────────────────────────────┼──────────────────────────┼───────────────────────┼─────────────┼───────────┤
  │ agents/response_handler.py        │ ✅                       │ ❌                    │ ❌          │ OMITTED   │
  ├───────────────────────────────────┼──────────────────────────┼───────────────────────┼─────────────┼───────────┤
  │ triggers/system_rules.py          │ ✅                       │ ❌                    │ ❌          │ OMITTED   │
  ├───────────────────────────────────┼──────────────────────────┼───────────────────────┼─────────────┼───────────┤
  │ triggers/memory_trigger_engine.py │ ✅                       │ ❌                    │ ❌          │ OMITTED   │
  ├───────────────────────────────────┼──────────────────────────┼───────────────────────┼─────────────┼───────────┤
  │ skills/manager.py                 │ ✅ SkillManager          │ ❌                    │ ❌          │ OMITTED   │
  ├───────────────────────────────────┼──────────────────────────┼───────────────────────┼─────────────┼───────────┤
  │ core/tracing.py                   │ ✅                       │ ❌ Separate trace DB  │ ⚠️          │ DIVERGENT │
  └───────────────────────────────────┴──────────────────────────┴───────────────────────┴─────────────┴───────────┘

  ---
  2. Execution Flow Comparison

  Production Turn Lifecycle (orchestrator.py:813-1149):
  1a. STM.force_fit()           → Overflow guard
  1b. ContextAwareRetriever     → Context-aware LTM search
      OR ltm.search()           → Fallback direct search
  1c. SkillManager.detect()     → Inject skill hints
  2.  LLM.chat_with_recovery()  → Main inference with tools
  3a. LearningScorer.collect()  → Self-assessment (every N turns)
  3b. MemoryTriggerEngine       → Unified trigger processing
      ├── SystemRules.evaluate()
      ├── MemoryAgent.review()
      └── Execute ADD/UPDATE/DELETE

  Evaluation Query Execution (inference_pipeline.py:344-393):
  1. ltm_store.search()         → Direct search only
  2. trace_search()             → Record results

  ---
  3. Critical Divergence: Query Expansion

  Production (orchestrator.py:167-173):
  ltm_store = LTMStore(
      config=config,
      persist_path=ltm_path,
      semantic_db_path=semantic_db_path,
      enable_semantic_search=config.ENABLE_SEMANTIC_SEARCH,
      llm_client=self._llm,  # ← Enables QueryExpansion
  )

  Evaluation (reproducible_runner.py:256-261):
  ltm_store = LTMStore(
      config=config,
      persist_path=ltm_path,
      semantic_db_path=semantic_db_path,
      enable_semantic_search=(mode == "semantic"),
      # llm_client NOT passed → QueryExpansion DISABLED
  )

  Impact: If ENABLE_QUERY_EXPANSION=True in production, queries get expanded into variants for better recall. Evaluation tests raw single-query semantic search.

  ---
  4. Critical Divergence: Context-Aware Retrieval

  Production (orchestrator.py:845-854):
  if self._context_retriever is not None:
      relevant = self._context_retriever.retrieve(
          current_query=user_input,
          recent_messages=self._stm.messages(),
          current_turn=self._stm.current_turn(),
          top_k=5,
      )
  else:
      relevant = self._ltm.search(user_input, top_k=5)

  Evaluation (inference_pipeline.py:369):
  results = ltm_store.search(query.query_text, top_k=top_k)

  Impact: Production uses conversation context to inform retrieval. Evaluation tests query-in-isolation retrieval.

  ---
  5. Memory Operations Not Validated

  ┌───────────────────────┬────────────────────────────────────────────────┬─────────────────────┐
  │       Operation       │               Production Trigger               │ Evaluation Coverage │
  ├───────────────────────┼────────────────────────────────────────────────┼─────────────────────┤
  │ ADD                   │ MemoryAgent.review(), force_memory_persistence │ ❌ Not tested       │
  ├───────────────────────┼────────────────────────────────────────────────┼─────────────────────┤
  │ UPDATE                │ MemoryAgent on content similarity              │ ❌ Not tested       │
  ├───────────────────────┼────────────────────────────────────────────────┼─────────────────────┤
  │ DELETE                │ SystemRules overflow, MemoryAgent pruning      │ ❌ Not tested       │
  ├───────────────────────┼────────────────────────────────────────────────┼─────────────────────┤
  │ Learning Score Update │ LearningScorer feedback loop                   │ ❌ Not tested       │
  ├───────────────────────┼────────────────────────────────────────────────┼─────────────────────┤
  │ STM Overflow Guard    │ force_fit() on every turn                      │ ❌ Not tested       │
  ├───────────────────────┼────────────────────────────────────────────────┼─────────────────────┤
  │ Skill Injection       │ SkillManager.detect_skills()                   │ ❌ Not tested       │
  └───────────────────────┴────────────────────────────────────────────────┴─────────────────────┘

  ---
  6. Representative Validity Assessment

  What the benchmark validates:
  - Semantic embedding quality (if mode="semantic")
  - Vector index recall/precision
  - LTMStore.search() ranking behavior
  - Hybrid scoring formula (recency + learning_score)

  What the benchmark does NOT validate:
  - Memory persistence in conversational context
  - Trigger-based memory operations
  - Learning score evolution over turns
  - STM overflow management
  - Skill-based context enrichment
  - Query expansion effectiveness
  - Context-aware retrieval decisions
  - Memory agent decision-making (ADD/UPDATE/DELETE)

  ---
  7. Verdict

  Coverage: ~20% of production memory subsystem

  Representativeness: The benchmark measures raw semantic retrieval quality in isolation. It is a valid test of the LTM search component but does not validate the end-to-end memory lifecycle that users experience.

  Gap Severity:
  - High impact: Query expansion bypass, context-aware retrieval bypass
  - Medium impact: No memory operations testing, no learning score dynamics
  - Low impact: No STM testing (evaluation intentionally focuses on LTM)

  Recommendation: The benchmark results should be interpreted as "LTM semantic search quality" metrics, not as "memory system effectiveness" metrics. To claim representativeness, additional test phases would need to validate:
  1. Memory operation triggers (ADD/UPDATE/DELETE)
  2. Learning score evolution
  3. Context-aware retrieval effectiveness
  4. Query expansion contribution to recall