---
name: refactor-with-tests
description: Refactors a module and writes unit tests. Invoke when asked to refactor code.
allowed-tools: Read, Write, Bash(npm run test:*, uv run pytest:*)
---
When refactoring a module:
1. Read the existing code and all its callers
2. Identify the refactoring goal from the task
3. Make the refactor in small, compilable increments
4. After each step, run surgical tests targeting the modified module:
   - JS/TS: `npm run test -- --testPathPattern=<module>`
   - Python: `uv run pytest <module_path> -v`
5. Write new unit tests covering the changed behavior
6. Ensure all existing tests still pass
7. Commit: "refactor(<module>): <what changed>"