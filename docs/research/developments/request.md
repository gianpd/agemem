Ok, now let's push this a little further to something really usefull. Let's help us to define a strategical code developments plan, which companies can really use. 



## Project Overview

AgeMem-Hybrid is an inference-only memory management system for LLM agents, implementing unified long-term memory (LTM) and short-term memory (STM) management. It is a principled adaptation of the AgeMem paper (Yu et al., 2026) that compensates for the lack of RL training through a three-layer hybrid control architecture.



## Task

The LTM features need to be tested and verified. The approach MUST be real: an agent must interact with the system by starting the main.py, asking questions to the real agent, and verify if ALL the LTM Rules are executed correctly: is LTM stored after 10 chat messages ? is LTM stored when an agent get an important info ? 



## Procedure

The claude code agent must interact with the real system agent

The claude code agent must fix the code if and when it understand the bug

A sub agent must create the unit test which must be tailored to investigate the initial problem the claude code fixed (specific description of the initial problem statements)



## HINT and TECH STACK

Must use progress.md and init.sh Anthropic suggestion finding procedure.

Must use sub-agents efficiently for sub-tasks

Must have a PLAN DAG to follow. 

MUst make use of .claude/skills for specific tasks: name: refactor-with-tests for example





This strategy will be easly integrated into a real-production workflow and re-used for different tasks in future. 



## Expected output and acceptance criteria







Provide an initial prompt which ask the first agent session to plan the road map (the PLAN DAG)



Provide the prompt which instruct the agent to start a progress.md init.sh and further atomic tasks session, using the previous road-map DAG



Provide the mechanism/process which allows the agent session to correctly make use of SKILLS and sub-agents and progress.md in an autonomous way



### Acceptance Criteria

1.  **Interactive Planning:** The Admin must be able to define and validate the DAG workflow through a human-in-the-loop process.

2.  **Autonomous Execution:** Once initiated, the system must operate without further Admin intervention, allowing the user to disengage while agents execute the plan.

3.  **Reliable Completion:** The DAG must complete successfully within the expected timeframe, resolving all edge cases without errors.

4.  **Full Auditability:** All decision points, logic paths, and progress updates must be clearly visible and traceable within `progress.md`.

