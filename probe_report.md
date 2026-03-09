# LTM Probe Report

**Generated:** 2026-03-09T11:24:26.638694+00:00
**Total Turns:** 7
**Bugs Found:** 10

## Summary

| Metric | Value |
|--------|-------|
| Turns Simulated | 7 |
| Final LTM Entries | 0 |
| Total Bugs | 10 |
| Critical Bugs | 10 |

## Turn-by-Turn Observations

### Turn 1

**Input:** Hi, I'm Marco and I work as a civil engineer on bridge restoration projects....

**Response:** Hello! I'm your assistant. I can help with various tasks....

| Metric | Value |
|--------|-------|
| STM Utilization | 37.5% |
| STM Tokens | 749 |
| LTM Entries | 0 |
| Learning Score | N/A |
| Feedback Collected | No |

**Operations Applied:**

**Bugs Observed:**
- ⚠️ LTM-10 FAIL: No RETRIEVE operation on turn 1

---

### Turn 2

**Input:** I'm currently preparing a bid for a historic bridge restoration in Florence....

**Response:** I understand. Let me help you with that project....

| Metric | Value |
|--------|-------|
| STM Utilization | 38.6% |
| STM Tokens | 772 |
| LTM Entries | 0 |
| Learning Score | N/A |
| Feedback Collected | No |

**Operations Applied:**

**Bugs Observed:**
- ⚠️ LTM-10 FAIL: No RETRIEVE operation on turn 2

---

### Turn 3

**Input:** What are the SOA requirements for participating in public tenders in Italy?...

**Response:** That's interesting information about your work....

| Metric | Value |
|--------|-------|
| STM Utilization | 39.6% |
| STM Tokens | 793 |
| LTM Entries | 0 |
| Learning Score | N/A |
| Feedback Collected | No |

**Operations Applied:**

**Bugs Observed:**
- ⚠️ LTM-05 FAIL: Learning score should collect at turn 3 but returned None
- ⚠️ LTM-10 FAIL: No RETRIEVE operation on turn 3

---

### Turn 4

**Input:** I prefer working with reinforced concrete and have 15 years of experience....

**Response:** I've noted your preferences for future reference....

| Metric | Value |
|--------|-------|
| STM Utilization | 40.8% |
| STM Tokens | 815 |
| LTM Entries | 0 |
| Learning Score | N/A |
| Feedback Collected | No |

**Operations Applied:**

**Bugs Observed:**
- ⚠️ LTM-10 FAIL: No RETRIEVE operation on turn 4

---

### Turn 5

**Input:** The project deadline is tight - only 6 months for full restoration....

**Response:** Based on what you've shared, here's my recommendation....

| Metric | Value |
|--------|-------|
| STM Utilization | 41.9% |
| STM Tokens | 838 |
| LTM Entries | 0 |
| Learning Score | N/A |
| Feedback Collected | No |

**Operations Applied:**

**Bugs Observed:**
- ⚠️ LTM-10 FAIL: No RETRIEVE operation on turn 5

---

### Turn 6

**Input:** Can you help me research best practices for historic bridge preservation?...

**Response:** I remember you mentioned that earlier. Let me build on it....

| Metric | Value |
|--------|-------|
| STM Utilization | 44.2% |
| STM Tokens | 884 |
| LTM Entries | 0 |
| Learning Score | N/A |
| Feedback Collected | No |

**Operations Applied:**

**Bugs Observed:**
- ⚠️ LTM-05 FAIL: Learning score should collect at turn 6 but returned None
- ⚠️ LTM-10 FAIL: No RETRIEVE operation on turn 6

---

### Turn 7

**Input:** Please remember that I always work with the same team of 5 specialists....

**Response:** Let me recall what we discussed previously....

| Metric | Value |
|--------|-------|
| STM Utilization | 45.3% |
| STM Tokens | 906 |
| LTM Entries | 0 |
| Learning Score | N/A |
| Feedback Collected | No |

**Operations Applied:**

**Bugs Observed:**
- ⚠️ LTM-10 FAIL: No RETRIEVE operation on turn 7

---

## LTM Rules Status

| Rule ID | Status | Notes |
|---------|--------|-------|
| LTM-01 | PASS | |
| LTM-02 | PASS | |
| LTM-03 | NOT TESTED | |
| LTM-04 | NOT TESTED | |
| LTM-05 | FAIL | |
| LTM-06 | PASS | |
| LTM-07 | NOT TESTED | |
| LTM-08 | NOT TESTED | |
| LTM-09 | NOT TESTED | |
| LTM-10 | FAIL | |
| LTM-11 | NOT TESTED | |
| LTM-12 | FAIL | |
