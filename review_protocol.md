# Cross-Agent Report Review Protocol

Two agents (Claude and Codex) independently generate comparison reports, then cross-review and self-revise over up to two rounds. You act as the message relay.

---

## Roles

| Role | Agent | Session |
|---|---|---|
| Agent A | Claude (Claude.ai) | Keep one persistent conversation per review cycle |
| Agent B | Codex (Codex.com) | Keep one persistent conversation per review cycle |

Start fresh sessions for each new ticker or date range. Within one cycle, stay in the same session so each agent accumulates context across rounds.

---

## File Naming Convention

```
reports/<TICKER>_<DATE>_<N>_report_comparison_by_claude.md
reports/<TICKER>_<DATE>_<N>_report_comparison_by_codex.md
reports/<TICKER>_<DATE>_<N>_review_of_codex_by_claude.md
reports/<TICKER>_<DATE>_<N>_review_of_claude_by_codex.md
```

`<N>` = number of HTML reports being compared.

---

## Round Structure

```
Round 0  →  Both agents write initial report independently
Round 1  →  Each agent reviews the other's report
Round 2  →  Each agent revises its own report based on feedback received
(optional)
Round 3  →  Second cross-review (only if scores still diverge > 1.0)
Round 4  →  Second self-revision
```

Stop after Round 2 unless scores disagree materially. Two review rounds is usually enough for convergence on findings; ranking disagreements that persist after Round 2 are legitimate differences in weighting, not errors.

---

## Step 0 — Initial Report (both agents, independently)

Paste this prompt into each agent's session. Do **not** share the reports with each other at this step.

```
Compare the quality of these N trading analysis reports section by section.

Reports to compare:
- reports/<TICKER>_<DATE1>_<HHMMSS>/<TICKER>_<DATE1>_<HHMMSS>_complete_report.html
- reports/<TICKER>_<DATE2>_<HHMMSS>/<TICKER>_<DATE2>_<HHMMSS>_complete_report.html
[add remaining reports]

Instructions:
1. Read each report fully before scoring.
2. Write a Run Configuration table including: report label, generated timestamp,
   analysis date, provider, model ID, research depth, thinking preset,
   effective thinking agents (accounting for structured-output suppression),
   backend URL, total elapsed time, LLM generation time, word count.
3. Score each section below on a 1–10 integer scale for all reports.
   Provide one paragraph of rationale per section explaining score differences.
4. Write an Overall Ranking table with total score and one-line reason.
5. Write a Verdict section (3–5 bullet points) summarising the key findings
   across all reports.

Sections to score (score each report):
- Run Metadata / CLI Context
- Market / Technical Analyst
- Social / Sentiment Analyst
- News / Macro Analyst
- Fundamentals Analyst
- Bull/Bear Research Debate
- Research Manager
- Trader Plan
- Risk Debate
- Portfolio Manager Decision
- Action Summary
- Cross-Section Consistency
- Formatting / Readability

Output: a single markdown document. Save it to
reports/<TICKER>_<DATE>_<N>_report_comparison_by_<agent>.md
```

---

## Step 1 — Cross-Review (Round 1)

After both agents have written their initial report, copy each report into the other agent's session.

**Prompt to send to Agent A (Claude), attaching Agent B's report:**

```
Here is Codex's comparison report:

[paste full content of codex report]

Review it against your own report. For each finding where you disagree
or think the assessment is incomplete:

1. Quote the specific claim from Codex's report.
2. State what is wrong or missing.
3. State what the correct assessment is and why.

Also note any findings in Codex's report that are correct and that
your own report missed or understated.

Structure your response as a numbered list of findings. Do not rewrite
your report yet — just produce the findings list.
```

**Prompt to send to Agent B (Codex), attaching Agent A's report:**

```
Here is Claude's comparison report:

[paste full content of claude report]

Review it against your own report. For each finding where you disagree
or think the assessment is incomplete:

1. Quote the specific claim from Claude's report.
2. State what is wrong or missing.
3. State what the correct assessment is and why.

Also note any findings in Claude's report that are correct and that
your own report missed or understated.

Structure your response as a numbered list of findings. Do not rewrite
your report yet — just produce the findings list.
```

Save each agent's findings list as a file:
```
reports/<TICKER>_<DATE>_<N>_review_of_codex_by_claude.md
reports/<TICKER>_<DATE>_<N>_review_of_claude_by_codex.md
```

---

## Step 2 — Self-Revision (Round 1)

Copy each agent's received feedback into its own session.

**Prompt to send to Agent A (Claude), attaching Codex's review of Claude's report:**

```
Here is Codex's review of your report:

[paste full content of codex's review of claude's report]

Apply this feedback to revise your comparison report. For each finding:
- If you agree: apply the correction and note what changed.
- If you disagree: keep your original assessment and briefly explain why.

Do not change scores or rankings just to agree with Codex — only
change them if the reasoning reveals a genuine error in your analysis.

Output your revised report in full. Note a "Changes" section at the
end listing what you changed and why.
```

**Prompt to send to Agent B (Codex), attaching Claude's review of Codex's report:**

```
Here is Claude's review of your report:

[paste full content of claude's review of codex's report]

Apply this feedback to revise your comparison report. For each finding:
- If you agree: apply the correction and note what changed.
- If you disagree: keep your original assessment and briefly explain why.

Do not change scores or rankings just to agree with Claude — only
change them if the reasoning reveals a genuine error in your analysis.

Output your revised report in full. Note a "Changes" section at the
end listing what you changed and why.
```

---

## Termination Condition

After Round 2, compare the two revised reports:

| Condition | Action |
|---|---|
| Overall scores differ by ≤ 1.0 and ranking order agrees | Stop. Document the disagreements as legitimate weighting differences. |
| Overall scores differ by > 1.0 or ranking order disagrees on top 2 | Run one more cross-review + self-revision round (Rounds 3–4). |
| Scores still diverge after Round 4 | Stop. Write a joint summary noting the persistent disagreement and its cause. |

Do not run more than 2 full cross-review cycles. Persistent disagreement after that is usually a weighting difference (e.g., one agent penalises ATR-inconsistent stops more heavily), not an error in either report.

---

## Notes

**Context bleed:** Each agent accumulates the other's reasoning over rounds. By Round 2, outputs may converge toward the same phrasing even when the underlying assessment differs. Watch for this — a score change should be accompanied by new reasoning, not just agreement phrasing.

**Structured-output suppression:** When scoring reports that used local LLM providers (LM Studio, Ollama), remind both agents that `bind_structured()` suppresses thinking tokens for Research Manager, Trader, and Portfolio Manager regardless of the CLI thinking preset. Effective thinking only reaches free-prose agents (Bull, Bear, Risk Analysts). Both agents should reflect this in the "Effective thinking agents" row of the config table.

**ATR baseline:** For stop-placement scoring, the correct benchmark is 1.5–2× ATR below entry. Always extract the ATR value from the Market/Technical section of each report and compute the multiple explicitly. Do not score stops as "appropriate" based on dollar value alone.
