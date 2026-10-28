# Project state: FedECS-LT paper improvement loop

Started 29 Oct 2026 at the user's request (multi-agent review, revise, verify; at most 6 cycles before a stopping
assessment). The orchestrator is the main Claude Code session; reviewers are real subagents (Agent tool, general-purpose
type, run in the background in parallel). Every reviewer reads the same declared manuscript version.

| Field | Value |
|---|---|
| WORKFLOW_STATE | INDEPENDENT_REVIEW |
| Current manuscript version | v00 (git 1e6ffae; snapshot `manuscript_v00_original/`, read-only) |
| Authoritative manuscript | `docs/papers/content/*.tex` (+ slots from `docs/papers/fill.py`); versions = git commits listed in `revision_log.md` |
| Cycle | 1 |
| Agents assigned | A, B, C, D, E, F, G (independent); H after them; I after revision; J at the final audit |
| Agents completed | none |
| Open P0 / P1 | not yet reviewed |
| Next action | collect the seven reviews of v00 |

## Fixed rules (CLAUDE.md, never overridden by a reviewer)
- No result is typed into the paper: numbers enter only through `fill.py` slots from `fedecs_lt/results/`.
- Claims listed as false in `docs/review/adversarial_review_dossier.md` are not reintroduced.
- Experiments not yet returned (folder 7: grouped baselines, 50 clients, pooled reference; ablations NB30/NB31; sweeps
  NB40) cannot be fabricated: issues needing them are BLOCKED until the results come back.
- The user's decisions (docs/review/action_plan.md, sections 22-25) stand unless the user changes them.
