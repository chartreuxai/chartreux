# Orchestration strategy: default

Packaging and routing preferences for delegated work. This strategy shapes how
launches are packaged and scheduled; it introduces no process stages and no
approval requirements.

## Brief discipline

- Give every launch a self-contained brief: the task, the scope it lands in, and any intervening changes since that agent last ran.
- Name the skills the child should load, so the methodology rides in the brief instead of being rediscovered.
- Keep briefs bounded: one assignment per launch; split work that exceeds a single bounded assignment.

## Parallelism

- Batch independent, non-conflicting launches into one parallel wave instead of serializing them.
- Keep conflicting work — same files, same subsystem — serial.
- Prefer a small number of well-briefed launches over many thin ones.

## Reuse

- Check the agent inventory before launching: re-task an idle agent for a genuine continuation, with the current scope and intervening changes in the brief.
- Reserve fresh launches for new or unrelated work.
- Close agents that will not be reused, so the roster stays honest.
