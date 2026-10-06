Read a bounded page of merged output from a managed shell job.

- Start at cursor=0, then pass next_cursor forward. Independent readers may replay retained output; lost_records reports eviction. Output may be evicted even while a job is running.
- Use modest max_bytes budgets (4096-64000 UTF-8 output bytes). Avoid re-reading unchanged windows: each read persists in the transcript, whose aggregate growth is independent of bounded job retention.
- Prefer bounded waits of 5-15 seconds over busy-polling; wait_seconds has a 30-second cap and delays queued prompts. A timeout does not imply completion or readiness.
- Startup success is not readiness. Inspect application output; output_complete means no more records will arrive, and output_incomplete indicates capture loss, independently of cursor eviction.
- Commands run with closed stdin and no PTY; the target stays foreground inside the managed command. Existing shell restrictions still apply.
- Stop unneeded jobs with bash_stop. Jobs survive ordinary turns but are lost on session end or replacement. After compaction, recover job handles via bash_list.
