List accessible managed shell job summaries, without output bodies.

- By default list only unfinished jobs; include_finished=true includes retained finished jobs. Root contexts see all jobs, child contexts only their own.
- After compaction, recover job handles via bash_list. Jobs survive ordinary turns and compaction but are lost on session end or replacement; old handles do not restore jobs.
- Use bash_read with forwarded next_cursor values for output, which may be evicted. Prefer modest byte budgets and bounded waits of 5-15 seconds (30-second cap) over busy-polling. Avoid re-reading unchanged windows: every read grows the transcript independently of registry retention.
- Startup success is not readiness. Managed commands keep the target foreground, with closed stdin and no PTY; existing shell restrictions still apply.
- Stop unneeded jobs with bash_stop.
