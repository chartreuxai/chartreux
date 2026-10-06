Start a session-owned managed shell job and return its summary and next_cursor=0.

- Use bash for finite foreground work; use bash_start for a long-running command while continuing other work. Keep the target process in the foreground inside the managed command; do not detach it.
- Existing bash shell restrictions and canonical tools.bash policy still apply. No launch-time cwd or env controls are accepted.
- Stdin is closed and there is no PTY: interactive programs are unsupported. Output is captured locally, with stdout and stderr merged.
- Startup success is not readiness. Use bash_read to observe application readiness, forwarding next_cursor on each read.
- Prefer bounded waits of 5-15 seconds over busy-polling (30-second cap); waits delay queued prompts. Use modest read budgets and avoid re-reading unchanged windows: every read grows the transcript independently of registry retention.
- Stop unneeded jobs with bash_stop. Jobs survive ordinary turns and compaction, but are lost on session end or replacement. Output may be evicted; read reports lost_records.
- After compaction, recover job handles via bash_list.
