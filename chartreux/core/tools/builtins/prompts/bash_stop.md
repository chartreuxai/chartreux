Stop an unneeded managed shell job and return its final summary.

- Stop targets the owned process group, with a bounded termination grace followed by forced termination if needed. already_finished is true when the job was already finished before this request.
- Use bash_read with forwarded cursors to retrieve retained final output; output may be evicted. Prefer modest read budgets and bounded waits of 5-15 seconds (30-second cap), not busy-polling or re-reading unchanged windows that grow the transcript.
- Jobs have closed stdin and no PTY; targets stay foreground inside managed commands. Existing shell restrictions still apply; startup success does not imply readiness.
- Jobs survive ordinary turns and compaction but are lost on session end or replacement. After compaction, recover job handles via bash_list.
