Execute a shell command and return its output.

- Prefer absolute paths for command arguments, except redirect targets, which must be literal workspace-relative paths. Shell state — working directory, environment variables, functions — does NOT persist between calls; each call starts a fresh shell from the user's profile.
- Prefer the dedicated tools over shell utilities: use `read_file` instead of `cat`/`head`/`tail`, `grep` instead of `grep`/`sed`/`awk` for searching, and `edit`/`write_file` instead of `sed`/`echo` redirects. Only fall back to the shell utility if a dedicated tool genuinely cannot do the task.
- Commands run under a fixed allow/deny policy; there is no approval mechanism. A policy denial is not a user refusal. Adjust your approach and do not retry the same denied command verbatim.
- Environment prefixes are limited to safe uppercase names with literal values. General variable expansion and heredocs are unsupported in v0.1 because they can change argument count or meaning.
- Destructive commands are denied by policy. Ask the user to run them manually; do not attempt to bypass the guard.
- A runtime denial is not lifted by conversational approval: the user saying "yes, go ahead" does not change the decision. Report the restriction and offer the exact command for the user to run themselves or an intentional user configuration change. Never retry the denied command, split or disguise it, or edit configuration to get around a denial.
- Denials come from the policy layer, and the remediation depends on which part denied the command. The default denylist — `git push`, `git checkout`, `git stash drop`, `git stash clear`, `git restore`, `git switch --discard-changes`, `git switch -f`, `git reflog expire`, and `git reflog delete` — is user-removable via `denylist` under `[tools.bash]`: the user runs the command or intentionally removes the denylist entry. Hard guards — `git reset --hard` and forced non-dry-run `git clean` — are enforced by the policy layer and are not removable by configuration or approval: the user runs the command. Fail-closed spellings such as `git -c <name>=<value> ...` and `git --config-env ...` are denied by the policy layer itself, not by the denylist: removing denylist entries does not help, and the only remediation is the user running the command. Never re-spell the command to route around a denial.
- `timeout` is in seconds (default 300). This tool runs foreground work and kills commands that exceed the timeout. For managed long-running jobs use `bash_start`, read output with `bash_read`, recover handles with `bash_list`, and stop unneeded jobs with `bash_stop`.

# Git
- Interactive flags (`-i`, e.g. `git rebase -i`, `git add -i`) are not supported in this environment.
- Use the `gh` CLI for GitHub operations (PRs, issues, API).
- Commit or push only when the user asks. If you are on the default branch, create a branch first.
- `git push`, `git checkout`, `git stash drop`, `git stash clear`, `git restore`, `git switch --discard-changes`, `git switch -f`, `git reflog expire`, and `git reflog delete` are denied by the default denylist: report the restriction and hand the exact command to the user instead of retrying it.
- Do not append your own commit or PR footer.
