Exact string replacement in the file at `file_path`. You must `read_file` first. When editing text from `read_file` output, never include any part of the line-number prefix in `old_string` or `new_string`. If an edit fails, re-read the file before retrying.

Modes (only these combinations are valid):
- Default: `old_string`/`new_string` replace the single occurrence of `old_string`; more than one occurrence is an error, so provide more context to make the anchor unique.
- `replace_all`: replace every non-overlapping occurrence of `old_string`.
- `replace_all` plus `expected_site_count`: replace every occurrence, but only when the anchor matches exactly that many sites; any other count writes nothing and returns a count-mismatch error naming the first sites by line number.
- `multi_anchor`: an ordered list of `old_string`/`new_string` pairs, passed instead of `old_string`/`new_string`. Each anchor must match exactly once, the anchors must be listed in file order, and they must not overlap. All anchors are validated before anything is written and the assembled result lands in one atomic write, so a failure at any anchor leaves the file untouched.

`expected_site_count` without `replace_all` is a parameter error, and `multi_anchor` cannot be combined with `replace_all=true` or `expected_site_count`.

A successful result reports the applied change and the post-edit file state: `applied_diff` shows each replaced site as whole lines (old, then new) under its current line number, `context` is a numbered window of the file's current content around the first site with current line numbers, `replaced_count` is how many sites were replaced, and `revision` identifies the post-edit file state — compare it with a `read_file` `revision` to detect a concurrent change. Client-mediated writes (remote sessions) report no `revision`.
