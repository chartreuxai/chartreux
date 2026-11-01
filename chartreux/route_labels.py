"""Display labels shared by every purpose-route surface (v0.7.0, O2).

``chartreux doctor``'s route section and the ``/routes`` browser report the same
facts about the same table, so both must call a preview a preview and an
unresolved count unknown with the same words. The literals live here rather
than in the inspection helper because the browser is a TUI surface: it may not
reach into ``chartreux.core``, directly or transitively.
"""

from __future__ import annotations

# Every preview is an estimate, and both surfaces say so with this literal.
PREVIEW_LABEL = "preview, not a launch guarantee"
# The diversity count when at least one target cannot be resolved.
UNKNOWN = "UNKNOWN"
