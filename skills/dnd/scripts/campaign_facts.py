#!/usr/bin/env python3
"""
campaign_facts.py — parse the canonical campaign markdown into structured facts.

The shared read layer under `graph_seed.py`, `brain.py`, and `check.py`. Every
one of those tools needs the same three things — the live situation, the NPC
index, and the faction roster — and every one of them needs to read them
*deterministically*, without an LLM in the loop. This module is that layer.

Canonical sources (all under the campaign dir):
  state.md   — Current Situation, Pinned Facts, World State, Live State Flags,
               Active Quests, Open Threads, Campaign Arc, header counters.
  npcs.md    — the index table. One row per named NPC: name, role, faction,
               location, attitude, notes. Surface traits only.
  world.md   — `## Factions` block. `### <Name> (*kind*)` per faction.

Deliberately NOT parsed here: npcs-full.md, npc-files/*.md, source/*.md,
answer-key.md. Those are deep canon read on demand, not index material. The
whole point of the index layer is that it stays small enough to always hold in
context.

Pinned Facts are carried through **verbatim** and never paraphrased or
summarized. They are the facts the table has declared must never drift, so
any lossy handling of them here would defeat the purpose of the tool.

Stdlib only. No writes.
"""
from __future__ import annotations

import re

from utf8io import read_text, TextDecodeError

try:  # pragma: no cover - import guard for odd install layouts
    from paths import campaign_dir
except ImportError:  # pragma: no cover
    campaign_dir = None  # type: ignore


# ── section splitting ────────────────────────────────────────────────────────

def sections(text: str) -> dict:
    """Split markdown into {heading_text: body} keyed by `##`/`###` headings.

    Only `##`-level headings become keys; `###` sub-headings stay inline in the
    body of their parent, because callers that want a subsection (world.md's
    `## Factions` -> `### The Hourless`) can split further with `sub_sections`.

    Heading text is stripped of leading `#` and trailing whitespace but the
    original level is preserved in `_levels` for callers that need it.
    """
    out: dict = {}
    _levels: dict = {}
    current = None
    buf: list = []
    in_fence = False
    for line in text.splitlines():
        # A `#` inside a fenced code block (state.md's YAML arc) is not a
        # heading. Track the fence in the same pass rather than re-scanning.
        if _FENCE.match(line):
            in_fence = not in_fence
        elif in_fence:
            if current is not None:
                buf.append(line)
            continue
        else:
            m = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
            if m:
                if current is not None:
                    # First occurrence wins. Heading text alone is not a unique
                    # key: world.md has `### Three Truths` under three different
                    # `##` blocks, and letting a later duplicate clobber an
                    # earlier one silently deleted the whole `## Factions` body.
                    # Document order means the outermost block is seen first,
                    # which is the one callers mean by a bare name.
                    out.setdefault(current, "\n".join(buf).strip())
                current = m.group(2).strip()
                _levels.setdefault(current, len(m.group(1)))
                buf = []
                continue
        if current is not None:
            buf.append(line)
    if current is not None:
        out.setdefault(current, "\n".join(buf).strip())
    out["_levels"] = _levels
    return out


_FENCE = re.compile(r"^\s*(```|~~~)")


def sub_sections(body: str) -> dict:
    """Split a section body into {### heading: body}."""
    out: dict = {}
    current = None
    buf: list = []
    for line in body.splitlines():
        m = re.match(r"^#{3,6}\s+(.*?)\s*$", line)
        if m:
            if current is not None:
                out[current] = "\n".join(buf).strip()
            current = m.group(1).strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        out[current] = "\n".join(buf).strip()
    return out


def block_after(text: str, heading: str) -> str:
    """Return every line under the first `heading` at any level, sub-headings included.

    `sections()` keys the *innermost* open heading, so a `##` block whose body is
    made only of `###` children (world.md's `## Factions`) comes back empty —
    the `###` took over as `current` before anything was buffered. This walks
    the document directly instead: everything from the heading line to the next
    heading of the same or shallower level.

    First occurrence wins, matching `sections()`.
    """
    lines = text.splitlines()
    start = None
    level = 0
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
        if m and m.group(2).strip() == heading:
            start, level = i + 1, len(m.group(1))
            break
    if start is None:
        return ""
    out: list = []
    for line in lines[start:]:
        m = re.match(r"^(#{1,6})\s", line)
        if m and len(m.group(1)) <= level:
            break
        out.append(line)
    return "\n".join(out).strip()


def edit_distance(a: str, b: str, cap: int = 4) -> int:
    """Damerau-Levenshtein distance, abandoned early once it exceeds `cap`.

    Damerau rather than plain Levenshtein because the transposition case is the
    single most common human typo in a name — 'Tarn' mistyped as 'Tran' is a
    transposition plus one substitution, which plain Levenshtein scores as 2
    and therefore misses a typo that a name-suggestion feature exists to catch.
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > cap:
        return cap + 1
    prev2: list = []
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        best = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if (i > 1 and j > 1 and a[i - 1] == b[j - 2]
                    and a[i - 2] == b[j - 1]):
                cur[j] = min(cur[j], prev2[j - 2] + 1)
            best = min(best, cur[j])
        if best > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[lb]


_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")


def _dearticle(s: str) -> str:
    return _ARTICLE_RE.sub("", s.lower()).strip()


def close_names(target: str, pool, cap: int = 2) -> list:
    """Names from `pool` within edit distance `cap` of `target`, best first.

    Ranked by (distance, length difference, name) so an exact-length
    one-character miss outranks a same-distance word of a different shape.

    Compared both as-is and with a leading article stripped, because faction
    and place names carry articles inconsistently ('The Quiet Hand' vs a column
    that says 'Quiet Hand'). Without that, a one-character typo scores 5 and is
    indistinguishable from a completely different name.
    """
    t = str(target).lower()
    t_da = _dearticle(t)
    scored = []
    for cand in pool:
        c = str(cand).lower()
        c_da = _dearticle(c)
        d = min(edit_distance(t, c, cap=cap), edit_distance(t_da, c_da, cap=cap))
        if d <= cap:
            scored.append((d, abs(len(c) - len(t)), str(cand)))
    scored.sort()
    return [s[2] for s in scored]


def _clean_name(h: str) -> str:
    """Strip the trailing emphasis a faction heading carries: `The Hourless (*cult*)`."""
    return re.sub(r"\s*\(\*.*?\*\)\s*$", "", h).strip()


def _clean_label(label: str) -> str:
    """Normalize a `**Label:**` key: strip, collapse whitespace, drop a trailing colon."""
    return re.sub(r"\s+", " ", label).strip().rstrip(":").strip()


# ── state.md ────────────────────────────────────────────────────────────────

_COUNTERS = {
    "session_count": re.compile(r"\*\*Session count:\*\*\s*(\d+)", re.I),
    "ruleset": re.compile(r"\*\*Ruleset:\*\*\s*(2014|2024)", re.I),
    "created": re.compile(r"\*\*Created:\*\*\s*([\d-]+)", re.I),
    "last_session": re.compile(r"\*\*Last session:\*\*\s*(\S+)", re.I),
}

_NONE_MARKERS = re.compile(r"^\s*[\(\[\*]?\s*(none|n/?a|nothing|—|-)\s*[\)\]\*]?\s*$", re.I)


def _bullets(body: str) -> list:
    """**Top-level** `- ` bullets of a section, verbatim, minus blank/italic-none.

    Indentation matters: a nested bullet is sub-structure under its parent, not
    a peer fact. `## World State` nests a `**Faction states:**` list of its own,
    and hoisting those into the parent dict silently overwrites the real
    `In-world date` / `Season` keys. Only zero-indent bullets count.
    """
    out = []
    for line in body.splitlines():
        m = re.match(r"^[-*]\s+(.*\S)\s*$", line)  # no leading \s -> top level only
        if m and not _NONE_MARKERS.match(m.group(1)):
            out.append(m.group(1))
    return out


def _strip_md(s: str) -> str:
    """Drop `**bold**` / `*italic*` / `` `code` `` markers, keep the words."""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", s)
    return s


def _kv_bullets(body: str) -> dict:
    """`- key: value` bullets into an ordered dict. Values keep their prose.

    The key is markdown-stripped before matching, so `- **Location:** ...`
    keys on `Location` rather than `**Location`.
    """
    out: dict = {}
    for b in _bullets(body):
        m = re.match(r"^([A-Za-z][A-Za-z0-9 _/()-]{0,60}?)\s*:\s*(.+)$", _strip_md(b))
        if m:
            out[m.group(1).strip()] = _strip_md(m.group(2).strip())
    return out


# `**Cover:**` and friends are bold pseudo-headings, not `###` headings. The
# trailing gloss (`*(only list factions with non-neutral standing)*`) is part
# of the label line, not the body.
_BOLD_HEAD = re.compile(r"^\*\*([^*:]{1,60}):?\*\*\s*(\*\(.*?\)\*)?:?\s*$")


def _bold_sections(body: str) -> dict:
    """Split a section body on `**Bold label:**` pseudo-headings."""
    out: dict = {}
    current = None
    buf: list = []
    for line in body.splitlines():
        m = _BOLD_HEAD.match(line.strip())
        if m:
            if current is not None:
                out[current] = "\n".join(buf).strip()
            current = m.group(1).strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        out[current] = "\n".join(buf).strip()
    return out


def _fenced_yaml(body: str) -> dict:
    """Parse the first ```yaml fenced block into {key: value}.

    `## Campaign Arc` stores its payload in a YAML fence, so the normal
    heading split puts the whole block in the body. Only zero-indent
    `key: value` lines are taken — the nested `current_chapter_detail:` mapping
    is deliberately flattened away, since `arc.md` is the canonical home for it
    and callers want the pointer, not an inlined copy.
    """
    m = re.search(r"```ya?ml\s*\n(.*?)```", body, re.S)
    if not m:
        return {}
    out: dict = {}
    lines = m.group(1).splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m2 = re.match(r"^([a-z_][a-z0-9_]*):\s*(.*)$", line)
        if not m2:
            i += 1
            continue
        key, val = m2.group(1), m2.group(2).strip()
        if val in (">", "|", ">-", "|-"):
            # Folded/literal block scalar: the real value is the indented lines
            # that follow. `steering_notes` uses `>` and is the single most
            # load-bearing field in the arc, so it must not be dropped.
            block: list = []
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or lines[j].startswith(("  ", "\t"))):
                if lines[j].strip():
                    block.append(lines[j].strip())
                j += 1
            out[key] = " ".join(block)
            i = j
            continue
        out[key] = val.strip('"')
        i += 1
    return out


def parse_state(text: str) -> dict:
    """Parse state.md into a fact dict.

    Returns:
      {
        "session_count": int, "ruleset": "2014"|"2024", "created": str,
        "situation": {k: v},            # Current Situation key/value bullets
        "pinned_facts": [str, ...],     # VERBATIM
        "world_state": {k: v},
        "quests": [str, ...],
        "threads": [str, ...],
        "faction_moves": [str, ...],
        "recent_events": [str, ...],
        "cover": [str, ...],
        "faction_stances": {name: level},
        "npc_dispositions": {name: level},
        "session_flags": {k: v},
        "dm_notes": {k: v},             # hidden block — GM-only, never player-facing
        "arc": {k: v},                  # flattened `## Campaign Arc` YAML
        "raw": text,
      }
    """
    sec = sections(text)
    head = "\n".join(text.splitlines()[:4])

    out: dict = {"raw": text, "situation": {}, "world_state": {}, "session_flags": {},
                 "faction_stances": {}, "npc_dispositions": {}, "dm_notes": {},
                 "arc": {}, "pinned_facts": []}

    for key, rx in _COUNTERS.items():
        m = rx.search(head)
        if m:
            out[key] = int(m.group(1)) if key == "session_count" else m.group(1)
    out.setdefault("session_count", 0)
    out.setdefault("ruleset", "2014")

    out["situation"] = _kv_bullets(sec.get("Current Situation", ""))
    out["pinned_facts"] = _bullets(sec.get("Pinned Facts", ""))
    out["world_state"] = _kv_bullets(sec.get("World State", ""))
    out["quests"] = _bullets(sec.get("Active Quests", ""))
    out["threads"] = _bullets(sec.get("Open Threads & Rumours", ""))
    out["faction_moves"] = _bullets(sec.get("Faction Moves", ""))
    out["recent_events"] = _bullets(sec.get("Recent Events", ""))
    out["session_flags"] = _kv_bullets(sec.get("Session Flags", ""))

    # Live State Flags carries three labelled sub-blocks as bold pseudo-headings.
    live = _bold_sections(sec.get("Live State Flags", ""))
    out["cover"] = _bullets(live.get("Cover", ""))
    out["faction_stances"] = _kv_bullets(live.get("Faction stances", ""))
    out["npc_dispositions"] = _kv_bullets(live.get("NPC dispositions", ""))

    out["dm_notes"] = _kv_bullets(sec.get("DM Notes (hidden from players)", ""))
    out["arc"] = _fenced_yaml(sec.get("Campaign Arc", ""))
    return out


# ── npcs.md index ────────────────────────────────────────────────────────────

def parse_npcs_index(text: str) -> list:
    """Parse the npcs.md index table into a list of row dicts.

    The header row is `| Name | Role | Faction | Location | Attitude | Notes |`;
    separator rows (`|---|`) are skipped. Cells are stripped of emphasis and
    surrounding whitespace. A row with no name is skipped.

    `location` may be a comma-separated progression ("First-year hall, frog
    pond (Witherbloom dorms from Y2)") — kept verbatim, since the parenthetical
    is often the more useful half.
    """
    rows: list = []
    cols = ("name", "role", "faction", "location", "attitude", "notes")
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # Skip separator rows (`|---|---|`) and the header. A separator must be
        # *skipped*, not kept — keeping it appends a junk row named `------`.
        if not cells or re.match(r"^:?-{2,}:?$", cells[0]):
            continue
        if cells[0].strip().lower() == "name":
            continue
        if len(cells) < 2:
            continue
        row = {}
        for i, col in enumerate(cols):
            row[col] = re.sub(r"\*\*(.+?)\*\*", r"\1", cells[i]).strip() if i < len(cells) else ""
        if row["name"]:
            rows.append(row)
    return rows


# ── world.md factions ────────────────────────────────────────────────────────

def parse_factions(text: str) -> list:
    """Parse `## Factions` -> `### Name (*kind*)` from world.md.

    Each faction block is a list of `**Label:** value` bullets (Goals, Methods,
    Resources, Opposition, Secret, Current activity, Attitude toward party,
    sometimes a long canon note). Those labels are the useful part, so each is
    captured as a named field rather than flattened into a blurb.

    Returns [{name, kind, labels: {label: value}, blurb}] where `blurb` is the
    first non-list prose line, if the block opens with one. `attitude` is also
    lifted to the top level because it is a *party standing* — a second source
    that can be cross-checked against `state.md`'s faction stances.
    """
    # `## Factions` is a pure container (only `###` children), so scan the
    # document for its block rather than reading a body that came back empty.
    fsec = sub_sections(block_after(text, "Factions"))
    out: list = []
    for head, body in fsec.items():
        name = _clean_name(head)
        kind = ""
        m = re.search(r"\(\*(.+?)\*\)", head)
        if m:
            kind = m.group(1).strip()

        labels: dict = {}
        blurb = ""
        for raw in body.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line[0] not in "-*":
                if not blurb and len(line.split()) >= 4:
                    blurb = _strip_md(line)
                continue
            # Strip the bullet, then split a `**Label:** value` pair if present.
            item = re.sub(r"^[-*]\s+", "", line)
            m = re.match(r"^\*\*([^*]{1,60}?)\*\*\s*:?\s*(.*)$", item)
            if m:
                labels[_clean_label(m.group(1))] = _strip_md(m.group(2).strip())
            elif labels:
                # Indented continuation of the previous label (e.g. the long
                # post-invasion canon note). Append rather than drop — losing a
                # "never mix these two sets of deans" rule is a real canon bug.
                k = list(labels)[-1]
                labels[k] = (labels[k] + " " + _strip_md(item)).strip()
            elif not blurb:
                blurb = _strip_md(item)

        out.append({
            "name": name,
            "kind": kind,
            "labels": labels,
            "attitude": labels.get("Attitude toward party", ""),
            "blurb": blurb,
        })
    return out


# ── source-index.md ──────────────────────────────────────────────────────────

def parse_source_index(text: str) -> list:
    """Parse the chapter table in source-index.md.

    Returns [{chapter, file, scope}]. The `Supp` / `NPC` / `Y3` pseudo-rows
    (which point at globs and directories rather than one chapter file) are
    included but flagged `is_file: False` so callers can treat them as indexes
    rather than lazy chapter reads.
    """
    out: list = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        if re.match(r"^:?-{2,}:?$", cells[0]):
            continue
        if cells[0].strip().lower() == "chapter":
            continue
        chapter, f, scope = cells[0], cells[1], cells[2]
        out.append({
            "chapter": chapter,
            "file": f,
            "scope": re.sub(r"\*\*(.+?)\*\*", r"\1", scope),
            "is_file": bool(re.match(r"^[\w./-]+\.md$", f)),
        })
    return out


# ── top-level loader ─────────────────────────────────────────────────────────

def load(campaign: str) -> dict:
    """Load and parse every canonical index file for a campaign.

    Returns a dict with `state`, `npcs`, `factions`, `chapters`, plus a
    `missing` list naming index files that were absent (older campaigns, or
    sandbox campaigns where world.md holds the nodes instead). Absent files
    degrade to empty lists — never an exception — so callers can render a
    partial brain rather than failing a session load.
    """
    if campaign_dir is None:  # pragma: no cover
        raise RuntimeError("paths.campaign_dir unavailable")
    root = campaign_dir(campaign)
    out: dict = {"missing": [], "root": root}

    def _read(fname):
        p = root / fname
        if not p.exists():
            out["missing"].append(fname)
            return None
        try:
            return read_text(p)
        except TextDecodeError:
            out["missing"].append(fname + " (undecodable)")
            return None

    st = _read("state.md")
    out["state"] = parse_state(st) if st else parse_state("")
    np = _read("npcs.md")
    out["npcs"] = parse_npcs_index(np) if np else []
    wm = _read("world.md")
    out["factions"] = parse_factions(wm) if wm else []
    si = _read("source-index.md")
    out["chapters"] = parse_source_index(si) if si else []
    return out


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Dump parsed campaign facts as JSON.")
    ap.add_argument("-c", "--campaign", required=True)
    a = ap.parse_args()
    print(json.dumps(load(a.campaign), indent=2, ensure_ascii=False, default=str))
