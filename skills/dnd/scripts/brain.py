#!/usr/bin/env python3
"""
brain.py — generate `brain.md`, the always-hot campaign brief.

The problem this solves: the load procedure in `SKILL-commands.md` is a list of
instructions about *which files to read and which not to* — state.md, world.md,
the npcs.md index, plus a long set of "do NOT load this" carve-outs for arc.md,
world-nodes.md, source/, session-log.md. That is a lot of prompt surface whose
job is to describe the same fact set, and every rule in it is a polite request
that decays under context pressure. It is also fragile in the other direction:
the instructions do not carry the facts themselves, so a DM that skips a read
has no fallback.

`brain.md` inverts that. It is a single generated file, assembled
deterministically from the index sources plus `graph.json`, that carries the
facts directly. The load procedure becomes "read brain.md" — one read, no
carve-outs, no recall of which file holds which fact. Anything the brain does
not contain is by construction not needed at load time; depth is still pulled
on demand via `/dm:dnd npc <name>` or the chapter's `source/<id>.md`.

Design constraints:

  * **Deterministic.** No LLM in the loop. Same inputs -> byte-identical output,
    so the file is diffable, reviewable in git, and safe to regenerate (the
    header carries no wall-clock time; only source mtimes).
  * **Verbatim where it matters.** Pinned Facts and chapter source pointers are
    copied exactly. Paraphrasing a pinned fact is how it drifts.
  * **Budgeted.** `--max-words` (default 2200) caps the whole file, header included; sections are
    dropped in reverse priority order and every omission is reported on stderr
    and marked in the file, so a truncated brain never silently looks complete.
  * **Self-marking staleness.** The header records the mtime of every source it
    read. If any is newer, the brain says so loudly — a stale brief is worse than
    no brief.

  * The `## DM Notes` `relationships:` line (a mentor +2, a rival -1) is its own
    5-point scale, NOT the disposition scale. Both are surfaced, never merged.

Usage:
    python3 brain.py -c <campaign>              # write <campaign>/brain.md
    python3 brain.py -c <campaign> --stdout    # print instead of writing
    python3 brain.py -c <campaign> --check     # exit 1 if brain.md is stale
    python3 brain.py -c <campaign> --max-words 1200
"""
from __future__ import annotations

import argparse
import re
import sys

import campaign_facts as cf
import campaign_graph as cg

BRAIN_NAME = "brain.md"

# Sections in load-priority order. `budget_rank` is the order they are dropped
# when the word budget is exceeded: higher drops first. `always` sections are
# never dropped — dropping Pinned Facts would defeat the point of the file.
_SECTIONS = [
    ("now", "NOW", 0, True),
    ("pinned", "PINNED FACTS (verbatim — never contradict, never restate loosely)", 1, True),
    ("present", "IN SCENE", 2, False),
    ("factions", "FACTIONS", 3, False),
    ("loops", "OPEN LOOPS", 4, False),
    ("rel", "RELATIONSHIPS", 5, False),
    ("chapter", "CHAPTER WINDOW", 6, False),
    ("pending", "PENDING CANON DECISIONS", 7, False),
    ("index", "DEEP FILES (read on demand, not at load)", 8, False),
]

# state.md DM Notes keys that represent an undecided canon fork. These are the
# highest-value thing a short brief can carry: each is a player choice that has
# not been made yet, and misremembering one is how a DM talks itself out of the
# player's later decision.
_PENDING_RE = re.compile(r"\bpending\b|\bnot yet\b|^\s*none\s*\(|\(.*\bpending", re.I)


def _wc(text: str) -> int:
    return len(text.split())


def _active_edges(graph: dict, at_session=None) -> list:
    out = []
    for e in graph.get("edges", []):
        if cg._edge_active_at(e, at_session):
            out.append(e)
    return out


def _label(graph: dict, nid: str) -> str:
    for n in graph.get("nodes", []):
        if n["id"] == nid:
            return n.get("name", nid)
    return nid


def edges_of_type(graph: dict, at_session, *types) -> list:
    """Active edges of the given type(s), at the given session."""
    return [e for e in _active_edges(graph, at_session) if e.get("type") in types]


def _loc_matches(loc: str, candidate: str) -> bool:
    """Is an npcs.md Location cell the same place as the state's Location?

    Tested in both directions because the two fields are written at different
    granularities and neither is a prefix of the other: state may say "Vrell's nook
    in the Athenaeum stacks" while the index row says "Athenaeum stacks".
    A one-directional substring test finds nothing and silently empties the
    on-scene cast, which is the single most useful thing in this file.
    """
    if not loc or not candidate:
        return False
    a, b = loc.lower(), candidate.lower()
    if a in b or b in a:
        return True
    # Compare on the significant words, so "Vrell's nook in the Athenaeum
    # stacks" still matches a row reading "Athenaeum".
    stop = {"the", "in", "at", "of", "s"}
    wa = {w for w in re.findall(r"[a-z0-9]+", a) if w not in stop and len(w) > 2}
    wb = {w for w in re.findall(r"[a-z0-9]+", b) if w not in stop and len(w) > 2}
    return bool(wa & wb)


def _short(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return cut + "…"


def _clean_val(v: str) -> str:
    """Strip markdown furniture that leaks into a captured value.

    A `**Label:**` bullet that is the last one in a section picks up the
    following `---` rule, so 'friendly (Ninefold), hostile when erratic ---'
    ends up in the brain. Cosmetic here, but it is the kind of noise that makes
    a generated file stop looking trustworthy.
    """
    v = re.sub(r"\s*-{3,}\s*$", "", str(v).strip())
    return v.strip()


def build(campaign: str, max_words: int = 2200, at_session=None) -> tuple:
    """Assemble the brain text. Returns (text, meta). Pure read; writes nothing."""
    facts = cf.load(campaign)
    st = facts["state"]
    graph = cg._load(campaign)
    session = at_session if at_session is not None else (st.get("session_count") or 0)

    loc = st.get("situation", {}).get("Location", "")
    sections: dict = {}

    # ── NOW ────────────────────────────────────────────────────────────────
    now: list = []
    for k, v in st.get("situation", {}).items():
        now.append(f"- **{k}:** {v}")
    ws = st.get("world_state", {})
    for k, v in ws.items():
        now.append(f"- **{k}:** {v}")
    for c in st.get("cover", []):
        now.append(f"- **Cover:** {c}")
    sections["now"] = "\n".join(now) or "_No Current Situation block._"

    # ── PINNED FACTS (verbatim) ────────────────────────────────────────────
    pinned = st.get("pinned_facts", [])
    sections["pinned"] = "\n".join(f"- {p}" for p in pinned) if pinned else "_None pinned._"

    # ── IN SCENE ───────────────────────────────────────────────────────────
    # Cast at the current location, plus anyone the party has a non-neutral
    # disposition toward, plus a capped roster of the rest.
    #
    # Dispositions come from graph.json, not from re-matching state.md's
    # disposition keys against npcs.md names: those two files disagree by design
    # ("Vell" vs "Magister Vell", "Sera" vs "Esteemed Prof. Sera"), so a name
    # join silently drops the two most important NPCs in the campaign. The seeder already resolved those to node
    # ids; reuse that instead of re-deriving it a second, worse way.
    npcs = facts["npcs"]
    node_name = {n["id"]: n.get("name", n["id"]) for n in graph.get("nodes", [])}
    disposed_ids = {
        e["to"]: e for e in edges_of_type(graph, session, "disposition")
    }

    loc_l = loc.lower()
    at_loc = [r for r in npcs if _loc_matches(loc_l, r.get("location") or "")]

    def _line(r):
        bits = [r["name"]]
        if r.get("location"):
            bits.append(_short(r["location"], 40))
        if r.get("attitude") and r["attitude"].lower() not in ("neutral", "none", "-", ""):
            bits.append(f"[{r['attitude']}]")
        if r.get("notes"):
            bits.append(f"— {_short(r['notes'], 60)}")
        return "  " + " · ".join(bits)

    scene: list = []
    if at_loc:
        scene.append("**At the current location** — read each one's `npc-files/` entry "
                     "before voicing them:")
        scene += [_line(r) for r in at_loc]
    if disposed_ids:
        scene.append("")
        scene.append("**Party disposition** (graph.json, from state.md Live State Flags):")
        for nid, e in sorted(disposed_ids.items(), key=lambda kv: _label(graph, kv[0])):
            name = _label(graph, nid)
            note = f"  — {e['note']}" if e.get("note") else ""
            scene.append(f"  {name}: {e.get('level','?')}{note}")
    disp_names = {_label(graph, nid).lower() for nid in disposed_ids}
    others = [r for r in npcs
              if r not in at_loc and r["name"].lower() not in disp_names]
    # Rank the long tail by how likely it is to matter: non-neutral attitude
    # first, then anyone with a graph edge at all, then alphabetical. Without
    # this the tail is 33 undifferentiated lines that crowd out the sections
    # carrying actual decisions.
    def _rank(r):
        att = (r.get("attitude") or "").lower()
        non_neutral = 0 if att in ("neutral", "none", "-", "") else 1
        known = 1 if r["name"].lower() in {n.lower() for n in node_name.values()} else 0
        return (-non_neutral, -known, r["name"])
    others.sort(key=_rank)
    CAP = 14
    if others:
        scene.append("")
        scene.append("**Elsewhere** (full index in `npcs.md`; depth in `npcs-full.md`):")
        scene += [_line(r) for r in others[:CAP]]
        if len(others) > CAP:
            scene.append(f"  … and {len(others) - CAP} more in `npcs.md`")
    sections["present"] = "\n".join(scene) or "_No NPC index._"

    # ── FACTIONS ───────────────────────────────────────────────────────────
    fac: list = []
    stances = st.get("faction_stances", {})
    for f in facts["factions"]:
        head = f"**{f['name']}**"
        if f.get("kind"):
            head += f" ({f['kind']})"
        if f.get("blurb"):
            head += f" — {f['blurb']}"
        fac.append(head)
        for label in ("Goals", "Secret", "Current activity", "Attitude toward party"):
            if f["labels"].get(label):
                fac.append(f"  - {label}: {_short(_clean_val(f['labels'][label]), 110)}")
        if f["name"] in stances:
            fac.append(f"  - **Party standing (state.md):** {stances[f['name']]}")
    sections["factions"] = "\n".join(fac) or "_No factions block._"

    # ── OPEN LOOPS ─────────────────────────────────────────────────────────
    loops: list = []
    if st.get("quests"):
        loops.append("**Quests:**")
        loops += [f"  - {q}" for q in st["quests"]]
    if st.get("threads"):
        loops.append("")
        loops.append("**Threads & rumours:**")
        loops += [f"  - {t}" for t in st["threads"]]
    beats = st.get("arc", {}).get("outstanding_beats", "")
    if beats:
        loops.append("")
        loops.append("**Outstanding arc beats** (must land before the chapter closes):")
        loops.append(f"  {beats}")
    sections["loops"] = "\n".join(loops) or "_Nothing outstanding._"

    # ── RELATIONSHIPS ──────────────────────────────────────────────────────
    # Ranked by decision value, not by edge id. A seeded graph is mostly
    # `based_at` and `member_of` — structural facts that are already visible in
    # IN SCENE and FACTIONS. Listing all of them buries the handful of
    # disposition/standing edges that actually govern how an NPC behaves, so
    # those come first and the structural bulk is capped.
    edges = _active_edges(graph, session)
    type_rank = {"disposition": 0, "standing": 0, "owes": 1, "opposes": 1,
                 "loyal_to": 1, "advances_thread": 1, "blocks_thread": 1,
                 "member_of": 2, "includes": 2, "based_at": 3}
    per_type_cap = {"disposition": 20, "standing": 20, "member_of": 12,
                    "includes": 6, "based_at": 6}
    rel: list = []
    seen_type: dict = {}
    dropped_edges: list = []
    for e in sorted(edges, key=lambda x: (type_rank.get(x.get("type", ""), 4),
                                          x.get("type", ""), x.get("id", ""))):
        t = e.get("type", "?")
        if seen_type.get(t, 0) >= per_type_cap.get(t, 8):
            dropped_edges.append(e)
            continue
        seen_type[t] = seen_type.get(t, 0) + 1
        lvl = f":{e['level']}" if e.get("level") else ""
        a, b = _label(graph, e["from"]), _label(graph, e["to"])
        note = f"  — {_short(e['note'], 46)}" if e.get("note") else ""
        rel.append(f"- {a} --[{t}{lvl}]--> {b}{note}")
    if dropped_edges:
        rel.append(f"- _… {len(dropped_edges)} further structural edges omitted; "
                   f"query `campaign_graph.py scene-context` for the live scene._")
    sections["rel"] = "\n".join(rel) or "_graph.json has no active edges — run `graph_seed.py`._"

    # ── CHAPTER WINDOW ─────────────────────────────────────────────────────
    arc = st.get("arc", {})
    ch: list = []
    if arc:
        ch.append(f"- **Current chapter:** {arc.get('current_chapter','?')}"
                  f"  →  next: {arc.get('next_chapter','?')}")
        ch.append(f"- **Source to read before running a scene:** "
                  f"`{arc.get('arc_file','arc.md')}` and the chapter file it points at")
    steering = arc.get("steering_notes", "")
    if steering:
        ch.append("")
        ch.append("**Steering notes** (verbatim from state.md):")
        ch.append(f"> {steering}")
    src = {c["chapter"]: c for c in facts["chapters"] if c["is_file"]}
    cur = arc.get("current_chapter", "")
    if cur in src:
        ch.append("")
        ch.append(f"**Chapter source:** `{src[cur]['file']}` — {src[cur]['scope']}")
    sections["chapter"] = "\n".join(ch) or "_No arc._"

    # ── PENDING CANON DECISIONS ────────────────────────────────────────────
    pend: list = []
    for k, v in st.get("dm_notes", {}).items():
        if _PENDING_RE.search(str(v)):
            pend.append(f"- **{k}:** {_short(v, 120)}")
    if pend:
        pend.insert(0, "Undecided forks. Do not resolve these for the player — "
                       "record their answer with the matching field at save.")
    sections["pending"] = "\n".join(pend) if pend else "_No pending forks._"

    # ── DEEP FILES INDEX ───────────────────────────────────────────────────
    idx: list = []
    npc_dir = facts["root"] / "npc-files"
    if npc_dir.is_dir():
        names = sorted(p.stem for p in npc_dir.glob("*.md"))
        if names:
            idx.append(f"- **Per-NPC deep files** (`npc-files/<name>.md`) — voice, per-year "
                       f"knowledge, guardrails. Read before any substantial scene with: "
                       f"{', '.join(names)}")
    chs = [c for c in facts["chapters"] if c["is_file"]]
    if chs:
        idx.append(f"- **Chapter sources** (`source/<id>.md`) — {len(chs)} chapters, one file "
                   f"each. Read only the current chapter's.")
    for extra, label in (("npcs-full.md", "full NPC entries"),
                         ("world.md", "world foundations + house mechanics"),
                         ("answer-key.md", "GM-only secrets — never reveal"),
                         ("lore-bridges.md", "worldbuilding logic"),
                         ("arc.md", "full act/chapter tree"),
                         ("world-nodes.md", "quest seed bank")):
        if (facts["root"] / extra).exists():
            idx.append(f"- `{extra}` — {label}")
    sections["index"] = "\n".join(idx) or "_No deep files._"

    # ── assemble with budget ───────────────────────────────────────────────
    sources = ["state.md", "npcs.md", "world.md", "source-index.md", "graph.json"]
    stamps = []
    root = facts["root"]
    for s in sources:
        p = root / s
        if p.exists():
            # Float, not int: truncating to whole seconds hides a source that
            # was edited in the same second the brain was written, which is
            # exactly the "regenerate then patch" case that produces a stale
            # brain nobody notices.
            stamps.append((s, p.stat().st_mtime))

    dropped: list = []
    keep = {k for k, _, _, _ in _SECTIONS}
    body_parts: dict = {k: sections[k] for k, _, _, _ in _SECTIONS}

    def _render(keys):
        out = []
        for key, title, _, _ in _SECTIONS:
            if key not in keys:
                continue
            out.append(f"## {title}")
            out.append("")
            out.append(body_parts[key])
            out.append("")
        return "\n".join(out)

    def _header(dropped_keys):
        # No wall-clock timestamp: same inputs must give byte-identical output.
        # The mtime fingerprints below are the only time-derived content, and
        # they are a function of the sources, not of when brain.py ran.
        h = [
            f"# Campaign Brain: {campaign}",
            "",
            f"**Ruleset:** {st.get('ruleset','2014')}  |  **Session:** {session}",
            "",
            "<!-- generated by brain.py from "
            + ", ".join(sources)
            + " — do not edit by hand; regenerate with `python3 scripts/brain.py -c "
            + campaign + "` -->",
            "",
            "**Source fingerprints (mtime):** "
            + "  ".join(f"{s}={int(t)}" for s, t in stamps),
            "",
        ]
        if dropped_keys:
            h.append(
                f"> **TRUNCATED.** Word budget {max_words} exceeded; omitted: "
                f"{', '.join(dropped_keys)}. Pull those from the source files directly.")
            h.append("")
        return "\n".join(h)

    # Drop the LOWEST-priority optional section (highest rank number) until the
    # whole file, header included, fits the budget.
    while _wc(_header(dropped) + "\n" + _render(keep)) > max_words:
        victim = next((k for k, _, rank, always in sorted(_SECTIONS, key=lambda s: -s[2])
                       if k in keep and not always), None)
        if victim is None:
            break
        keep.discard(victim)
        dropped.append(victim)

    text = _header(dropped) + "\n" + _render(keep)
    meta = {
        "campaign": campaign,
        "words": _wc(text),
        "max_words": max_words,
        "dropped": dropped,
        "stamps": dict(stamps),
        "facts_missing": facts["missing"],
    }
    return text, meta


def _staleness(campaign: str, meta: dict) -> list:
    """Sources whose mtime is newer than brain.md. Non-empty means the brain is stale."""
    bpath = cf.resolve(campaign) / BRAIN_NAME
    if not bpath.exists():
        return ["(brain.md does not exist)"]
    bt = bpath.stat().st_mtime
    return [s for s, t in meta["stamps"].items() if t > bt]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--campaign", required=True)
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if brain.md is missing or stale; write nothing")
    ap.add_argument("--max-words", type=int, default=2200)
    ap.add_argument("--at-session", type=int, default=None)
    a = ap.parse_args()

    try:
        text, meta = build(a.campaign, max_words=a.max_words, at_session=a.at_session)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    bpath = cf.resolve(a.campaign) / BRAIN_NAME

    if a.check:
        stale = _staleness(a.campaign, meta)
        if stale:
            print(f"STALE: {a.campaign}/{BRAIN_NAME} is older than {', '.join(stale)}",
                  file=sys.stderr)
            return 1
        print(f"ok: {a.campaign}/{BRAIN_NAME} is current ({meta['words']} words)")
        return 0

    if a.stdout:
        print(text)
        return 0

    bpath.write_text(text, encoding="utf-8")
    print(f"wrote {bpath}  ({meta['words']} words, budget {meta['max_words']})",
          file=sys.stderr)
    if meta["dropped"]:
        print(f"  truncated: omitted {', '.join(meta['dropped'])}", file=sys.stderr)
    if meta["facts_missing"]:
        print(f"  missing index files: {', '.join(meta['facts_missing'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
