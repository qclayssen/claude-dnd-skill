#!/usr/bin/env python3
"""
graph_seed.py — seed graph.json deterministically from the campaign index files.

`SKILL-commands.md` tells the DM to run `/dm:dnd graph init <name>` at load time
when `graph.json` is missing, but no `init` subcommand existed in
`campaign_graph.py` — the flow existed only as prose for the model to improvise
from, which is exactly the kind of step that gets skipped or done differently
every time. This script is that missing step, done deterministically.

It proposes seed nodes and typed edges derived from the canonical index files
(`npcs.md`, `world.md`, `state.md`, `characters/*.md`) and, with `--apply`,
writes them into `graph.json` through the same helpers `campaign_graph.py` uses
so the on-disk format is identical.

  * Dry-run by default. Nothing is written without `--apply`.
  * Idempotent: re-running proposes only what is genuinely missing, so it is
    safe to run after every `/dm:dnd save` as a drift check.
  * No LLM in the loop. Every node and edge traces to a specific line of a
    canonical file, and each carries that provenance in `note`.

What it seeds:
  nodes  party · pc (characters/) · npc (npcs.md) · faction (world.md + the
        Faction column) · place (the Location column)
  edges  npc --member_of--> faction
        npc --based_at---> place
        pc  --includes---> party        (party --includes--> pc)
        party --disposition--> npc      (from state.md NPC dispositions)
        party --standing----> faction   (from state.md faction stances)

It does NOT guess relationships the sources do not state. A missing edge is
left to `/dm:dnd graph extract` or the save-time relationship sweep; seeding an
invented edge is worse than seeding none.

Usage:
    python3 graph_seed.py -c <campaign>              # propose (dry run)
    python3 graph_seed.py -c <campaign> --apply      # write graph.json
    python3 graph_seed.py -c <campaign> --json       # machine-readable proposal
    python3 graph_seed.py -c <campaign> --no-places  # skip place nodes
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import campaign_facts as cf
import campaign_graph as cg
from utf8io import read_text as cf_utf8_read

# Values in the Faction / Location columns that mean "no such thing".
_NON_FACTION = {"", "-", "—", "none", "n/a", "independent", "unknown"}
_NON_PLACE = {"", "-", "—", "none", "n/a", "unknown", "everywhere but campus",
              "everywhere", "various", "mobile", "varies"}

# The `## DM Notes` `relationships:` line uses its own 1-based-ish score scale
# (a mentor +2, a rival -1). Deliberately NOT mapped onto disposition here — the
# scales mean different things, and conflating them is how a graph starts lying.

# Map a free-text stance onto the normalized disposition scale. Longest match
# first: "unfriendly" must be tested before "friendly", and "surface friendly"
# is surface-level, not genuinely friendly.
_STANCE_PATTERNS = [
    ("allied", "allied"),
    ("friendly", "friendly"),
    ("neutral", "neutral"),
    ("suspicious", "suspicious"),
    ("hostile", "hostile"),
    ("unfriendly", "suspicious"),
    ("wary", "suspicious"),
    ("allied", "allied"),
]


def _slug(s: str) -> str:
    """Node id slug. Matches campaign_graph._slug so ids agree across both tools."""
    return "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_")


def _node_id(ntype: str, name: str) -> str:
    return f"{ntype}_{_slug(name)}"


def map_stance(text: str) -> str:
    """Map a free-text stance ('surface unfriendly (secretly protective)') to a level.

    Tries the named-stance scale first (first token by position wins, so
    'unfriendly' beats a later 'friendly'). Failing that, tries the numeric
    relationship scale the character sheets and `dm_notes` use ('+2', '-1').

    Returns '' when nothing matches. Callers must treat '' as "unrecognized" and
    skip the edge rather than default it — defaulting to `neutral` here would
    record a confident reading of a scale we did not actually understand, which
    is the same class of bug as inventing canon.
    """
    low = text.lower()
    best = ""
    best_pos = len(low) + 1
    for token, level in _STANCE_PATTERNS:
        pos = low.find(token)
        if pos >= 0 and pos < best_pos:
            best, best_pos = level, pos
    if best:
        return best
    return _map_score(text)


def _map_score(text: str) -> str:
    """Map a signed relationship score (+2 / -1 / 0) onto the stance scale.

    The `relationships:` line in DM Notes and the disposition bullets in Live
    State Flags use the same 5-point feel: a mentor at +2 is as strong as it gets,
    Orrin -1 is a mild wariness, 0 is undecided.
    """
    m = re.search(r"([+-]?\d+)", text)
    if not m:
        return ""
    try:
        n = int(m.group(1))
    except ValueError:
        return ""
    if n >= 2:
        return "allied"
    if n == 1:
        return "friendly"
    if n == 0:
        return "neutral"
    if n == -1:
        return "suspicious"
    return "hostile"


def _surface_flag(text: str) -> str:
    """Note whether a stance is explicitly a surface/secret one.

    'surface unfriendly (secretly protective)' is two facts. Collapsing it to
    `suspicious` would lose the second — and that second half is exactly the
    kind of thing a DM forgets and then contradicts on stage.
    """
    low = text.lower()
    flags = []
    if "surface" in low or "outward" in low or "publicly" in low:
        flags.append("surface")
    if any(k in low for k in ("secret", "secretly", "hidden", "actually")):
        flags.append("secret")
    return "+".join(flags)


def split_location(loc: str) -> tuple:
    """Split an index Location cell into (canonical place, aliases).

    'First-year hall, frog pond (Witherbloom dorms from Y2)' -> the first segment
    is canonical and the rest become aliases, so a player asking about 'the frog
    pond' still resolves to the same place node.
    """
    loc = loc.strip()
    if not loc:
        return "", []
    parts = [p.strip() for p in loc.split(",") if p.strip()]
    if not parts:
        return "", []
    head = parts[0]
    aliases = []
    for p in parts[1:]:
        p = re.sub(r"\s*\([^)]*\)\s*$", "", p).strip() or p.strip()
        if p:
            aliases.append(p)
    m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", head)
    if m:
        head = m.group(1).strip()
        aliases.insert(0, m.group(2).strip())
    return head, [a for a in aliases if a]


def match_faction(raw: str, known: list) -> str:
    """Resolve a Faction-column value to a known faction name, else ''.

    Tries exact, then case-insensitive, then containment either way, then a
    close-ratio fuzzy match. 'Faculty' resolves to 'Strixhaven Faculty'; 'The
    Tally (Vess)' resolves to 'The Tally (Vess the Tallykeeper)'.
    """
    if not raw:
        return ""
    low = raw.strip().lower()
    for f in known:
        if f.lower() == low:
            return f
    for f in known:
        if f.lower() in low or low in f.lower():
            return f
    base = re.sub(r"\s*\(.*?\)\s*", " ", raw).strip().lower()
    for f in known:
        fb = re.sub(r"\s*\(.*?\)\s*", " ", f).strip().lower()
        if fb and (fb in base or base in fb):
            return f
    # Edit distance rather than a difflib ratio: 'Qiet Hand' vs 'The Quiet Hand'
    # is a one-character typo but a low ratio purely because of the length gap,
    # so a ratio threshold either misses real typos or admits unrelated names.
    for name in cf.close_names(raw, known, cap=3):
        return name
    return ""


class Proposal:
    """Accumulates proposed nodes/edges, deduplicated and provenance-tagged."""

    def __init__(self, existing: dict):
        self.nodes: list = []
        self.edges: list = []
        self.existing = existing
        self.have_nodes = {n["id"] for n in existing.get("nodes", [])}
        self._node_names = {n["id"]: n.get("name", "") for n in existing.get("nodes", [])}
        self._node_keys = {n.get("name", "").lower(): n["id"] for n in existing.get("nodes", [])}
        # (from, to, type) triples already present and not closed — the same
        # idempotence key campaign_graph._existing_edge_match uses.
        self.have_edges = {
            (e["from"], e["to"], e["type"])
            for e in existing.get("edges", [])
            if e.get("until_session") is None and not e.get("superseded_by")
        }
        self._node_seen: set = set()
        self._edge_seen: set = set()
        self.unresolved: list = []

    def node(self, ntype: str, name: str, note: str = "", tags=None, summary: str = "",
             force_id: str = "") -> str:
        """Register a node. Returns its id (existing or new).

        `force_id` pins the id for the handful of nodes whose id is a contract
        rather than a derivation — the party node is `party` because
        campaign_graph.PARTY_NODE_ID and every disposition edge depend on it.
        """
        name = name.strip()
        if not name:
            return ""
        key = name.lower()
        if key in self._node_keys:
            return self._node_keys[key]
        nid = force_id or _node_id(ntype, name)
        if nid in self.have_nodes:
            return nid
        if nid in self._node_seen:
            return nid
        self._node_seen.add(nid)
        entry = {"id": nid, "type": ntype, "name": name}
        if tags:
            entry["tags"] = sorted(set(tags))
        if summary:
            entry["summary"] = summary
        if note:
            entry["source_note"] = note
        self.nodes.append(entry)
        self._node_keys[key] = nid
        return nid

    def edge(self, frm: str, to: str, etype: str, note: str = "", since=None, level=None):
        if not frm or not to or frm == to:
            return
        if (frm, to, etype) in self.have_edges or (frm, to, etype) in self._edge_seen:
            return
        self._edge_seen.add((frm, to, etype))
        e = {"from": frm, "to": to, "type": etype,
             "since_session": since, "until_session": None}
        if level:
            e["level"] = level
        if note:
            e["note"] = note
        self.edges.append(e)


def build(campaign: str, want_places: bool = True) -> dict:
    """Build the proposal for a campaign. Pure read; writes nothing."""
    facts = cf.load(campaign)
    existing = cg._load(campaign)
    st = facts["state"]
    p = Proposal(existing)

    since = st.get("session_count") or 0
    # A campaign with no sessions yet is still foundational: everything read off
    # the index is canon from session 1, not "as of session 0".
    since = since if since > 0 else None

    # ── factions ────────────────────────────────────────────────────────────
    faction_names = [f["name"] for f in facts["factions"]]
    faction_ids: dict = {}
    for f in facts["factions"]:
        fid = p.node("faction", f["name"],
                     note="world.md ## Factions",
                     tags=[f["kind"]] if f["kind"] else None)
        faction_ids[f["name"].lower()] = fid

    # ── party ───────────────────────────────────────────────────────────────
    # id MUST be campaign_graph.PARTY_NODE_ID: every disposition/standing edge
    # keys off it, and `set-disposition` resolves that literal. A `party_the_party`
    # id would create a second party node and silently orphan every stance.
    party_id = p.node("party", "The Party", note="state.md", force_id=cg.PARTY_NODE_ID)

    # ── player characters ───────────────────────────────────────────────────
    root = facts["root"]
    chars_dir = root / "characters"
    pc_ids: list = []
    if chars_dir.is_dir():
        for cpath in sorted(chars_dir.glob("*.md")):
            try:
                csec = cf.sections(cf_utf8_read(cpath))
            except Exception:
                continue  # undecodable or unreadable — skip, don't abort the seed
            cname = cpath.stem.replace("-", " ").replace("_", " ").title()
            for h in csec:
                if h.lower().startswith(("character:", "pc:")):
                    cname = h.split(":", 1)[1].strip()
                    break
            # A PC's headline is its `## Identity` line — race/class/level in one
            # glance, which is what a scene-context subgraph should show.
            summary = ""
            ident = csec.get("Identity", "")
            if ident:
                summary = cf._strip_md(
                    re.sub(r"\s*\|\s*", " · ", ident.splitlines()[0].strip().lstrip("- ")))
            cid = p.node("pc", cname, note=f"characters/{cpath.name}",
                         summary=summary[:200] or None)
            if cid:
                pc_ids.append(cid)
                p.edge(party_id, cid, "includes", note="characters/")

    # ── NPCs ────────────────────────────────────────────────────────────────
    npc_rows = facts["npcs"]
    npc_by_name: dict = {}
    for row in npc_rows:
        name = row["name"]
        role = row.get("role", "")
        npc_id = p.node("npc", name, note="npcs.md index",
                        tags=[t.strip() for t in role.split(",") if t.strip()][:4] or None,
                        summary=(row.get("notes") or "")[:200] or None)
        if not npc_id:
            continue
        npc_by_name[name.lower()] = npc_id

        # npc --member_of--> faction
        raw_fac = (row.get("faction") or "").strip()
        if raw_fac.lower() not in _NON_FACTION:
            resolved = match_faction(raw_fac, faction_names)
            if resolved:
                p.edge(npc_id, faction_ids[resolved.lower()], "member_of",
                       note=f"npcs.md Faction column: {raw_fac!r}", since=since)
            else:
                p.unresolved.append(("faction", name, raw_fac))

        # npc --based_at--> place
        if want_places:
            loc_raw = (row.get("location") or "").strip()
            if loc_raw.lower() not in _NON_PLACE:
                place, aliases = split_location(loc_raw)
                if place and place.lower() not in _NON_PLACE:
                    pid = p.node("place", place, note="npcs.md Location column",
                                 tags=aliases[:3] or None)
                    if pid:
                        p.edge(npc_id, pid, "based_at",
                               note=f"npcs.md Location: {loc_raw!r}", since=since)

    # ── party stances ───────────────────────────────────────────────────────
    for fname, level_raw in st.get("faction_stances", {}).items():
        resolved = match_faction(fname, faction_names)
        if not resolved:
            p.unresolved.append(("faction_stance", fname, level_raw))
            continue
        level = map_stance(level_raw)
        if not level:
            p.unresolved.append(("faction_stance_unmapped", fname, level_raw))
            continue
        p.edge(party_id, faction_ids[resolved.lower()], "standing",
               note=f"state.md faction stance: {level_raw!r}", since=since,
               level=level)

    for dname, level_raw in st.get("npc_dispositions", {}).items():
        target = npc_by_name.get(dname.strip().lower())
        if not target:
            resolved = match_faction(dname, [n["name"] for n in facts["npcs"]])
            target = npc_by_name.get(resolved.lower()) if resolved else None
        if not target:
            p.unresolved.append(("npc_disposition", dname, level_raw))
            continue
        level = map_stance(level_raw)
        if not level:
            p.unresolved.append(("npc_disposition_unmapped", dname, level_raw))
            continue
        note = f"state.md disposition: {level_raw!r}"
        sf = _surface_flag(level_raw)
        if sf:
            note += f" [{sf}]"
        p.edge(party_id, target, "disposition", note=note, since=since, level=level)

    # ── cross-check world.md's per-faction attitude against state.md ─────────
    # Both are canonical party standings. A mismatch means one is stale, which is
    # exactly the drift this whole system exists to catch — surface it at seed
    # time rather than discovering it mid-scene.
    #
    # `state.md` lists ONLY non-neutral factions, so an absent entry means
    # neutral rather than unknown. Comparing against neutral is what makes this
    # check useful: world.md saying 'allied' while state.md omits the faction is
    # a real disagreement, and skipping absent entries would hide every case
    # where a standing quietly changed.
    conflicts: list = []
    for f in facts["factions"]:
        att = (f.get("attitude") or "").strip()
        if not att:
            continue
        fid = faction_ids.get(f["name"].lower())
        sname = match_faction(f["name"], list(st.get("faction_stances", {}).keys()))
        sval = st.get("faction_stances", {}).get(sname, "neutral") if sname else "neutral"
        a = map_stance(att)
        b = map_stance(sval)
        if a and b and a != b:
            conflicts.append({
                "faction": f["name"],
                "world_md": att.strip(),
                "state_md": sval if sname else "neutral (absent from state.md)",
                "mapped": {"world.md": a, "state.md": b},
                "node": fid,
            })

    return {
        "campaign": campaign,
        "nodes": p.nodes,
        "edges": p.edges,
        "unresolved": p.unresolved,
        "stance_conflicts": conflicts,
        "counts": {
            "existing_nodes": len(existing.get("nodes", [])),
            "existing_edges": len(existing.get("edges", [])),
            "new_nodes": len(p.nodes),
            "new_edges": len(p.edges),
            "npcs_indexed": len(npc_rows),
            "factions_indexed": len(facts["factions"]),
            "pcs_indexed": len(pc_ids),
        },
    }


def apply_proposal(campaign: str, prop: dict) -> dict:
    """Write the proposal into graph.json. Backs up an existing file first."""
    data = cg._load(campaign)
    before_nodes = len(data["nodes"])
    before_edges = len(data["edges"])

    gpath = cg._graph_path(campaign)
    if gpath.exists():
        import datetime
        import shutil
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = gpath.with_name(f"graph.json.pre-seed-{ts}")
        shutil.copy2(gpath, backup)
        prop["backup"] = str(backup)

    # Reuse the party-node id constant so dispositions key off the same node.
    for n in prop["nodes"]:
        if n["id"] in {x["id"] for x in data["nodes"]}:
            continue
        data["nodes"].append(n)
    for e in prop["edges"]:
        e = dict(e)
        e["id"] = cg._next_edge_id(data["edges"])
        data["edges"].append(e)
    cg._save(campaign, data)
    prop["applied"] = {
        "nodes_before": before_nodes,
        "nodes_after": len(data["nodes"]),
        "edges_before": before_edges,
        "edges_after": len(data["edges"]),
    }
    return prop


def _report(prop: dict, stream=sys.stdout) -> None:
    c = prop["counts"]
    print(f"# graph seed proposal — campaign '{prop['campaign']}'", file=stream)
    print(f"existing: {c['existing_nodes']} nodes, {c['existing_edges']} edges", file=stream)
    print(f"proposed: {c['new_nodes']} nodes, {c['new_edges']} edges", file=stream)
    print(f"sources:  {c['npcs_indexed']} npcs.md rows, {c['factions_indexed']} world.md "
          f"factions, {c['pcs_indexed']} characters/", file=stream)
    print(file=stream)

    by_type: dict = {}
    for n in prop["nodes"]:
        by_type.setdefault(n["type"], []).append(n)
    if by_type:
        print("## new nodes", file=stream)
        for t in sorted(by_type):
            print(f"  {t} ({len(by_type[t])}):", file=stream)
            for n in sorted(by_type[t], key=lambda x: x["name"]):
                s = f" — {n['summary']}" if n.get("summary") else ""
                print(f"    {n['id']}  {n['name']}{s}", file=stream)
        print(file=stream)

    if prop["edges"]:
        name_of = {n["id"]: n["name"] for n in prop["nodes"]}
        print(f"## new edges ({len(prop['edges'])})", file=stream)
        by_kind: dict = {}
        for e in prop["edges"]:
            by_kind.setdefault(e["type"], []).append(e)
        for t in sorted(by_kind):
            print(f"  {t} ({len(by_kind[t])})", file=stream)
            for e in by_kind[t]:
                a = name_of.get(e["from"], e["from"])
                b = name_of.get(e["to"], e["to"])
                lvl = f":{e['level']}" if e.get("level") else ""
                print(f"    {a} --[{t}{lvl}]--> {b}", file=stream)
        print(file=stream)

    if prop.get("unresolved"):
        print("## UNRESOLVED (no edge written — check these by hand)", file=stream)
        for kind, who, raw in prop["unresolved"]:
            print(f"  [{kind}] {who!r} <- {raw!r}", file=stream)
        print(file=stream)

    if prop.get("stance_conflicts"):
        print("## STANCE CONFLICTS (world.md vs state.md — pick one)", file=stream)
        for cf_ in prop["stance_conflicts"]:
            print(f"  {cf_['faction']}: world.md={cf_['world_md']!r} "
                  f"state.md={cf_['state_md']!r} -> {cf_['mapped']}", file=stream)
        print(file=stream)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--campaign", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="write graph.json (default is a dry run)")
    ap.add_argument("--json", action="store_true", help="emit the proposal as JSON")
    ap.add_argument("--no-places", action="store_true", help="skip place nodes")
    ap.add_argument("--allow-empty", action="store_true",
                    help="exit 0 even when nothing is proposed")
    a = ap.parse_args()

    try:
        prop = build(a.campaign, want_places=not a.no_places)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if a.json:
        print(json.dumps(prop, indent=2, ensure_ascii=False))
    else:
        _report(prop)

    if a.apply:
        prop = apply_proposal(a.campaign, prop)
        if not a.json:
            ap_ = prop["applied"]
            print(f"# APPLIED to {cg._graph_path(a.campaign)}", file=sys.stderr)
            if prop.get("backup"):
                print(f"# backup: {prop['backup']}", file=sys.stderr)
            print(f"# nodes {ap_['nodes_before']} -> {ap_['nodes_after']}, "
                  f"edges {ap_['edges_before']} -> {ap_['edges_after']}", file=sys.stderr)
    elif not a.json:
        print("# dry run — pass --apply to write graph.json", file=sys.stderr)

    if prop["counts"]["new_nodes"] or prop["counts"]["new_edges"]:
        return 0
    return 0 if a.allow_empty else 1


if __name__ == "__main__":
    sys.exit(main())
