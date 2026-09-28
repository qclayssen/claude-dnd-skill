"""
test_campaign_brain.py — campaign_facts / graph_seed / brain / check.

Covers the three grounding modules added for the anti-hallucination work:

  campaign_facts  the deterministic parse layer over state.md / npcs.md /
                  world.md / source-index.md. Every test here pins a bug that
                  actually bit during development, because a parser that
                  silently returns {} looks exactly like a parser that works.

  graph_seed      deterministic graph seeding from the index files, the stance
                  mapping, and idempotence.

  brain           the generated always-hot brief: verbatim pinned facts, on-scene
                  cast, pending-fork extraction, and the word budget.

  check           the grounding gate: it must catch an invented name and must
                  NOT fire on canon. The false-positive tests are the important
                  ones — a gate that cries wolf mid-scene gets disabled, and a
                  disabled gate protects nothing.

Run from repo root:
    python3 -m unittest tests.test_campaign_brain -v
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
SKILL = REPO / "skills" / "dnd" if (REPO / "skills" / "dnd").is_dir() else REPO
sys.path.insert(0, str(SKILL / "scripts"))

import brain  # noqa: E402
import campaign_facts as cf  # noqa: E402
import campaign_graph as cg  # noqa: E402
import check  # noqa: E402
import graph_seed  # noqa: E402

SCRIPTS = SKILL / "scripts"

# A small but structurally faithful campaign: nested bullets, bold
# pseudo-headings, a fenced YAML arc, a container `## Factions` with only `###`
# children, and a name collision (`Three Truths` at two levels). Every one of
# those are shapes real imported campaigns use in practice.
STATE_MD = """\
# Campaign: unittest-camp
**Created:** 2026-09-01  **Last session:** 3  **Session count:** 3  **Ruleset:** 2024

## Current Situation
- **Location:** the Saltmarsh docks, Greyhaven
- **In-world date/time:** 4th of Frostwane, midwinter
- **Party:** Brann - Halfling Rogue 3 | HP 21/21
- **Party status:** Brann alone

## Pinned Facts
- Brann never steals from the poor, only the greedy.
- The lighthouse keeper is dead. Do not say so until session 8.

## World State
- **In-world date:** 4th of Frostwane
- **Season:** deep winter  **Weather:** sleet
- **Threat arc stage:** 2 - Rising
- **Faction states:**
  - The Salt Guild: tightening the ledgers
  - The Quiet Hand: unknown

## Active Quests
*(none yet)*

## Open Threads & Rumours
- Who paid off the dockside debt?

## Faction Moves
*(none yet)*

## Live State Flags

**Cover:**
- posing as a dock clerk

**Faction stances** *(only list factions with non-neutral standing toward the party)*:
- The Salt Guild: hostile

**NPC dispositions** *(only list NPCs with changed or notable standing)*:
- Captain Iven Rask: -1 (distrusts Brann)
- Sister Halda: friendly

## Campaign Arc
```yaml
type: structured
current_act: 1
current_chapter: "2.1"
next_chapter: "2.2"
outstanding_beats: ["The Ledger Burning", "The Gull's Warning"]
steering_notes: >
  Open cold on the docks.
  Then cut to the guildhall before the tide turns.
```

## Session Flags
- autosave: on
- chaos_factor: 6

## DM Notes (hidden from players)
- debt_owed: 40 (to the Quiet Hand)
- gauge_choice: pending (2.3; confess / deny / flee)
- cipher_word: not yet coined (2.4)
- vess_lien: none
"""

NPCS_MD = """\
# NPCs: unittest-camp

| Name | Role | Faction | Location | Attitude | Notes |
|------|------|---------|----------|----------|-------|
| Captain Iven Rask | Harbour captain | The Salt Guild | Saltmarsh docks | unfriendly | Owes the Quiet Hand |
| Sister Halda | Archivist | independent | Greyhaven chapel | friendly | Knows the ledger's shape |
| Wick Odal | Fence | The Quiet Hand | anywhere | neutral | Never uses a real name |
| Mother Crane | Innkeeper | independent | The Drowned Bell | neutral | Feeds everyone |
| The Gull | Scryer | none | Greyhaven | hostile | Speaks in stolen weather |
"""

WORLD_MD = """\
# World: unittest-camp

## World Foundations
Greyhaven sits on a cold coast.

## The Settlement: Greyhaven
### Three Truths
The sea takes everyone.

## Factions

### The Salt Guild (*harbour authority*)
- **Goals:** keep the ledgers balanced.
- **Secret:** the harbourmaster skims the tithe.
- **Attitude toward party:** hostile

### The Quiet Hand (*smugglers' guild*)
- **Goals:** own every debt in Greyhaven.
- **Secret:** Rask is their clerk.
- **Attitude toward party:** neutral
"""

SOURCE_INDEX_MD = """\
# Source Index: unittest-camp

| Chapter | File | Scope |
|---------|------|-------|
| 2.1 | source/2.1.md | The Saltmarsh Debt: docks / guildhall |
| 2.2 | source/2.2.md | The Tide Turns: the Gull's roost |
| Supp | side-quests.md | Side quests and festivals |
| NPC | npc-files/*.md | Per-NPC deep files |
"""


def _write_campaign(root: pathlib.Path, name: str) -> pathlib.Path:
    d = root / "campaigns" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.md").write_text(STATE_MD, encoding="utf-8")
    (d / "npcs.md").write_text(NPCS_MD, encoding="utf-8")
    (d / "world.md").write_text(WORLD_MD, encoding="utf-8")
    (d / "source-index.md").write_text(SOURCE_INDEX_MD, encoding="utf-8")
    (d / "npcs-full.md").write_text(
        "# NPC entries\n\n## Captain Iven Rask\nHarbourmaster of Greyhaven.\n", encoding="utf-8")
    npc_files = d / "npc-files"
    npc_files.mkdir(exist_ok=True)
    (npc_files / "iven-rask.md").write_text("# Iven Rask\nGuardrails: never blinks.\n",
                                            encoding="utf-8")
    chars = d / "characters"
    chars.mkdir(exist_ok=True)
    (chars / "Brann.md").write_text(
        "# Brann\n\n## Identity\n- **Race:** Halfling | **Class:** Rogue 3 | **Level:** 3\n",
        encoding="utf-8")
    return d


class _CampaignCase(unittest.TestCase):
    """Base: an ephemeral campaign on disk with DND_CAMPAIGN_ROOT pointed at it."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.root = pathlib.Path(self._td.name)
        self.campaign = f"unittest-brain-{os.getpid()}-{id(self)}"
        self.camp = _write_campaign(self.root, self.campaign)
        self._old = os.environ.get("DND_CAMPAIGN_ROOT")
        os.environ["DND_CAMPAIGN_ROOT"] = str(self.root)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._old is None:
            os.environ.pop("DND_CAMPAIGN_ROOT", None)
        else:
            os.environ["DND_CAMPAIGN_ROOT"] = self._old

    def _run(self, script, *args, expect=None):
        env = os.environ.copy()
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / script), *args],
            capture_output=True, encoding="utf-8", text=True, env=env,
        )
        if expect is not None:
            self.assertEqual(r.returncode, expect,
                             msg=f"stdout={r.stdout}\nstderr={r.stderr}")
        return r


# ── campaign_facts: the parse layer ─────────────────────────────────────────

class ParseStateTests(unittest.TestCase):

    def test_counters_and_kv(self):
        st = cf.parse_state(STATE_MD)
        self.assertEqual(st["session_count"], 3)
        self.assertEqual(st["ruleset"], "2024")
        self.assertEqual(st["situation"]["Location"], "the Saltmarsh docks, Greyhaven")
        self.assertEqual(st["world_state"]["Season"], "deep winter  Weather: sleet")

    def test_nested_bullets_do_not_clobber_parent_keys(self):
        """`## World State` nests a `Faction states:` list. Hoisting those into
        the parent dict used to overwrite the real In-world date / Season keys."""
        st = cf.parse_state(STATE_MD)
        self.assertIn("In-world date", st["world_state"])
        self.assertIn("Season", st["world_state"])
        self.assertNotIn("The Salt Guild", st["world_state"])

    def test_bold_pseudo_headings(self):
        """Live State Flags uses `**Cover:**`, not `### Cover`."""
        st = cf.parse_state(STATE_MD)
        self.assertEqual(st["cover"], ["posing as a dock clerk"])
        self.assertEqual(st["faction_stances"]["The Salt Guild"], "hostile")
        self.assertIn("Captain Iven Rask", st["npc_dispositions"])
        self.assertIn("Sister Halda", st["npc_dispositions"])

    def test_pinned_facts_verbatim(self):
        """Pinned Facts must survive byte-for-byte — paraphrasing them is the
        drift this whole system exists to prevent."""
        st = cf.parse_state(STATE_MD)
        self.assertEqual(len(st["pinned_facts"]), 2)
        self.assertEqual(st["pinned_facts"][0],
                         "Brann never steals from the poor, only the greedy.")

    def test_fenced_yaml_arc(self):
        st = cf.parse_state(STATE_MD)
        self.assertEqual(st["arc"]["current_chapter"], "2.1")
        self.assertEqual(st["arc"]["next_chapter"], "2.2")
        self.assertIn("Ledger Burning", st["arc"]["outstanding_beats"])
        # The folded `>` scalar is the load-bearing field; it must not be ">".
        self.assertIn("Open cold on the docks", st["arc"]["steering_notes"])
        self.assertNotEqual(st["arc"]["steering_notes"], ">")

    def test_fence_does_not_create_headings(self):
        """A `#` inside the YAML fence is not a heading."""
        sec = cf.sections(STATE_MD)
        self.assertNotIn("type", sec)
        self.assertIn("Current Situation", sec)

    def test_duplicate_heading_first_wins(self):
        """world.md has `### Three Truths` twice. A later duplicate must not
        clobber an earlier section — that silently deleted `## Factions`."""
        sec = cf.sections(WORLD_MD)
        self.assertIn("Three Truths", sec)
        self.assertIn("Factions", sec)
        self.assertIn("The sea takes everyone", sec["Three Truths"])

    def test_dm_notes(self):
        st = cf.parse_state(STATE_MD)
        self.assertEqual(st["dm_notes"]["debt_owed"], "40 (to the Quiet Hand)")
        self.assertEqual(st["session_flags"]["chaos_factor"], "6")


class ParseNpcsIndexTests(unittest.TestCase):

    def setUp(self):
        self.rows = cf.parse_npcs_index(NPCS_MD)

    def test_all_data_rows(self):
        self.assertEqual(len(self.rows), 5)
        self.assertEqual(self.rows[0]["name"], "Captain Iven Rask")

    def test_separator_row_dropped(self):
        """The `|---|---|` row must be skipped. Keeping it appends a junk row
        literally named `------` and, worse, the inverted test also dropped
        every real row."""
        self.assertNotIn("------", [r["name"] for r in self.rows])

    def test_cells(self):
        r = self.rows[0]
        self.assertEqual(r["faction"], "The Salt Guild")
        self.assertEqual(r["attitude"], "unfriendly")


class ParseFactionsTests(unittest.TestCase):

    def setUp(self):
        self.f = {x["name"]: x for x in cf.parse_factions(WORLD_MD)}

    def test_container_section_found(self):
        """`## Factions` has only `###` children, so the flat heading map returns
        an empty body for it. Scanned directly instead."""
        self.assertIn("The Salt Guild", self.f)
        self.assertEqual(len(self.f), 2)

    def test_labels_and_attitude(self):
        g = self.f["The Quiet Hand"]
        self.assertEqual(g["kind"], "smugglers' guild")
        self.assertEqual(g["labels"]["Secret"], "Rask is their clerk.")
        self.assertEqual(g["attitude"], "neutral")


class ParseSourceIndexTests(unittest.TestCase):

    def test_chapters_and_glob_rows(self):
        ch = cf.parse_source_index(SOURCE_INDEX_MD)
        self.assertEqual(len(ch), 4)
        self.assertTrue(ch[0]["is_file"])
        self.assertTrue(ch[2]["is_file"])   # a single named file
        self.assertFalse(ch[3]["is_file"])  # a glob, not one chapter file


# ── graph_seed ──────────────────────────────────────────────────────────────

class StanceMappingTests(unittest.TestCase):

    def test_named_stances(self):
        self.assertEqual(graph_seed.map_stance("friendly (wants him)"), "friendly")
        self.assertEqual(graph_seed.map_stance("allied"), "allied")

    def test_unfriendly_beats_friendly(self):
        """'unfriendly' contains 'friendly'. Longest/earliest-token wins, or a
        hostile NPC is recorded as friendly."""
        self.assertEqual(graph_seed.map_stance("surface unfriendly"), "suspicious")
        self.assertEqual(graph_seed.map_stance("unfriendly (secretly protective)"),
                         "suspicious")

    def test_numeric_scale(self):
        self.assertEqual(graph_seed.map_stance("+2 (guardian)"), "allied")
        self.assertEqual(graph_seed.map_stance("+1"), "friendly")
        self.assertEqual(graph_seed.map_stance("0"), "neutral")
        self.assertEqual(graph_seed.map_stance("-1 (distrusts Brann)"), "suspicious")
        self.assertEqual(graph_seed.map_stance("-3"), "hostile")

    def test_unrecognized_returns_empty_not_neutral(self):
        """An unrecognized stance must NOT become a confident `neutral` edge —
        that is inventing canon, which is the bug this guards against."""
        self.assertEqual(graph_seed.map_stance("complicated"), "")
        self.assertEqual(graph_seed.map_stance(""), "")

    def test_surface_flag(self):
        self.assertEqual(graph_seed._surface_flag("surface unfriendly (secretly protective)"),
                         "surface+secret")
        self.assertEqual(graph_seed._surface_flag("friendly"), "")


class FactionMatchTests(unittest.TestCase):

    KNOWN = ["The Salt Guild", "The Quiet Hand"]

    def test_exact(self):
        self.assertEqual(graph_seed.match_faction("The Salt Guild", self.KNOWN),
                         "The Salt Guild")

    def test_parenthetical_variant(self):
        self.assertEqual(graph_seed.match_faction("The Quiet Hand (smugglers)",
                                                  self.KNOWN), "The Quiet Hand")

    def test_partial_and_fuzzy(self):
        self.assertEqual(graph_seed.match_faction("Salt Guild", self.KNOWN),
                         "The Salt Guild")
        self.assertEqual(graph_seed.match_faction("Qiet Hand", self.KNOWN),
                         "The Quiet Hand")

    def test_unknown(self):
        self.assertEqual(graph_seed.match_faction("The Gilded Wolves", self.KNOWN), "")


class SplitLocationTests(unittest.TestCase):

    def test_first_segment_canonical(self):
        place, aliases = graph_seed.split_location(
            "First-year hall, frog pond (Witherbloom dorms from Y2)")
        self.assertEqual(place, "First-year hall")
        self.assertIn("frog pond", aliases)

    def test_single(self):
        self.assertEqual(graph_seed.split_location("Owl Roost"), ("Owl Roost", []))


class GraphSeedTests(_CampaignCase):

    def test_dry_run_writes_nothing(self):
        r = self._run("graph_seed.py", "-c", self.campaign, expect=0)
        self.assertFalse((self.camp / "graph.json").exists())
        self.assertIn("dry run", r.stderr)
        self.assertIn("proposed:", r.stdout)

    def test_apply_creates_graph(self):
        self._run("graph_seed.py", "-c", self.campaign, "--apply", expect=0)
        data = json.loads((self.camp / "graph.json").read_text(encoding="utf-8"))
        names = {n["name"] for n in data["nodes"]}
        self.assertIn("Captain Iven Rask", names)
        self.assertIn("The Salt Guild", names)
        self.assertIn("Brann", names)
        self.assertIn("The Party", names)

    def test_party_node_id_is_the_shared_constant(self):
        """Every disposition edge keys off `party`. A `party_the_party` id
        would create a second party node and orphan every stance."""
        self._run("graph_seed.py", "-c", self.campaign, "--apply", expect=0)
        data = json.loads((self.camp / "graph.json").read_text(encoding="utf-8"))
        self.assertIn(cg.PARTY_NODE_ID, {n["id"] for n in data["nodes"]})
        for e in data["edges"]:
            if e["type"] in ("disposition", "standing"):
                self.assertEqual(e["from"], cg.PARTY_NODE_ID)

    def test_edges_have_provenance(self):
        self._run("graph_seed.py", "-c", self.campaign, "--apply", expect=0)
        data = json.loads((self.camp / "graph.json").read_text(encoding="utf-8"))
        noted = [e for e in data["edges"] if e.get("note")]
        self.assertTrue(noted, "every seeded edge should record its source line")

    def test_idempotent(self):
        self._run("graph_seed.py", "-c", self.campaign, "--apply", expect=0)
        first = json.loads((self.camp / "graph.json").read_text(encoding="utf-8"))
        r = self._run("graph_seed.py", "-c", self.campaign, "--allow-empty", expect=0)
        second = json.loads((self.camp / "graph.json").read_text(encoding="utf-8"))
        self.assertEqual(len(first["nodes"]), len(second["nodes"]))
        self.assertEqual(len(first["edges"]), len(second["edges"]))
        self.assertIn("proposed: 0 nodes, 0 edges", r.stdout)

    def test_scene_context_reachable(self):
        self._run("graph_seed.py", "-c", self.campaign, "--apply", expect=0)
        r = self._run("campaign_graph.py", "scene-context", "--campaign", self.campaign,
                      "--place", "Saltmarsh docks", "--present", "Captain Iven Rask",
                      "--hops", "2", expect=0)
        self.assertIn("Captain Iven Rask", r.stdout)
        self.assertIn("relationships", r.stdout)

    def test_unresolved_reported_not_guessed(self):
        """An unresolvable faction must be reported, never invented."""
        (self.camp / "npcs.md").write_text(
            NPCS_MD + "\n| Oddball | Drifter | The Gilded Wolves | nowhere | neutral | x |\n",
            encoding="utf-8")
        prop = graph_seed.build(self.campaign)
        self.assertIn(("faction", "Oddball", "The Gilded Wolves"), prop["unresolved"])

    def test_stance_conflict_detected(self):
        """world.md and state.md both state a party standing. A mismatch is
        drift, and must surface at seed time."""
        conflicted = WORLD_MD.replace(
            "- **Attitude toward party:** neutral",
            "- **Attitude toward party:** allied")
        (self.camp / "world.md").write_text(conflicted, encoding="utf-8")
        prop = graph_seed.build(self.campaign)
        self.assertTrue(any(c["faction"] == "The Quiet Hand"
                            for c in prop["stance_conflicts"]))


# ── brain ──────────────────────────────────────────────────────────────────

class BrainBuildTests(_CampaignCase):

    def _brain(self, **kw):
        text, meta = brain.build(self.campaign, **kw)
        return text, meta

    def test_writes_file(self):
        r = self._run("brain.py", "-c", self.campaign, expect=0)
        self.assertTrue((self.camp / "brain.md").exists())
        self.assertIn("wrote", r.stderr)

    def test_deterministic(self):
        """Same inputs -> same bytes, modulo the generated-at stamp."""
        def strip_stamp(s):
            return "\n".join(line for line in s.splitlines()
                             if not line.startswith("**Generated:**"))
        a, _ = self._brain()
        b, _ = self._brain()
        self.assertEqual(strip_stamp(a), strip_stamp(b))

    def test_pinned_facts_verbatim(self):
        text, _ = self._brain()
        self.assertIn("Brann never steals from the poor, only the greedy.", text)
        self.assertIn("The lighthouse keeper is dead. Do not say so until session 8.", text)

    def test_on_scene_cast_found(self):
        """The state's Location ('the Saltmarsh docks, Greyhaven') and the index
        row ('Saltmarsh docks') are written at different granularities and
        neither is a prefix of the other. A one-directional match found nothing
        and silently emptied this section."""
        text, _ = self._brain()
        self.assertIn("At the current location", text)
        self.assertIn("Captain Iven Rask", text)

    def test_dispositions_resolve_despite_title_mismatch(self):
        """Titles in the two files need not agree: state.md's Live State Flags
        are free text and commonly say 'Iven Rask' where npcs.md says
        'Captain Iven Rask'. Dispositions are read from the graph, so joining
        names across the two files cannot silently drop them."""
        (self.camp / "npcs.md").write_text(
            NPCS_MD.replace("Captain Iven Rask | Harbour captain",
                            "Hon. Captain Iven Rask | Harbour captain"),
            encoding="utf-8")
        self._run("graph_seed.py", "-c", self.campaign, "--apply", expect=0)
        text, _ = self._brain()
        self.assertIn("Hon. Captain Iven Rask", text)
        self.assertIn("suspicious", text)  # -1 mapped through the graph

    def test_pending_forks_extracted(self):
        text, _ = self._brain()
        self.assertIn("PENDING CANON DECISIONS", text)
        self.assertIn("gauge_choice", text)
        self.assertIn("cipher_word", text)

    def test_pending_excludes_resolved_none(self):
        """`vess_lien: none` is a *settled* value, not a pending fork. The
        'none (' pattern must not catch a bare 'none'."""
        text, _ = self._brain()
        self.assertNotIn("vess_lien", text)

    def test_relationships_prioritised(self):
        """A seeded graph is mostly based_at/member_of. Dispositions must sort
        first or they drown in structural edges."""
        self._run("graph_seed.py", "-c", self.campaign, "--apply", expect=0)
        text, _ = self._brain()
        rel = text.split("## RELATIONSHIPS", 1)[1].split("## ", 1)[0]
        first = rel.strip().splitlines()[1]
        self.assertIn("disposition", first)

    def test_deep_files_indexed(self):
        text, _ = self._brain()
        self.assertIn("iven-rask", text)   # npc-files stem, listed for pre-scene reads
        self.assertIn("npcs-full.md", text)

    def test_word_budget_drops_low_priority_first(self):
        text, meta = self._brain(max_words=60)
        self.assertTrue(meta["dropped"])
        self.assertIn("TRUNCATED", text)
        # Pinned Facts are `always` and must survive any budget.
        self.assertIn("PINNED FACTS", text)
        self.assertNotIn("## RELATIONSHIPS", text)

    def test_markdown_rule_not_leaked_into_faction_values(self):
        text, _ = self._brain()
        self.assertNotIn("---", text.split("## FACTIONS", 1)[1].split("## ", 1)[0])

    def test_staleness_detected(self):
        self._run("brain.py", "-c", self.campaign, expect=0)
        self._run("brain.py", "-c", self.campaign, "--check", expect=0)
        os.utime(self.camp / "npcs.md", None)
        r = self._run("brain.py", "-c", self.campaign, "--check", expect=1)
        self.assertIn("STALE", r.stderr)

    def test_check_missing_brain_is_stale(self):
        r = self._run("brain.py", "-c", self.campaign, "--check", expect=1)
        self.assertIn("STALE", r.stderr)

    def test_missing_index_files_degrade_not_crash(self):
        for f in ("npcs.md", "world.md", "source-index.md"):
            (self.camp / f).unlink()
        text, meta = self._brain()
        self.assertIn("NOW", text)
        self.assertIn("npcs.md", meta["facts_missing"])


# ── check ──────────────────────────────────────────────────────────────────

class GroundingCheckTests(_CampaignCase):

    def _seed(self):
        self._run("graph_seed.py", "-c", self.campaign, "--apply", expect=0)

    def test_canon_names_clean(self):
        rep = check.check_draft(
            self.campaign,
            "Captain Iven Rask studies the tide charts. Sister Halda waits in the "
            "chapel, and Mother Crane has already put the kettle on.")
        self.assertEqual(rep["unknown"], [])

    def test_invented_name_caught(self):
        rep = check.check_draft(self.campaign, "Archivist Corrin waits by the door.")
        self.assertEqual([u["name"] for u in rep["unknown"]], ["Corrin"])

    def test_invented_name_offers_near_miss(self):
        rep = check.check_draft(self.campaign, "Captain Rask greets Vhal at the door.")
        names = {u["name"]: u for u in rep["unknown"]}
        self.assertIn("Vhal", names)
        self.assertTrue(names["Vhal"].get("did_you_mean"),
                        "a one-letter miss is a typo, not an invention")

    def test_titles_are_not_names(self):
        rep = check.check_draft(
            self.campaign, "Captain Iven Rask arrives. Mother Crane waves.")
        self.assertEqual(rep["unknown"], [])

    def test_month_names_not_flagged(self):
        rep = check.check_draft(self.campaign, "November rain. Chapter One. Roll Perception.")
        self.assertEqual(rep["unknown"], [])

    def test_allow_list(self):
        rep = check.check_draft(self.campaign, "Archivist Corrin waits.",
                                allow=["Corrin"])
        self.assertEqual(rep["unknown"], [])

    def test_vocabulary_covers_deep_files(self):
        """A name that exists only in npc-files/ is still canon. The vocabulary
        scans the whole corpus, not just the index."""
        rep = check.check_draft(self.campaign, "The Drowned Bell is shuttered.")
        self.assertEqual(rep["unknown"], [])

    def test_disposition_drift_advisory(self):
        self._seed()
        rep = check.check_draft(self.campaign, "Sister Halda sneers and refuses him.")
        self.assertTrue(rep["disposition_drift"])
        self.assertEqual(rep["disposition_drift"][0]["name"], "Sister Halda")

    def test_disposition_can_be_disabled(self):
        self._seed()
        rep = check.check_draft(self.campaign, "Sister Halda sneers and refuses him.",
                                check_disposition=False)
        self.assertEqual(rep["disposition_drift"], [])

    def test_strict_exit_codes(self):
        self._run("check.py", "-c", self.campaign, "--text", "Archivist Corrin waits.",
                  expect=0)                      # advisory by default
        self._run("check.py", "-c", self.campaign, "--text", "Archivist Corrin waits.",
                  "--strict", expect=1)
        self._run("check.py", "-c", self.campaign, "--text", "Archivist Corrin waits.",
                  "--allow", "Corrin", "--strict", expect=0)
        self._run("check.py", "-c", self.campaign, "--text", "Mother Crane nods.",
                  "--strict", expect=0)

    def test_json_output(self):
        r = self._run("check.py", "-c", self.campaign, "--text", "Archivist Corrin waits.",
                      "--json", expect=0)
        data = json.loads(r.stdout)
        self.assertEqual(data["unknown"][0]["name"], "Corrin")

    def test_stdin_source(self):
        env = os.environ.copy()
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "check.py"), "-c", self.campaign, "--strict"],
            input="Archivist Corrin waits.", capture_output=True, encoding="utf-8",
            text=True, env=env)
        self.assertEqual(r.returncode, 1)

    def test_empty_draft_is_usage_error(self):
        self._run("check.py", "-c", self.campaign, "--text", "   ", expect=2)

    def test_derived_brain_does_not_feed_vocabulary(self):
        """brain.md is generated from the other sources. If it fed the
        vocabulary, one hand-edit would permanently legitimize a name — and the
        check would then pass on exactly the hallucination it exists to catch."""
        self._run("brain.py", "-c", self.campaign, expect=0)
        bp = self.camp / "brain.md"
        bp.write_text(bp.read_text(encoding="utf-8")
                      + "\n- The Drowned Bell is run by Wick Odal's cousin, Gormund Bale.\n",
                      encoding="utf-8")
        rep = check.check_draft(self.campaign, "Gormund Bale is waiting by the door.")
        self.assertEqual([u["name"] for u in rep["unknown"]], ["Gormund", "Bale"])

    def test_no_false_positives_on_canon_corpus(self):
        """The decisive test. Every capitalized token in every campaign file is
        by definition in the vocabulary, so this must be exactly zero. A gate
        that fires on canon gets switched off, and a switched-off gate is
        worse than no gate because it looks like protection."""
        total = 0
        for p in sorted(self.camp.rglob("*.md")):
            if p.name == "brain.md" or check._is_codeish(p):
                continue
            text = p.read_text(encoding="utf-8")
            rep = check.check_draft(self.campaign, text, check_disposition=False)
            total += len(rep["unknown"])
        self.assertEqual(total, 0, f"{total} false positives on canon text")


if __name__ == "__main__":
    unittest.main()
