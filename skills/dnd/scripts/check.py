#!/usr/bin/env python3
"""
check.py — pre-narration grounding check.

Every other anti-hallucination instruction in this system is a request to the
model: "read their full entry before voicing them", "stay consistent with
established canon". Requests decay. This is a mechanical gate instead.

The core check is simple and has a very low false-positive rate:

    a capitalized name in the draft that appears NOWHERE in the campaign corpus
    is a name the DM invented.

The corpus is the whole campaign directory — state.md, npcs.md, npcs-full.md,
npc-files/*.md, world.md, arc.md, source/*.md, side-quests, and so on. For a
typical long imported campaign that is hundreds of KB across ~100 files, so
"Magister Ilva Crane" and "Ona Wick" and every name that appears only in a
single chapter are all in the vocabulary. A DM that invents "Archivist Corrin"
produces a capitalized token that is in no file, because no such person was ever
written down. That is the whole trick: the campaign is its own vocabulary, and
drift is what falls outside it.

Secondary check: disposition drift. If the graph says an NPC is `suspicious`
with a surface/secret split, and the draft narrates them warmly, that is worth
a look. Advisory only — tone words are noisy.

It does NOT try to verify rules, dice, or lore. It is a name-and-relationship
gate, and it is deliberately conservative: a false accusation mid-scene costs
far more than a missed one.

Exit codes:
    0  clean (or warnings only, without --strict)
    1  unknown proper nouns found under --strict
    2  usage error

Usage:
    python3 check.py -c my-campaign --text "Ilva opens the ledger."
    python3 check.py -c my-campaign --file draft.md
    cat draft.md | python3 check.py -c my-campaign
    python3 check.py -c my-campaign --file draft.md --strict --json
    python3 check.py -c my-campaign --file draft.md --allow Orlan --allow Corrin
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import brain
import campaign_facts as cf
import campaign_graph as cg
from utf8io import read_text, TextDecodeError

# Words that are legitimately capitalized in prose but are never names. Kept
# deliberately small: every entry here is a word whose presence in a draft
# would otherwise be reported as an invented name. Adding a word silences a
# class of false positives, so this list should only grow when a real false
# positive is observed — not speculatively.
STOPWORDS = {
    # sentence / clause starters that get title-cased
    "The", "A", "An", "And", "But", "Or", "Nor", "For", "Yet", "So", "If",
    "Then", "When", "While", "Where", "Which", "Who", "Whom", "What", "Why",
    "How", "That", "This", "These", "Those", "There", "Here", "Now", "Not",
    "No", "Yes", "He", "She", "It", "They", "We", "You", "I", "His", "Her",
    "Its", "Their", "Our", "Your", "My", "As", "At", "By", "Do", "Does",
    "Did", "Have", "Has", "Had", "Is", "Are", "Was", "Were", "Be", "Been",
    "Being", "Can", "Could", "Will", "Would", "Shall", "Should", "May",
    "Might", "Must", "One", "Two", "Three", "Each", "Every", "Some", "Most",
    "All", "Both", "Few", "Many", "Much", "More", "Less", "Other", "Another",
    "Even", "Never", "Always", "Still", "Just", "Only", "Also", "Very",
    "Too", "Because", "Before", "After", "Until", "Unless", "Once", "Again",
    # D&D / system vocabulary that shows up capitalised in play text
    "Hit", "Miss", "Damage", "Attack", "Roll", "Save", "Saving", "Critical",
    "Nat", "Advantage", "Disadvantage", "Level", "Spell", "Bonus", "Armor",
    "Armour", "Initiative", "Round", "Turn", "Encounter", "Combat", "Death",
    "Saving", "Perception", "Arcana", "Investigation", "Insight", "Persuasion",
    "Deception", "Intimidation", "Performance", "Sleight", "Stealth", "Survival",
    "History", "Nature", "Religion", "Medicine", "First", "Second", "Third",
    "Fourth", "Fifth", "Day", "Night", "Week", "Month", "Year", "Morning",
    "Evening", "Dusk", "Dawn", "Midnight", "Rest", "Long", "Short", "Class",
    "Race", "Background", "Alignment", "Feat", "Feature", "Session", "Chapter",
    "Arc", "Scene", "Beat", "Hook", "Thread", "Faction", "Canon", "Meta",
    "Note", "Secret", "Public", "Private", "GM", "DM", "NPC", "PC", "HP", "AC",
    "XP", "DC", "GM-only", "Warning", "Important", "Note:", "Rule", "House",
    "Good", "Evil", "Lawful", "Chaotic", "Neutral", "True", "False", "Yes",
    "Left", "Right", "Side", "East", "West", "North", "South", "Up", "Down",
    "Out", "Over", "Under", "Back", "Away", "Along", "Around", "Through",
    "Order", "Hand", "Point", "Line", "Circle", "Square", "Number", "Amount",
    "Half", "Quarter", "Double", "Triple", "Final", "Last", "Next", "Another",
    # Real-world calendar names. Campaigns almost always run on an invented
    # calendar, so these never appear in canon and always false-positive.
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "Weekend", "Today", "Tomorrow", "Yesterday",
}

# Honorifics and titles. A capitalised token that is only ever a title is not a
# name, and the name attached to it is the real candidate.
TITLES = {
    "Magister", "Magistra", "Prof", "Professor", "Dean", "Coach", "Lady",
    "Lord", "Sir", "Dame", "Captain", "Colonel", "Master", "Mistress", "Dr",
    "Doctor", "Archmage", "Archdruid", "Hierophant", "Guildmaster", "Warden",
    "Keeper", "Tallykeeper", "Esteemed", "Honoured", "Honored", "Reverend",
    "Father", "Mother", "Brother", "Sister", "Father", "Baron", "Count",
    "Countess", "Duke", "Duchess", "Prince", "Princess", "Queen", "King",
    "Oracle", "Archai", "Archaic", "Adjunct", "Instructor", "Tutor",
}

# Stance vocabulary for the advisory disposition-drift check. Grouped by the
# direction it implies, so a warm word against a `hostile` node is flagged while
# a `suspicious` (surface) node is given more latitude.
WARM = {
    "warmly", "kindly", "gently", "smiles", "smiled", "smiling", "laughs",
    "nods", "agreed", "approves", "welcomes", "helped", "helps", "trusts",
    "trusted", "affectionate", "fond", "grins", "beam", "reassures", "soothes",
    "comforts", "thanks", "grateful", "friendly", "allied", "loyal",
}
COLD = {
    "coldly", "hatefully", "snarls", "sneers", "refuses", "refused", "spits",
    "hostile", "enemy", "enemies", "threatens", "threatened", "glares", "hates",
    "hated", "hostility", "betrays", "betrayed", "attacks", "lied", "lies",
    "mocking", "mock", "contempt", "spurns", "rejects", "denies",
}

# Corpus files scanned to build the known-name vocabulary. `source/` and the
# per-NPC/glossary files are included deliberately: a name that exists only in
# one chapter of the module is still real canon, and excluding it would make the
# check cry wolf on every callback.
_SKIP_DIRS = {".git", "__pycache__"}
_SKIP_SUFFIX = (".json", ".patch", ".png", ".jpg", ".jpeg", ".gif", ".svg")


def _is_codeish(path: pathlib.Path) -> bool:
    if path.suffix.lower() in _SKIP_SUFFIX:
        return True
    return any(part in _SKIP_DIRS for part in path.parts)


def build_vocabulary(campaign: str, extra_files=None) -> set:
    """Every capitalized word token that appears anywhere in the campaign corpus.

    Returns a set of lowercase tokens. A word the DM invented is absent; a word
    that was ever written down is present, regardless of which file.

    `brain.md` is excluded even though it lives in the campaign dir: it is
    *derived* from the other sources, so letting it feed the vocabulary would
    make the ground truth depend on a generated file — one hand-edit to
    `brain.md` would permanently legitimize that name, and the check would then
    pass on exactly the hallucination it exists to catch.
    """
    root = cf.campaign_dir(campaign)
    words: set = set()
    paths = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not _is_codeish(p) and p.suffix.lower() in (".md", ".txt"):
            paths.append(p)
    for p in (extra_files or []):
        if p and pathlib.Path(p).exists():
            paths.append(pathlib.Path(p))
    derived = {brain.BRAIN_NAME, "graph.json"}
    for p in paths:
        if p.name in derived or p.suffix.lower() in (".json",):
            continue
        try:
            text = read_text(p)
        except (TextDecodeError, OSError):
            continue
        for tok in _TOKEN_RE.findall(text):
            words.add(tok.lower())
    return words


# A capitalized word token: an initial capital followed by lowercase letters,
# allowing internal apostrophes/hyphens (Oona, Tulk, Raven's) and diacritics.
_TOKEN_RE = re.compile(r"\b[A-Z][a-zà-ÿ'’\-]*\b")


def _known_names(campaign: str) -> set:
    """Names the index layer knows about, lowercase. Subtracted from candidates
    so a real NPC is never reported even if the corpus scan somehow missed it."""
    names: set = set()
    graph = cg._load(campaign)
    for n in graph.get("nodes", []):
        for part in re.split(r"[\s,]+", n.get("name", "")):
            if part:
                names.add(part.lower().strip("()"))
    for row in cf.load(campaign)["npcs"]:
        for part in re.split(r"[\s,]+", row["name"]):
            if part:
                names.add(part.lower())
    return names


def extract_candidates(text: str) -> list:
    """Capitalized tokens in the draft that could plausibly be a proper noun.

    Filters, in order: multi-character minimum; not a stopword; not a bare
    title; not the sole capital of a title+name pair (the name is then taken
    from the token after it); not immediately repeated.
    """
    out: list = []
    toks = _TOKEN_RE.findall(text)
    for i, tok in enumerate(toks):
        if len(tok) < 2:
            continue
        if tok in STOPWORDS:
            continue
        if tok in TITLES:
            # `Magister Ilva Crane` — the title is not the name; keep scanning
            # and let the following token stand on its own.
            continue
        if i > 0 and toks[i - 1] in TITLES:
            # previous token was a title, so this is the name itself
            pass
        key = tok.lower()
        if key in out_keys(out):
            continue
        out.append(tok)
    return out


def out_keys(items):
    return {x.lower() for x in items}


def _position_map(text: str) -> dict:
    """lowercase token -> first character offset, for reporting context."""
    pos = {}
    for m in re.finditer(r"[A-Za-z][A-Za-z'’\-]*", text):
        low = m.group(0).lower()
        pos.setdefault(low, m.start())
    return pos


def _context(text: str, offset: int, width: int = 45) -> str:
    a = max(0, offset - width)
    b = min(len(text), offset + width)
    frag = text[a:b].replace("\n", " ").strip()
    return ("…" if a else "") + frag + ("…" if b < len(text) else "")


def _suggest(name: str, known: set, vocab: set, limit: int = 2) -> list:
    """Known names plausibly intended by `name`, best first.

    Two passes. First a whole-string ratio against real multi-word names
    (catches 'Tran Ashdown' vs 'Tarn Ashdown'). Then a token-level
    pass: any known word within edit distance 1 of any word of `name` pulls in
    the full name that owns it. The second pass is what actually catches the
    single-token typos that matter most, because those are also the ones a
    whole-string ratio scores worst.
    """
    import campaign_facts as cf
    low = name.lower()
    out: list = []

    multi = sorted(n for n in known if " " in n)
    for cand in cf.close_names(low, multi, cap=3)[:limit]:
        if cand not in out:
            out.append(cand)

    words = [w for w in re.split(r"\s+", low) if w]
    kw = sorted(known | vocab)
    near_words = set()
    for w in words:
        # cap 2, not 1: a short name misspelled by two characters is still
        # obviously a typo of canon ('Tran' for 'Tarn' is a transposition *and*
        # a substitution), and suggestions are advisory — a spurious one costs
        # nothing, a missed typo gets narrated into the story.
        for k in cf.close_names(w, kw, cap=2):
            if k != w:
                near_words.add(k)
    if near_words:
        for cand in sorted(n for n in known
                           if any(w in n.split() for w in near_words)):
            if cand not in out:
                out.append(cand)
    return out[:limit]


def check_draft(campaign: str, draft: str, allow=None, check_disposition=True) -> dict:
    """Run the grounding checks. Returns a report dict; never raises on content."""
    allow_l = {a.lower() for a in (allow or [])}
    vocab = build_vocabulary(campaign)
    known = _known_names(campaign)
    pos = _position_map(draft)
    low_draft = draft.lower()

    unknown: list = []
    for cand in extract_candidates(draft):
        key = cand.lower()
        if key in allow_l or key in known or key in vocab:
            continue
        off = pos.get(key)
        unknown.append({
            "name": cand,
            "context": _context(draft, off) if off is not None else "",
            "offset": off,
        })

    # Near-miss suggestions. A token that is *almost* a real name is far more
    # likely a drift/typo of canon than a genuine invention, and the fix differs:
    # correct the spelling, versus write the new NPC into the index.
    #
    # Whole-string ratio alone misses the common case — 'Tran' vs 'Tarn' scores
    # 0.8, and 'Qiet Hand' vs 'Quiet Hand' scores below any useful cutoff
    # because of the length gap. Token-level comparison catches both: find the
    # known *word* that is one edit away, then surface the full known name
    # containing it.
    for u in unknown:
        u["did_you_mean"] = _suggest(u["name"], known, vocab)

    # Disposition drift (advisory).
    drift: list = []
    if check_disposition:
        graph = cg._load(campaign)
        for n in graph.get("nodes", []):
            if n.get("type") not in ("npc", "pc"):
                continue
            name = n.get("name", "")
            if not name or name.lower() not in low_draft:
                continue
            stances = [e for e in graph.get("edges", [])
                       if e.get("type") == "disposition" and e.get("to") == n["id"]
                       and cg._edge_active_at(e, None)]
            if not stances:
                continue
            level = stances[-1].get("level", "")
            sentence = _sentence_around(low_draft, name.lower())
            warm = any(w in sentence for w in WARM)
            cold = any(w in sentence for w in COLD)
            if level in ("hostile",) and warm:
                drift.append({"name": name, "graph_level": level, "draft_tone": "warm",
                              "excerpt": _context(draft, pos.get(name.lower(), 0), 70)})
            elif level in ("allied", "friendly") and cold:
                drift.append({"name": name, "graph_level": level, "draft_tone": "cold",
                              "excerpt": _context(draft, pos.get(name.lower(), 0), 70)})

    return {
        "campaign": campaign,
        "unknown": unknown,
        "disposition_drift": drift,
        "vocab_size": len(vocab),
        "checked_tokens": len(_TOKEN_RE.findall(draft)),
    }


def _sentence_around(low_text: str, needle: str) -> str:
    a = low_text.find(needle)
    if a < 0:
        return ""
    start = max(low_text.rfind(".", 0, a), low_text.rfind("\n", 0, a)) + 1
    end_candidates = [i for i in (low_text.find(".", a), low_text.find("\n", a)) if i > 0]
    end = min(end_candidates) if end_candidates else len(low_text)
    return low_text[start:end]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--campaign", required=True)
    # Not `required=True`: with neither --text nor --file the draft is read from
    # stdin, so `cat draft.md | check.py` works. Requiring the group would make
    # the piped form — the natural one for a pre-narration hook — impossible.
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--text", help="draft narration to check")
    src.add_argument("--file", help="file containing the draft")
    ap.add_argument("--allow", action="append", default=[],
                    help="accept this name as valid (repeatable)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when any unknown proper noun is found")
    ap.add_argument("--no-disposition", action="store_true",
                    help="skip the advisory disposition-drift check")
    ap.add_argument("--json", action="store_true", dest="as_json")
    a = ap.parse_args()

    if a.file:
        try:
            draft = read_text(pathlib.Path(a.file))
        except (OSError, TextDecodeError) as e:
            print(f"error: cannot read {a.file}: {e}", file=sys.stderr)
            return 2
    elif a.text:
        draft = a.text
    elif not sys.stdin.isatty():
        draft = sys.stdin.read()
    else:
        print("error: no draft given (use --text, --file, or pipe stdin)",
              file=sys.stderr)
        return 2

    if not draft.strip():
        print("error: empty draft", file=sys.stderr)
        return 2

    try:
        rep = check_draft(a.campaign, draft, allow=a.allow,
                          check_disposition=not a.no_disposition)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if a.as_json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        print(f"# grounding check — {a.campaign} "
              f"({rep['checked_tokens']} capitalized tokens vs "
              f"{rep['vocab_size']}-word campaign vocabulary)")
        if rep["unknown"]:
            print(f"\n## UNKNOWN PROPER NOUNS ({len(rep['unknown'])}) — "
                  f"not found anywhere in the campaign corpus")
            for u in rep["unknown"]:
                dym = ("  did you mean: " + ", ".join(u["did_you_mean"])
                       if u.get("did_you_mean") else "")
                print(f"  - {u['name']}{dym}")
                print(f"      …{u['context']}…")
        else:
            print("\nclean — every capitalized name in the draft exists in the campaign.")
        if rep["disposition_drift"]:
            print(f"\n## DISPOSITION DRIFT ({len(rep['disposition_drift'])}) — advisory")
            for d in rep["disposition_drift"]:
                print(f"  - {d['name']}: graph says {d['graph_level']}, "
                      f"draft reads {d['draft_tone']}")
                print(f"      …{d['excerpt']}…")
        if rep["unknown"] or rep["disposition_drift"]:
            print("\nResolve unknowns before narrating: either the name is canon "
                  "(add it to npcs.md + graph_seed.py --apply) or it is not "
                  "(rewrite), or pass --allow if it is legitimate new material.")

    if rep["unknown"] and a.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
