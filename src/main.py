"""
main.py — VideoSceneRAG  ·  Interactive Command-Line Assistant
==============================================================

Entry point for the full Video-RAG pipeline.

Features
--------
• Startup health checks      — validates API key, ChromaDB index, enriched metadata
• Rich terminal UI           — colour-coded output, section dividers, progress hints
• Full VideoRAG integration  — uses VideoRAG.ask() from phase4_rag.py (NLP intent,
                               hybrid retrieval, hallucination guard, confidence score)
• Citation panel             — every answer displays timestamped source scenes
• Scene preview panel        — shows which frames were retrieved for each query
• Session statistics         — tracks query count, avg confidence, intent breakdown
• Session log                — every Q&A saved to  data/logs/session_<timestamp>.jsonl
• Slash commands             — /help /clear /history /summary /stats /export /quit
• Graceful error recovery    — per-query exceptions are caught; session never crashes
• Colour-safe fallback       — degrades gracefully on terminals without ANSI support
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import signal
import textwrap
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ── Colour helpers (no external deps — pure ANSI) ─────────────────────────────

_USE_COLOUR = sys.stdout.isatty() and os.name != "nt" or (
    os.name == "nt" and os.environ.get("ANSICON") or os.environ.get("WT_SESSION")
)

def _c(code: str, text: str) -> str:
    """Wrap text in ANSI colour if terminal supports it."""
    if not _USE_COLOUR:
        return text
    codes = {
        "bold":    "\033[1m",
        "dim":     "\033[2m",
        "red":     "\033[91m",
        "green":   "\033[92m",
        "yellow":  "\033[93m",
        "blue":    "\033[94m",
        "magenta": "\033[95m",
        "cyan":    "\033[96m",
        "white":   "\033[97m",
        "reset":   "\033[0m",
    }
    return f"{codes.get(code, '')}{text}{codes['reset']}"


# ── Terminal layout ────────────────────────────────────────────────────────────

W = 72   # display width

def _hr(char: str = "─", colour: str = "dim") -> str:
    return _c(colour, char * W)

def _section(title: str, colour: str = "cyan") -> None:
    print(f"\n{_hr()}")
    print(_c(colour, f"  {title}"))
    print(_hr())

def _banner() -> None:
    lines = [
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║   📽️   V i d e o S c e n e R A G   —   AI Video Research Assistant   ║",
        "║        Ask questions about any indexed video in natural language      ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
    ]
    for line in lines:
        print(_c("cyan", line))
    print()

def _wrap(text: str, indent: int = 2) -> None:
    prefix = " " * indent
    for line in textwrap.wrap(text, width=W - indent):
        print(f"{prefix}{line}")


# ── Startup health checks ──────────────────────────────────────────────────────

def _check_env() -> bool:
    """Validate GEMINI_API_KEY is present."""
    if not os.getenv("GEMINI_API_KEY"):
        print(_c("red",
            "  ❌  GEMINI_API_KEY not found.\n"
            "      Add it to your .env file:  GEMINI_API_KEY=your_key_here"
        ))
        return False
    print(_c("green", "  ✅  GEMINI_API_KEY found"))
    return True


def _check_index() -> bool:
    """Confirm ChromaDB collections exist and are non-empty."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = [c.name for c in client.list_collections()]

        dense_ok  = "video_moments_dense"  in collections
        sparse_ok = "video_moments_sparse" in collections

        if not (dense_ok and sparse_ok):
            missing = [n for n, ok in [
                ("video_moments_dense",  dense_ok),
                ("video_moments_sparse", sparse_ok),
            ] if not ok]
            print(_c("red",
                f"  ❌  ChromaDB collections missing: {missing}\n"
                f"      Run:  python phase3_indexing.py index"
            ))
            return False

        dense_count = client.get_collection("video_moments_dense").count()
        print(_c("green", f"  ✅  ChromaDB index ready  ({dense_count} scenes indexed)"))
        return True

    except Exception as exc:
        print(_c("yellow", f"  ⚠️   ChromaDB check failed: {exc}"))
        return False


def _check_metadata() -> int:
    """Return scene count from enriched_metadata.json, or 0 if missing."""
    path = Path("data/processed/metadata/enriched_metadata.json")
    if not path.exists():
        print(_c("yellow", "  ⚠️   enriched_metadata.json not found — run Phase 1 → 2 first"))
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        n = len(data)
        has_vision  = sum(1 for s in data if s.get("visual_summary"))
        has_audio   = sum(1 for s in data if s.get("scene_transcript"))
        print(_c("green",
            f"  ✅  Metadata: {n} scenes  "
            f"(vision={has_vision}  audio={has_audio})"
        ))
        return n
    except Exception as exc:
        print(_c("yellow", f"  ⚠️   Metadata read error: {exc}"))
        return 0


def run_health_checks() -> bool:
    """Run all startup checks. Returns True if safe to proceed."""
    print(_c("bold", "\n  System checks…"))
    api_ok   = _check_env()
    idx_ok   = _check_index()
    meta_cnt = _check_metadata()

    if not api_ok:
        return False
    if not idx_ok:
        print(_c("yellow",
            "\n  ⚠️   Index missing — you can still start, but queries will fail.\n"
            "      Run the full pipeline first:\n"
            "        python phase1_sampling.py\n"
            "        python phase2_audio.py\n"
            "        python phase2_visual.py\n"
            "        python phase3_indexing.py index\n"
        ))
        ans = input("  Continue anyway? [y/N] › ").strip().lower()
        return ans == "y"

    return True


# ── Session logger ─────────────────────────────────────────────────────────────

class SessionLogger:
    """Appends every Q&A result to a JSONL file for offline review."""

    def __init__(self) -> None:
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = log_dir / f"session_{ts}.jsonl"
        self._fh  = self.path.open("a", encoding="utf-8")
        self._count = 0

    def log(self, result: dict) -> None:
        record = {
            "ts":         datetime.utcnow().isoformat(),
            "question":   result.get("question"),
            "intent":     result.get("intent"),
            "rewritten":  result.get("rewritten"),
            "answer":     result.get("answer"),
            "confidence": result.get("confidence"),
            "citations":  result.get("citations", []),
            "elapsed_s":  result.get("elapsed_s"),
            "warning":    result.get("warning"),
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        self._count += 1

    def close(self) -> None:
        self._fh.close()

    @property
    def filepath(self) -> str:
        return str(self.path)

    @property
    def count(self) -> int:
        return self._count


# ── Session statistics ─────────────────────────────────────────────────────────

class SessionStats:
    """Tracks per-session query metrics for the /stats command."""

    def __init__(self) -> None:
        self.total_queries  = 0
        self.total_elapsed  = 0.0
        self.confidences:   list[float] = []
        self.intent_counts: Counter     = Counter()
        self.warnings:      int         = 0
        self.started_at     = time.time()

    def record(self, result: dict) -> None:
        self.total_queries += 1
        self.total_elapsed += result.get("elapsed_s", 0.0)
        conf = result.get("confidence")
        if conf is not None:
            self.confidences.append(conf)
        intent = result.get("intent", "UNKNOWN")
        self.intent_counts[intent] += 1
        if result.get("warning"):
            self.warnings += 1

    def display(self) -> None:
        _section("Session Statistics", "magenta")
        elapsed_min = (time.time() - self.started_at) / 60
        avg_conf    = sum(self.confidences) / len(self.confidences) if self.confidences else 0
        avg_time    = self.total_elapsed / self.total_queries if self.total_queries else 0

        rows = [
            ("Session duration",    f"{elapsed_min:.1f} min"),
            ("Total queries",       str(self.total_queries)),
            ("Avg response time",   f"{avg_time:.2f}s"),
            ("Avg confidence",      f"{avg_conf:.0%}"),
            ("Low-confidence flags",str(self.warnings)),
        ]
        for label, value in rows:
            print(f"  {_c('dim', label + ':'): <36}  {_c('white', value)}")

        if self.intent_counts:
            print(f"\n  {_c('dim', 'Intent breakdown:')}")
            for intent, count in self.intent_counts.most_common():
                bar = "█" * count
                print(f"    {intent: <12} {_c('cyan', bar)}  {count}")
        print()


# ── Answer display ─────────────────────────────────────────────────────────────

def _confidence_bar(score: float) -> str:
    filled = int(score * 20)
    colour = "green" if score >= 0.7 else "yellow" if score >= 0.45 else "red"
    bar    = "█" * filled + "░" * (20 - filled)
    return f"{_c(colour, bar)}  {_c('bold', f'{score:.0%}')}"


def _intent_badge(intent: str) -> str:
    colours = {
        "CONCEPT":   "cyan",
        "TIMESTAMP": "blue",
        "COMPARE":   "magenta",
        "SUMMARISE": "yellow",
        "FOLLOWUP":  "green",
        "UNKNOWN":   "dim",
    }
    colour = colours.get(intent, "white")
    return _c(colour, f"[{intent}]")


def display_result(result: dict) -> None:
    """Render a full VideoRAG result to the terminal."""

    # ── Intent / rewrite header ───────────────────────────────────────────
    intent   = result.get("intent", "?")
    rewritten = result.get("rewritten", "")
    entities  = result.get("entities", [])

    print(f"\n  {_intent_badge(intent)}", end="")
    if rewritten and rewritten != result.get("question"):
        print(f"  {_c('dim', 'query rewritten →')} {_c('white', rewritten)}", end="")
    print()

    if entities:
        print(f"  {_c('dim', 'entities:')} {_c('cyan', '  ·  '.join(entities))}")

    # ── Answer ────────────────────────────────────────────────────────────
    print(f"\n{_hr('─', 'dim')}")
    answer = result.get("answer", "")

    # Highlight [Sn] citations in the answer
    def highlight_cite(m):
        return _c("yellow", m.group(0))

    if _USE_COLOUR:
        answer_display = re.sub(r"\[S\d+\]", highlight_cite, answer)
    else:
        answer_display = answer

    for line in answer_display.splitlines():
        _wrap(line, indent=2)

    # ── Citation panel ────────────────────────────────────────────────────
    citations = result.get("citations", [])
    if citations:
        print(f"\n  {_c('bold', '📌 Sources')}")
        for c in citations:
            ref      = _c("yellow", c.get("ref", ""))
            ts       = _c("cyan",   c.get("timestamp", "?"))
            dtype    = _c("dim",    c.get("diagram_type", ""))
            summary  = c.get("visual_summary", "")[:80]
            print(f"    {ref}  {ts}  {dtype}")
            if summary:
                print(f"         {_c('dim', summary)}")

    # ── Scene preview panel ───────────────────────────────────────────────
    scenes = result.get("scenes", [])
    if scenes:
        print(f"\n  {_c('bold', '🎞  Retrieved scenes')}  {_c('dim', f'(top {len(scenes)})')}")
        for i, s in enumerate(scenes, 1):
            ts     = s.get("timestamp", "?")
            dur    = s.get("duration_sec", "?")
            dtype  = s.get("diagram_type", "other")
            frame  = s.get("frame_id", "")
            score  = s.get("rrf_score")
            score_str = f"  score={score:.4f}" if score else ""
            print(
                f"    {_c('dim', f'[{i}]')} "
                f"{_c('cyan', ts)}  "
                f"{_c('dim', f'{dur}s')}  "
                f"{_c('magenta', dtype)}"
                f"{_c('dim', score_str)}  "
                f"{_c('dim', frame)}"
            )

    # ── Confidence + warnings ─────────────────────────────────────────────
    print()
    conf = result.get("confidence")
    if conf is not None:
        print(f"  {_c('dim', 'Confidence')}  [{_confidence_bar(conf)}]")

    warning = result.get("warning")
    if warning:
        print(f"\n  {_c('yellow', warning)}")

    elapsed = result.get("elapsed_s")
    if elapsed:
        print(f"  {_c('dim', f'⏱  {elapsed}s')}")

    print(f"\n{_hr()}\n")


# ── Command handlers ───────────────────────────────────────────────────────────

def _cmd_help() -> None:
    _section("Commands", "cyan")
    cmds = [
        ("/help",         "Show this help message"),
        ("/clear",        "Clear conversation memory (start fresh)"),
        ("/history",      "Show conversation history for this session"),
        ("/summary",      "Generate a comprehensive summary of the full video"),
        ("/stats",        "Show session statistics (query count, confidence, etc.)"),
        ("/export",       "Export the current session log path"),
        ("/quit  /exit",  "Exit the assistant"),
        ("",              ""),
        ("Any question",  "Ask anything about the indexed video"),
        ("",              ""),
        ("Examples:",     ""),
        ("  ›",           "What is the main topic of this video?"),
        ("  ›",           "What diagram was shown around 3 minutes?"),
        ("  ›",           "Compare the two approaches discussed in the lecture"),
        ("  ›",           "What code was on screen at 5:30?"),
    ]
    for cmd, desc in cmds:
        if cmd == "":
            print()
        elif desc == "":
            print(f"  {_c('dim', cmd)}")
        else:
            print(f"  {_c('cyan', cmd): <28}  {desc}")
    print()


def _cmd_history(rag) -> None:
    hist = rag._memory.as_history()
    if not hist:
        print(_c("dim", "\n  (no conversation history)\n"))
        return
    _section("Conversation History", "cyan")
    for i, turn in enumerate(hist, 1):
        print(f"  {_c('dim', f'[{i}]')} {_c('bold', 'Q:')} {turn['question']}")
        ans_short = turn["answer"][:200].replace("\n", " ")
        print(f"       {_c('dim', 'A:')} {ans_short}…\n")


def _cmd_export(logger: SessionLogger) -> None:
    print(f"\n  {_c('green', '📁 Session log →')} {logger.filepath}")
    print(_c("dim", f"     {logger.count} queries recorded\n"))


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_chat() -> None:
    load_dotenv()

    _banner()

    if not run_health_checks():
        print(_c("red", "\n  Startup checks failed. Exiting.\n"))
        sys.exit(1)

    # Deferred import — only after health checks pass
    try:
        from phase4_rag import VideoRAG
    except ImportError as exc:
        print(_c("red", f"\n  ❌  Cannot import VideoRAG: {exc}\n"))
        sys.exit(1)

    print(_c("dim", "\n  Initialising VideoRAG engine…"))
    try:
        rag = VideoRAG()
    except Exception as exc:
        print(_c("red", f"\n  ❌  Failed to initialise VideoRAG: {exc}\n"))
        sys.exit(1)

    logger = SessionLogger()
    stats  = SessionStats()
    last_result: dict | None = None

    # Graceful SIGINT
    def _handle_sigint(sig, frame):
        print(_c("yellow", "\n\n  Interrupted. Type /quit to exit or press Ctrl+C again.\n"))
    signal.signal(signal.SIGINT, _handle_sigint)

    print()
    print(_c("green", "  ✅  System ready."))
    print(_c("dim",   "  Type /help for commands, or ask your first question.\n"))
    print(_hr("═", "cyan"))

    # ── REPL loop ─────────────────────────────────────────────────────────
    while True:
        try:
            raw = input(_c("bold", "\n❓ › ")).strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not raw:
            continue

        lower = raw.lower()

        # ── Slash commands ────────────────────────────────────────────────
        if lower in ("/quit", "/exit", "/q"):
            break

        if lower == "/help":
            _cmd_help()
            continue

        if lower == "/clear":
            rag.clear_memory()
            print(_c("green", "\n  ✅  Conversation memory cleared.\n"))
            continue

        if lower == "/history":
            _cmd_history(rag)
            continue

        if lower == "/summary":
            print(_c("dim", "\n  Generating video summary — this may take a moment…\n"))
            try:
                result = rag.summarise_video()
                display_result(result)
                logger.log(result)
                stats.record(result)
                last_result = result
            except Exception as exc:
                print(_c("red", f"  ❌  Summary failed: {exc}\n"))
            continue

        if lower == "/stats":
            stats.display()
            continue

        if lower == "/export":
            _cmd_export(logger)
            continue

        # ── Regular question ──────────────────────────────────────────────
        print(_c("dim", "\n  🔎  Processing…"))
        t0 = time.time()

        try:
            result = rag.ask(raw)
        except KeyboardInterrupt:
            print(_c("yellow", "\n  ⚠️  Query interrupted.\n"))
            continue
        except Exception as exc:
            print(_c("red", f"\n  ❌  Query failed: {exc}"))
            if os.getenv("DEBUG"):
                traceback.print_exc()
            print(_c("dim",   "  The session is still active. Try a different question.\n"))
            continue

        display_result(result)
        logger.log(result)
        stats.record(result)
        last_result = result

    # ── Shutdown ──────────────────────────────────────────────────────────
    print()
    stats.display()
    _cmd_export(logger)
    logger.close()
    print(_c("cyan",
        "  👋  Session ended. Your log has been saved.\n"
        "      Great work on the project!\n"
    ))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_chat()
