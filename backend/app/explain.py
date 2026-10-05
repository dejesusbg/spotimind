"""Gemini 'why do these match?' text, built only from numbers we computed."""

from __future__ import annotations

import os
import time
from pathlib import Path

DEFAULT_MODEL = "gemini-3.8-flash"  # latest stable Flash per ai.google.dev/gemini-api/docs/models (2026-10); override with GEMINI_MODEL


def load_dotenv(path: Path = Path(__file__).resolve().parents[1] / ".env") -> None:
    """Read KEY=VALUE lines from backend/.env into the environment (real env vars win). Gitignored."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


def gemini_config() -> tuple[str | None, str]:
    return os.environ.get("GEMINI_API_KEY") or None, os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


# What each network is typically associated with in the neuroscience literature. Given to the model so it can
# interpret agreements/differences; phrased as associations, not as what a listener experiences.
NETWORK_ROLES = {
    "auditory": "early sound processing: loudness, pitch, timbre and how much acoustic detail there is to analyse",
    "visual": "visual processing; for sound-only input TRIBE's response here most likely reflects visual scenes it "
              "learned to expect alongside sounds in movies (an imagery-like association), not actual seeing",
    "somatomotor": "movement and body sensation; in music research motor areas are linked to beat and the urge to move",
    "dorsal_attention": "sustained, goal-directed attention to the outside world (following an unfolding signal)",
    "ventral_attention": "salience: noticing striking or unexpected events and shifting attention; also bodily arousal",
    "limbic": "orbitofrontal and temporal-pole areas tied to valuation, reward and emotional or semantic associations "
              "(fMRI signal here is noisy, so treat it cautiously)",
    "control": "executive control: working memory, tracking structure and effortful processing",
    "default": "internally directed thought: mind-wandering, memory, self-reflection and building meaning or narrative",
}


def build_prompt(a: dict, b: dict, net_names: dict[str, str], sims: dict[str, float],
                 catalog: dict[str, dict[str, float]]) -> str:
    """a/b: {title, artist, networks: {id: value}, percentiles: {id: 0-100}}. sims: {'brain', 'audio'}.
    catalog: {network id: {'std': x}} -- spread of per-song means across the whole catalog."""
    def label(s: dict) -> str:
        return f'"{s["title"]}"' + (f' by {s["artist"]}' if s.get("artist") else "")

    rows, gaps = [], {}
    for n in a["networks"]:
        va, vb = a["networks"][n], b["networks"][n]
        gaps[n] = abs(va - vb) / max(catalog[n]["std"], 1e-9)
        rows.append(
            f"- {net_names[n]} ({NETWORK_ROLES.get(n, 'n/a')}): song A {va:+.3f} ({a['percentiles'][n]:.0f}th percentile "
            f"of the catalog), song B {vb:+.3f} ({b['percentiles'][n]:.0f}th percentile); gap = {gaps[n]:.2f} catalog SDs"
        )
    closest = sorted(gaps, key=gaps.get)[:3]
    furthest = sorted(gaps, key=gaps.get, reverse=True)[:3]
    return "\n".join([
        "You interpret why two songs were matched by Spotimind, a demo that compares songs by their predicted "
        "brain responses.",
        "The numbers come from Meta's TRIBE v2 model, which predicts the fMRI response of an average person. It was "
        "trained on movies and audiobooks, not music. Each value is the mean predicted activity of one brain network "
        "over the whole song, in arbitrary model units. Percentiles place a song within this 100-song catalog. Gaps "
        "are measured in catalog standard deviations (SDs): under ~0.3 SD is nearly identical for this catalog, over "
        "~1 SD is a real difference.",
        "",
        f"Song A: {label(a)}",
        f"Song B: {label(b)}",
        f"Cosine similarity of the full brain-response patterns: {sims['brain']:.3f}",
        f"Cosine similarity of raw audio features (baseline, what an audio-only recommender sees): {sims['audio']:.3f}",
        "Per network (with what that network is typically associated with):",
        *rows,
        f"Most similar networks: {', '.join(net_names[n] for n in closest)}.",
        f"Most different networks: {', '.join(net_names[n] for n in furthest)}.",
        "",
        "Write one paragraph of 4-6 sentences for a curious non-expert:",
        "1. Open with the overall picture: how alike the two brain-response patterns are, and what the contrast "
        "with the audio-feature similarity suggests (e.g. similar predicted response despite different sound, or "
        "the other way round).",
        "2. Pick the 2-3 most informative networks (the closest and/or most different) and say what that agreement "
        "or difference would mean in terms of the network's role. Example of the depth wanted: 'Their control "
        "network levels are nearly identical, so the model expects both to place a similar load on tracking "
        "structure and holding things in mind.' Use percentiles to say whether a shared level is high or low.",
        "3. Close with what this pairing says about the match overall.",
        "Rules: use only the numbers and network roles given here. Phrase everything as the model's prediction "
        "('the model predicts', 'is typically associated with'), never as a fact about what listeners feel. Never "
        "turn a network level into an emotion, mood or enjoyment claim (not 'a stronger emotional response', but "
        "'more predicted activity in areas tied to valuation and emotional associations'). Do not "
        "invent musical facts (genre, instruments, lyrics, tempo, mood, artists' styles). Quote at most three "
        "numbers. No lists, no headings, no preamble.",
    ])


FALLBACK_MODEL = "gemini-3.5-flash"  # older stable Flash, used when the main model is overloaded
TRANSIENT_CODES = {429, 500, 503, 504}
RETRY_DELAYS_S = (1.0, 3.0)


class GeminiBusy(RuntimeError):
    """Every model/attempt hit a transient overload error."""


def generate(prompt: str, sleep=time.sleep) -> tuple[str, str]:
    """Returns (text, model used). Retries transient errors, then tries GEMINI_FALLBACK_MODEL.

    Raises RuntimeError if no key is configured, GeminiBusy if Gemini stays overloaded,
    and re-raises non-transient API errors (bad key, unknown model, ...) immediately.
    """
    key, model = gemini_config()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    from google import genai
    from google.genai import errors

    client = genai.Client(api_key=key)
    fallback = os.environ.get("GEMINI_FALLBACK_MODEL") or FALLBACK_MODEL
    models = [model] + ([fallback] if fallback and fallback != model else [])
    last: Exception | None = None
    for m in models:
        for attempt in range(len(RETRY_DELAYS_S) + 1):
            try:
                resp = client.models.generate_content(model=m, contents=prompt)
                return (resp.text or "").strip(), m
            except errors.APIError as e:
                if e.code not in TRANSIENT_CODES:
                    raise
                last = e
                if attempt < len(RETRY_DELAYS_S):
                    sleep(RETRY_DELAYS_S[attempt])
    raise GeminiBusy(str(last))
