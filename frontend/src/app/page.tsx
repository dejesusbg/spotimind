"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import AudioPlayer from "@/components/AudioPlayer";
import type { ViewPreset } from "@/components/BrainViewer";
import Cover, { useCoverColor } from "@/components/Cover";
import NetworkBars from "@/components/NetworkBars";
import Neighbors from "@/components/Neighbors";
import SongList from "@/components/SongList";
import {
  type Compare, type Mesh, type Meta, type NetworksDoc, type Song, type SongDetail, type Timeline,
  API_BASE, api, formatTime,
} from "@/lib/api";
import { INFERNO_CSS, displayRange, networkMeans, sampleTimeline } from "@/lib/brain";

const BrainViewer = dynamic(() => import("@/components/BrainViewer"), {
  ssr: false,
  loading: () => <div className="grid h-full place-items-center text-sm text-muted">Loading 3D view…</div>,
});

const MAIN_COLOR = "#ffffff";
const COMPARE_COLOR = "#539df5"; // DESIGN.md "announcement blue", used only for the comparison series

function Logo() {
  return (
    <div className="flex items-center gap-2">
      <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden>
        <circle cx="16" cy="16" r="16" fill="#1ed760" />
        <path d="M8 18c2.5-5 5-5 7.5 0s5 5 8.5-2" fill="none" stroke="#000" strokeWidth="2.6" strokeLinecap="round" />
        <circle cx="11" cy="11" r="1.6" fill="#000" /><circle cx="21" cy="11" r="1.6" fill="#000" />
      </svg>
      <span className="text-xl font-bold tracking-tight">Spotimind</span>
    </div>
  );
}

export default function Home() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [songs, setSongs] = useState<Song[]>([]);
  const [networks, setNetworks] = useState<NetworksDoc | null>(null);
  const [mesh, setMesh] = useState<Mesh | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SongDetail | null>(null);
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [compare, setCompare] = useState<Compare | null>(null);
  const [compareId, setCompareId] = useState<string | null>(null);
  const [compareDetail, setCompareDetail] = useState<SongDetail | null>(null);
  const [view, setView] = useState<ViewPreset>("left");
  const [live, setLive] = useState<number[] | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const mainRef = useRef<HTMLElement | null>(null);
  const coverColor = useCoverColor(selectedId);

  useEffect(() => {
    Promise.all([api.meta(), api.songs(), api.networks()])
      .then(([m, s, n]) => {
        setMeta(m);
        setSongs(s);
        setNetworks(n);
        if (s.length) setSelectedId(s[0].id);
      })
      .catch((e) => setError(`Could not reach the backend at ${API_BASE}: ${e.message}`));
    api.mesh().then(setMesh).catch((e) => setError(`Mesh failed to load: ${e.message}`));
  }, []);

  const select = useCallback((id: string) => {
    setSelectedId(id);
    setCompareId(null);
    setTimeline(null);
    setLive(null);
    mainRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    Promise.all([api.song(selectedId), api.timeline(selectedId), api.compare(selectedId, 10)])
      .then(([d, tl, c]) => {
        if (cancelled) return;
        setDetail(d);
        setTimeline(tl);
        setCompare(c);
      })
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  useEffect(() => {
    if (!compareId) return;
    api.song(compareId).then(setCompareDetail).catch((e) => setError(e.message));
  }, [compareId]);

  const range = useMemo<[number, number]>(() => (timeline ? displayRange(timeline) : [0, 1]), [timeline]);
  const netParcels = useMemo(() => networks?.networks.map((n) => n.parcels) ?? [], [networks]);

  // Live per-network values at the current playback time (~8 Hz; the 3D view runs at full frame rate).
  useEffect(() => {
    if (!timeline || !netParcels.length) return;
    const buf = new Float32Array(timeline.P);
    const id = window.setInterval(() => {
      const t = audioRef.current?.currentTime ?? 0;
      sampleTimeline(timeline, t, buf);
      setLive(networkMeans(buf, netParcels));
    }, 125);
    return () => window.clearInterval(id);
  }, [timeline, netParcels]);

  const song = songs.find((s) => s.id === selectedId) ?? null;
  const comparing = compareDetail && compareDetail.id === compareId ? compareDetail : null;

  if (error && !meta) {
    return (
      <main className="grid min-h-screen place-items-center p-8">
        <div className="max-w-lg rounded-lg bg-surface p-6 text-sm shadow-[var(--shadow-heavy)]">
          <p className="mb-2 font-bold text-negative">Backend unavailable</p>
          <p className="text-soft">{error}</p>
          <p className="mt-3 text-muted">Start it with <code className="text-white">cd backend && uv run uvicorn app.main:app --port 8000</code>.</p>
        </div>
      </main>
    );
  }

  return (
    <div className="flex h-screen flex-col gap-2 bg-black p-2">
      <div className="flex min-h-0 flex-1 gap-2">
        {/* Sidebar: logo + library */}
        <aside className="hidden w-[320px] shrink-0 flex-col gap-2 lg:flex">
          <div className="rounded-lg bg-bg px-5 py-4">
            <Logo />
            <p className="mt-2 text-xs leading-snug text-muted">Songs matched by predicted brain response, not by tags.</p>
          </div>
          <div className="min-h-0 flex-1 rounded-lg bg-bg p-2">
            <SongList songs={songs} selectedId={selectedId} onSelect={select} />
          </div>
        </aside>

        {/* Main view */}
        <main ref={mainRef} className="relative min-w-0 flex-1 overflow-y-auto rounded-lg bg-bg">
          {/* Header tinted by the cover art: "album art provides the colour" */}
          <header
            className="px-6 pb-6 pt-5 transition-[background] duration-700"
            style={{ background: `linear-gradient(180deg, ${coverColor ?? "#2a2a2a"} 0%, #121212 100%)` }}
          >
            <div className="mb-6 flex items-center justify-between gap-3">
              <div className="lg:hidden"><Logo /></div>
              <div className="ml-auto flex items-center gap-2">
                {meta?.mock && (
                  <span className="rounded-full bg-warning px-3 py-1 text-[11px] font-bold uppercase tracking-[1.4px] text-black">Mock data</span>
                )}
                {meta && (
                  <span className="rounded-full bg-black/40 px-3 py-1 text-[11px] text-soft">
                    {meta.n_songs} songs · TRIBE v2 · {meta.parcellation.split("_")[0]}
                  </span>
                )}
              </div>
            </div>
            {song && (
              <div className="flex items-end gap-6">
                <Cover id={song.id} title={song.title} size={176} shadow className="hidden sm:block" />
                <div className="min-w-0 pb-1">
                  <p className="text-xs font-semibold text-soft">Now exploring</p>
                  <h1 className="mt-1 truncate text-4xl font-bold leading-tight sm:text-5xl">{song.title}</h1>
                  <p className="mt-3 text-sm text-soft">
                    <span className="font-bold text-white">{song.artist ?? "Unknown artist"}</span>
                    <span className="mx-1.5">•</span>{formatTime(song.duration_s)}
                    <span className="mx-1.5">•</span>brain timeline {formatTime(song.timeline_start_s)}–{formatTime(song.timeline_start_s + song.processed_duration_s)}
                  </p>
                </div>
              </div>
            )}
          </header>

          <div className="space-y-8 px-6 pb-10">
            <section className="grid gap-4 2xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
              <div className="relative h-[440px] overflow-hidden rounded-lg bg-surface shadow-[var(--shadow-card)]">
                {mesh ? (
                  <BrainViewer mesh={mesh} timeline={timeline} range={range} audioRef={audioRef} view={view} />
                ) : (
                  <div className="grid h-full place-items-center text-sm text-muted">Loading cortex mesh…</div>
                )}
                <div className="pointer-events-none absolute left-4 top-4">
                  <p className="text-sm font-bold">Predicted brain response</p>
                  <p className="text-xs text-muted">Average listener · drag to orbit</p>
                </div>
                <div className="absolute right-4 top-4 flex gap-1.5">
                  {(["left", "right", "top", "back"] as ViewPreset[]).map((v) => (
                    <button key={v} onClick={() => setView(v)}
                      className={`rounded-full px-3 py-1 text-xs font-bold capitalize transition ${view === v ? "bg-white text-black" : "bg-surface-2/90 text-white hover:bg-card"}`}>
                      {v}
                    </button>
                  ))}
                </div>
                <div className="pointer-events-none absolute bottom-4 left-4 right-4 flex items-center gap-2 text-[11px] text-muted">
                  <span className="tabular-nums">{range[0].toFixed(2)}</span>
                  <div className="h-1 flex-1 rounded-full" style={{ background: INFERNO_CSS }} />
                  <span className="tabular-nums">{range[1].toFixed(2)}</span>
                  <span className="hidden md:inline">· colours clipped to this song’s 2nd–98th percentile</span>
                </div>
              </div>

              <div className="rounded-lg bg-surface p-5 shadow-[var(--shadow-card)]">
                <div className="mb-4 flex items-baseline justify-between gap-2">
                  <h2 className="text-lg font-bold">Brain networks</h2>
                  <span className="flex items-center gap-1.5 text-[11px] text-muted">
                    bar = whole-song average <span className="ml-1 inline-block h-3 w-0.5 rounded bg-green" /> now
                  </span>
                </div>
                {meta && detail && (
                  <NetworkBars
                    networks={meta.networks}
                    range={meta.network_range}
                    live={live}
                    series={[
                      { label: detail.title, values: detail.networks, color: MAIN_COLOR },
                      ...(comparing ? [{ label: comparing.title, values: comparing.networks, color: COMPARE_COLOR }] : []),
                    ]}
                  />
                )}
                <p className="mt-5 text-xs leading-relaxed text-muted">
                  Each bar is a plain average of the model’s prediction over that network’s regions (Schaefer-400 regions grouped
                  into Yeo’s 7 networks, plus auditory cortex on its own). One shared scale across all songs. Hover a network
                  for what it does; press <span className="font-bold text-white">Compare</span> on a neighbor to overlay it.
                </p>
              </div>
            </section>

            <section>
              {selectedId && (
                <Neighbors songId={selectedId} compare={compare} explainAvailable={!!meta?.explain_available}
                  compareId={compareId} onSelect={select} onCompare={setCompareId} />
              )}
            </section>

            <footer className="border-t border-white/10 pt-6 text-xs leading-relaxed text-muted">
              <p className="mb-1 font-bold text-soft">Honest limits</p>
              <p className="max-w-4xl">
                These are predictions for an <em>average</em> brain from Meta’s TRIBE v2, which was trained on fMRI recorded while
                people watched movies and listened to audiobooks, not music. Readings are about one per second and smeared by a few
                seconds, so trust the shape over seconds, not beats. In our checks, music raises auditory cortex but also visual
                areas, likely a habit learned from movies. Brain-space similarity is a proxy for “feels alike”, not a measure of
                anyone’s taste. TRIBE v2 is licensed CC BY-NC 4.0; Spotimind is a non-commercial portfolio demo.
              </p>
              {meta && (
                <p className="mt-2 text-muted/70">
                  {meta.n_songs} songs · {meta.parcellation} on {meta.mesh_space} · first {meta.trim_start_s}s / last {meta.trim_end_s}s trimmed
                  {meta.window_mode ? " · 90 s window" : ` · capped at ${meta.max_seconds / 60} min`}
                </p>
              )}
            </footer>
          </div>
        </main>
      </div>

      {/* Now-playing bar */}
      <div className="shrink-0 rounded-lg bg-black">
        {song ? <AudioPlayer key={song.id} song={song} audioRef={audioRef} /> : <div className="h-[88px]" />}
      </div>
    </div>
  );
}
