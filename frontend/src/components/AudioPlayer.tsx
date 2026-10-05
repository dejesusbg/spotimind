"use client";

import { type RefObject, useEffect, useState } from "react";

import Cover from "@/components/Cover";
import { type Song, api, formatTime } from "@/lib/api";

// Mounted with key={song.id}, so state resets when the song changes.
type Props = { song: Song; audioRef: RefObject<HTMLAudioElement | null> };

/** Full-width now-playing bar (DESIGN.md: maintained at all sizes, circular play control). */
export default function AudioPlayer({ song, audioRef }: Props) {
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(song.duration_s);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    const sync = () => {
      setPlaying(!el.paused);
      setTime(el.currentTime);
      if (Number.isFinite(el.duration)) setDuration(el.duration);
    };
    const events = ["play", "pause", "timeupdate", "seeked", "loadedmetadata", "ended"] as const;
    events.forEach((e) => el.addEventListener(e, sync));
    return () => events.forEach((e) => el.removeEventListener(e, sync));
  }, [audioRef]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (e.code !== "Space" || tag === "INPUT" || tag === "TEXTAREA" || tag === "BUTTON") return;
      e.preventDefault();
      const el = audioRef.current;
      if (el) void (el.paused ? el.play() : el.pause());
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [audioRef]);

  const toggle = () => {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) void el.play();
    else el.pause();
  };
  const seek = (t: number) => {
    const el = audioRef.current;
    if (el) el.currentTime = t;
    setTime(t);
  };

  const fill = duration ? (time / duration) * 100 : 0;
  // Bracket = the part of the song that has a brain timeline (after trims / 5-minute cap).
  const tlStart = (song.timeline_start_s / duration) * 100;
  const tlEnd = (Math.min(song.timeline_start_s + song.processed_duration_s, duration) / duration) * 100;

  return (
    <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,2fr)_minmax(0,1fr)] items-center gap-4 px-4 py-3">
      {/* The element lives here; the 3D view reads its currentTime every frame. */}
      <audio ref={audioRef} src={api.audioUrl(song.id)} preload="metadata" />

      <div className="flex min-w-0 items-center gap-3">
        <Cover id={song.id} title={song.title} size={56} />
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold">{song.title}</p>
          <p className="truncate text-xs text-muted">{song.artist ?? "Unknown artist"}</p>
        </div>
      </div>

      <div className="flex flex-col items-center gap-1.5">
        <div className="flex items-center gap-5">
          <button onClick={() => seek(Math.max(0, time - 10))} className="text-muted transition hover:text-white" aria-label="Back 10 seconds" title="Back 10 s">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" /></svg>
          </button>
          <button
            onClick={toggle}
            aria-label={playing ? "Pause" : "Play"}
            className="flex h-9 w-9 items-center justify-center rounded-full bg-white text-black transition hover:scale-105"
          >
            {playing ? (
              <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><rect x="3" y="2" width="3.5" height="12" rx="1" /><rect x="9.5" y="2" width="3.5" height="12" rx="1" /></svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5v11a.5.5 0 0 0 .76.43l9-5.5a.5.5 0 0 0 0-.86l-9-5.5A.5.5 0 0 0 4 2.5Z" /></svg>
            )}
          </button>
          <button onClick={() => seek(Math.min(duration, time + 10))} className="text-muted transition hover:text-white" aria-label="Forward 10 seconds" title="Forward 10 s">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12a9 9 0 1 1-3-6.7L21 8" /><path d="M21 3v5h-5" /></svg>
          </button>
        </div>
        <div className="flex w-full items-center gap-2">
          <span className="w-10 text-right text-[11px] tabular-nums text-muted">{formatTime(time)}</span>
          <div className="relative flex-1">
            <div
              className="pointer-events-none absolute -top-2 h-1 border-x border-t border-muted/40"
              style={{ left: `${tlStart}%`, width: `${tlEnd - tlStart}%` }}
              title="Part of the song with a brain timeline"
            />
            <input
              type="range" min={0} max={duration || 1} step={0.1} value={time}
              onChange={(e) => seek(Number(e.target.value))}
              className="seek relative w-full"
              style={{ "--fill": `${fill}%` } as React.CSSProperties}
              aria-label="Seek"
            />
          </div>
          <span className="w-10 text-[11px] tabular-nums text-muted">{formatTime(duration)}</span>
        </div>
      </div>

      <div className="hidden justify-end text-right text-[11px] leading-tight text-muted md:flex">
        <span>
          brain timeline<br />
          {formatTime(song.timeline_start_s)}–{formatTime(song.timeline_start_s + song.processed_duration_s)}
        </span>
      </div>
    </div>
  );
}
