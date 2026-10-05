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
  // The part of the song with a brain timeline (after trims / 5-minute cap) is drawn slightly lighter in the track.
  const tl0 = (song.timeline_start_s / duration) * 100;
  const tl1 = (Math.min(song.timeline_start_s + song.processed_duration_s, duration) / duration) * 100;
  const trackVars = { "--fill": `${fill}%`, "--tl0": `${tl0}%`, "--tl1": `${tl1}%` } as React.CSSProperties;

  return (
    <div className="grid h-[72px] grid-cols-[minmax(180px,1fr)_minmax(0,2fr)_minmax(120px,1fr)] items-center gap-4 px-2">
      {/* The element lives here; the 3D view reads its currentTime every frame. */}
      <audio ref={audioRef} src={api.audioUrl(song.id)} preload="metadata" />

      <div className="flex min-w-0 items-center gap-3">
        <Cover id={song.id} title={song.title} size={56} />
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold">{song.title}</p>
          <p className="truncate text-xs text-muted">{song.artist ?? "Unknown artist"}</p>
        </div>
      </div>

      <div className="mx-auto flex w-full max-w-[722px] flex-col items-center gap-1">
        <div className="flex items-center gap-6">
          <button onClick={() => seek(Math.max(0, time - 10))} className="text-muted transition hover:text-white" aria-label="Back 10 seconds" title="Back 10 s">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2"><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" /></svg>
          </button>
          <button
            onClick={toggle}
            aria-label={playing ? "Pause" : "Play"}
            className="flex h-8 w-8 items-center justify-center rounded-full bg-white text-black transition hover:scale-105"
          >
            {playing ? (
              <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><rect x="3" y="2" width="3.5" height="12" rx="1" /><rect x="9.5" y="2" width="3.5" height="12" rx="1" /></svg>
            ) : (
              <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M4.5 2.5v11a.5.5 0 0 0 .76.43l9-5.5a.5.5 0 0 0 0-.86l-9-5.5a.5.5 0 0 0-.76.43Z" /></svg>
            )}
          </button>
          <button onClick={() => seek(Math.min(duration, time + 10))} className="text-muted transition hover:text-white" aria-label="Forward 10 seconds" title="Forward 10 s">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2"><path d="M21 12a9 9 0 1 1-3-6.7L21 8" /><path d="M21 3v5h-5" /></svg>
          </button>
        </div>
        <div className="flex w-full items-center gap-2">
          <span className="w-10 shrink-0 text-right text-[11px] tabular-nums text-muted">{formatTime(time)}</span>
          <input
            type="range" min={0} max={duration || 1} step={0.1} value={time}
            onChange={(e) => seek(Number(e.target.value))}
            className="seek min-w-0 flex-1"
            style={trackVars}
            aria-label="Seek"
            title="Lighter part of the track = covered by the brain timeline"
          />
          <span className="w-10 shrink-0 text-[11px] tabular-nums text-muted">{formatTime(duration)}</span>
        </div>
      </div>

      <div className="hidden justify-end pr-2 text-right text-[11px] leading-tight text-muted md:flex">
        <span>
          brain timeline<br />
          <span className="tabular-nums">{formatTime(song.timeline_start_s)}–{formatTime(song.timeline_start_s + song.processed_duration_s)}</span>
        </span>
      </div>
    </div>
  );
}
