"use client";

import { useMemo, useState } from "react";

import Cover from "@/components/Cover";
import { type Song, formatTime } from "@/lib/api";

type Props = { songs: Song[]; selectedId: string | null; onSelect: (id: string) => void };

export default function SongList({ songs, selectedId, onSelect }: Props) {
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return songs;
    return songs.filter((s) => `${s.title} ${s.artist ?? ""}`.toLowerCase().includes(q));
  }, [songs, query]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between px-2 pb-3">
        <h2 className="text-base font-bold">Your Library</h2>
        <span className="text-xs text-muted">{songs.length} songs</span>
      </div>
      <label className="relative mb-2 block px-2">
        <svg className="pointer-events-none absolute left-5 top-1/2 -translate-y-1/2 text-muted" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
          <circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" />
        </svg>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search in library"
          className="w-full rounded-full bg-surface-2 py-2.5 pl-10 pr-4 text-sm text-white shadow-[var(--shadow-inset)] outline-none placeholder:text-muted focus:shadow-[0_0_0_2px_#fff_inset]"
        />
      </label>
      <ul className="min-h-0 flex-1 overflow-y-auto">
        {filtered.map((s) => {
          const active = s.id === selectedId;
          return (
            <li key={s.id}>
              <button
                onClick={() => onSelect(s.id)}
                className={`flex w-full items-center gap-3 rounded-md p-2 text-left transition-colors ${active ? "bg-card" : "hover:bg-surface-2"}`}
              >
                <Cover id={s.id} title={s.title} size={44} />
                <span className="min-w-0 flex-1">
                  <span className={`block truncate text-sm ${active ? "font-bold text-green" : "font-semibold text-white"}`}>{s.title}</span>
                  <span className="block truncate text-xs text-muted">{s.artist ?? "Unknown artist"}</span>
                </span>
                <span className="shrink-0 text-xs tabular-nums text-muted">{formatTime(s.duration_s)}</span>
              </button>
            </li>
          );
        })}
        {filtered.length === 0 && <li className="px-2 py-4 text-sm text-muted">No songs match “{query}”.</li>}
      </ul>
    </div>
  );
}
