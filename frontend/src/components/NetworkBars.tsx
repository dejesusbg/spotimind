"use client";

import type { NetworkInfo } from "@/lib/api";

type Series = { label: string; values: Record<string, number>; color: string };

type Props = {
  networks: NetworkInfo[];
  series: Series[]; // 1 series normally, 2 in compare mode
  range: { min: number; max: number }; // catalog-wide, so bars are comparable across songs
  live?: number[] | null; // current-time network values for series[0], same order as `networks`
};

/** Diverging bars around zero: mean predicted activation per network over the whole song. */
export default function NetworkBars({ networks, series, range, live }: Props) {
  const lim = Math.max(Math.abs(range.min), Math.abs(range.max), 1e-6) * 1.1;
  const pct = (v: number) => Math.max(-1, Math.min(1, v / lim)) * 50;

  return (
    <div className="space-y-2.5">
      {series.length > 1 && (
        <div className="flex gap-4 text-xs text-muted">
          {series.map((s) => (
            <span key={s.label} className="flex items-center gap-1.5">
              <span className="h-2 w-3 rounded-sm" style={{ background: s.color }} />
              <span className="max-w-48 truncate">{s.label}</span>
            </span>
          ))}
        </div>
      )}
      {networks.map((n, j) => (
        <div key={n.id} className="grid grid-cols-[8.5rem_1fr_3.5rem] items-center gap-3" title={n.description}>
          <span className="truncate text-xs text-soft">{n.name}</span>
          <div className="relative h-full min-h-4">
            <div className="absolute inset-y-0 left-1/2 w-px bg-white/15" />
            {series.map((s, si) => {
              const v = s.values[n.id] ?? 0;
              const w = Math.abs(pct(v));
              const h = series.length > 1 ? "h-1.5" : "h-2.5";
              const top = series.length > 1 ? (si === 0 ? "top-0.5" : "bottom-0.5") : "top-1/2 -translate-y-1/2";
              return (
                <div
                  key={s.label}
                  className={`absolute ${h} ${top} rounded-sm transition-all duration-300`}
                  style={{ background: s.color, width: `${w}%`, left: v >= 0 ? "50%" : `${50 - w}%` }}
                />
              );
            })}
            {live && (
              <div
                className="absolute top-1/2 h-3.5 w-0.5 -translate-y-1/2 rounded bg-green shadow-[0_0_6px_#1ed760]"
                style={{ left: `${50 + pct(live[j])}%` }}
                title="Right now"
              />
            )}
          </div>
          <span className="text-right font-mono text-xs text-muted">
            {(series[0].values[n.id] ?? 0) >= 0 ? "+" : ""}
            {(series[0].values[n.id] ?? 0).toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  );
}
