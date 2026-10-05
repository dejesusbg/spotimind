"use client";

import { useState } from "react";

import Cover from "@/components/Cover";
import { type Compare, type Explain, type Neighbor, api } from "@/lib/api";

type Props = {
  songId: string;
  compare: Compare | null;
  explainAvailable: boolean;
  compareId: string | null;
  onSelect: (id: string) => void;
  onCompare: (id: string | null) => void;
};

type ExplainState = { otherId: string; loading: boolean; result?: Explain; error?: string };

const pill = "shrink-0 rounded-full px-3 py-1 text-[11px] font-bold uppercase tracking-[1.4px] transition";

function Column({ title, subtitle, items, shared, compareId, onSelect, onCompare, onExplain, explainAvailable, explainingId }: {
  title: string;
  subtitle: string;
  items: Neighbor[];
  shared: Set<string>;
  compareId: string | null;
  onSelect: (id: string) => void;
  onCompare: (id: string | null) => void;
  onExplain: (id: string) => void;
  explainAvailable: boolean;
  explainingId: string | null;
}) {
  return (
    <div className="min-w-0">
      <h3 className="text-lg font-bold leading-tight">{title}</h3>
      <p className="mb-3 text-xs text-muted">{subtitle}</p>
      <div className="mb-1 grid grid-cols-[1.5rem_minmax(0,1fr)_3.5rem] gap-3 border-b border-white/10 px-2 pb-2 text-[11px] uppercase tracking-[1.4px] text-muted">
        <span className="text-right">#</span><span>Title</span><span className="text-right">Match</span>
      </div>
      <ol>
        {items.map((n, i) => {
          const both = shared.has(n.id);
          const comparing = compareId === n.id;
          return (
            <li key={n.id} className={`group grid grid-cols-[1.5rem_minmax(0,1fr)_3.5rem] items-center gap-3 rounded-md px-2 py-1.5 transition-colors hover:bg-white/[0.07] ${comparing ? "bg-white/[0.07]" : ""}`}>
              <span className="text-right text-sm tabular-nums text-muted">{i + 1}</span>
              <div className="flex min-w-0 items-center gap-3">
                <Cover id={n.id} title={n.title} size={40} />
                <button onClick={() => onSelect(n.id)} className="min-w-0 flex-1 text-left" title="Play this song">
                  <span className="flex items-center gap-2">
                    <span className="truncate text-sm font-semibold hover:underline">{n.title}</span>
                    {both && (
                      <span className="shrink-0 rounded-sm bg-green/15 px-1 text-[10px] font-bold uppercase text-green" title="Also in the other list">both</span>
                    )}
                  </span>
                  <span className="block truncate text-xs text-muted">{n.artist ?? "Unknown artist"}</span>
                </button>
                <span className="flex gap-1 opacity-100 transition sm:opacity-0 sm:group-hover:opacity-100 sm:has-[:focus-visible]:opacity-100" style={comparing ? { opacity: 1 } : undefined}>
                  <button
                    onClick={() => onCompare(comparing ? null : n.id)}
                    className={`${pill} ${comparing ? "bg-white text-black" : "text-white shadow-[inset_0_0_0_1px_#7c7c7c] hover:shadow-[inset_0_0_0_1px_#fff]"}`}
                    title="Show both songs' network bars side by side"
                  >
                    {comparing ? "Comparing" : "Compare"}
                  </button>
                  <button
                    onClick={() => onExplain(n.id)}
                    disabled={!explainAvailable}
                    className={`${pill} bg-surface-2 text-white hover:bg-card disabled:cursor-not-allowed disabled:opacity-30 ${explainingId === n.id ? "bg-card" : ""}`}
                    title={explainAvailable ? "Why do these match?" : "Add GEMINI_API_KEY to backend/.env to enable"}
                  >
                    Why?
                  </button>
                </span>
              </div>
              <span className="text-right text-sm tabular-nums text-muted">{n.score.toFixed(2)}</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

export default function Neighbors({ songId, compare, explainAvailable, compareId, onSelect, onCompare }: Props) {
  const [explain, setExplain] = useState<ExplainState | null>(null);
  if (!compare || compare.id !== songId) return <p className="text-sm text-muted">Loading neighbors…</p>;
  const shared = new Set(compare.overlap.shared_ids);
  const other = (id: string) => [...compare.brain, ...compare.audio].find((n) => n.id === id);
  const visibleExplain = explain && explain.otherId && other(explain.otherId) ? explain : null;

  const runExplain = async (otherId: string) => {
    setExplain({ otherId, loading: true });
    try {
      setExplain({ otherId, loading: false, result: await api.explain(songId, otherId) });
    } catch (e) {
      setExplain({ otherId, loading: false, error: e instanceof Error ? e.message : String(e) });
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold tabular-nums">{compare.overlap.shared}</span>
          <span className="text-sm text-muted">of {compare.k} songs appear in both lists</span>
        </div>
        <span className="rounded-full bg-surface-2 px-3 py-1 text-xs text-soft">Jaccard {compare.overlap.jaccard.toFixed(2)}</span>
        <span className="text-xs text-muted">Low overlap = the brain view groups songs differently from the raw sound.</span>
      </div>

      {visibleExplain && (
        <div className="rounded-lg bg-card p-5 shadow-[var(--shadow-heavy)]">
          <div className="mb-3 flex flex-wrap items-center gap-3">
            <Cover id={songId} title="" size={36} />
            <span className="text-muted">↔</span>
            <Cover id={visibleExplain.otherId} title="" size={36} />
            <div className="min-w-0 flex-1">
              <p className="text-[11px] font-bold uppercase tracking-[1.4px] text-muted">Why might these match?</p>
              <p className="truncate text-sm font-semibold">{other(visibleExplain.otherId)?.title}</p>
            </div>
            <span className="rounded-full bg-info/15 px-3 py-1 text-[11px] font-bold uppercase tracking-[1px] text-info">
              Interpretation, not a measurement
            </span>
          </div>
          {visibleExplain.loading && (
            <div className="space-y-2" aria-label="Loading">
              {[100, 92, 96, 70].map((w) => <div key={w} className="h-3 animate-pulse rounded-full bg-white/10" style={{ width: `${w}%` }} />)}
            </div>
          )}
          {visibleExplain.error && <p className="text-sm text-negative">{visibleExplain.error}</p>}
          {visibleExplain.result && !visibleExplain.result.available && <p className="text-sm text-muted">{visibleExplain.result.message}</p>}
          {visibleExplain.result?.available && (
            <>
              <p className="text-[15px] leading-relaxed text-soft">{visibleExplain.result.text}</p>
              <p className="mt-3 text-xs text-muted">
                Written by {visibleExplain.result.model} from the computed network averages, catalog percentiles and similarity
                scores only. Network roles are general associations from neuroscience, not what you personally feel.
              </p>
            </>
          )}
          <div className="mt-4 flex gap-2">
            {visibleExplain.error && (
              <button onClick={() => runExplain(visibleExplain.otherId)} className={`${pill} bg-white text-black hover:scale-105`}>Try again</button>
            )}
            <button onClick={() => setExplain(null)} className={`${pill} text-white shadow-[inset_0_0_0_1px_#7c7c7c] hover:shadow-[inset_0_0_0_1px_#fff]`}>Close</button>
          </div>
        </div>
      )}

      <div className="grid gap-8 xl:grid-cols-2">
        <Column title="Brain-space neighbors" subtitle="Most similar predicted brain-response patterns" items={compare.brain}
          shared={shared} compareId={compareId} onSelect={onSelect} onCompare={onCompare} onExplain={runExplain}
          explainAvailable={explainAvailable} explainingId={visibleExplain?.otherId ?? null} />
        <Column title="Audio-embedding neighbors" subtitle="Most similar raw sound features (Wav2Vec-BERT baseline)" items={compare.audio}
          shared={shared} compareId={compareId} onSelect={onSelect} onCompare={onCompare} onExplain={runExplain}
          explainAvailable={explainAvailable} explainingId={visibleExplain?.otherId ?? null} />
      </div>
    </div>
  );
}
