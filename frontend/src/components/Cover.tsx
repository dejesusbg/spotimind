"use client";

import { useEffect, useState } from "react";

import { api } from "@/lib/api";

type Props = { id: string; title: string; size: number; className?: string; shadow?: boolean };

/** Embedded cover art from the audio file; a neutral tile with a note icon when there is none. */
export default function Cover({ id, title, size, className = "", shadow = false }: Props) {
  const [failedId, setFailedId] = useState<string | null>(null);
  const style = { width: size, height: size };
  const radius = size >= 96 ? "rounded-md" : "rounded";
  const elevation = shadow ? "shadow-[var(--shadow-heavy)]" : "";

  if (failedId === id) {
    return (
      <div style={style} className={`${radius} ${elevation} grid shrink-0 place-items-center bg-card text-muted ${className}`}>
        <svg width={size * 0.4} height={size * 0.4} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
          <path d="M9 18V5l12-2v13M9 18a3 3 0 1 1-6 0 3 3 0 0 1 6 0Zm12-2a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" stroke="currentColor" strokeWidth="1.5" fill="none" />
        </svg>
      </div>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element -- local API image, no optimisation needed
    <img
      src={api.coverUrl(id)}
      alt={`Cover of ${title}`}
      style={style}
      loading="lazy"
      onError={() => setFailedId(id)}
      className={`${radius} ${elevation} shrink-0 object-cover ${className}`}
    />
  );
}

/** Average colour of the cover (for the header gradient), or null when unavailable. */
export function useCoverColor(id: string | null): string | null {
  const [state, setState] = useState<{ id: string; color: string | null } | null>(null);
  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    const img = new Image();
    img.crossOrigin = "anonymous"; // backend sends CORS headers, so the canvas stays readable
    img.onload = () => {
      try {
        const c = document.createElement("canvas");
        c.width = c.height = 8;
        const ctx = c.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(img, 0, 0, 8, 8);
        const d = ctx.getImageData(0, 0, 8, 8).data;
        let r = 0, g = 0, b = 0;
        for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; }
        const n = d.length / 4;
        // Darken a little so white text stays readable on top.
        const k = 0.75;
        if (!cancelled) setState({ id, color: `rgb(${Math.round((r / n) * k)}, ${Math.round((g / n) * k)}, ${Math.round((b / n) * k)})` });
      } catch {
        if (!cancelled) setState({ id, color: null });
      }
    };
    img.onerror = () => !cancelled && setState({ id, color: null });
    img.src = api.coverUrl(id);
    return () => {
      cancelled = true;
    };
  }, [id]);
  return state && state.id === id ? state.color : null;
}
