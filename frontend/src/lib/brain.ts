// Timeline sampling, display normalization and the colormap used by the 3D cortex.

import type { Timeline } from "./api";

/**
 * Linear interpolation of the parcel timeline at audio time t (seconds), written into `out` [P].
 * Row i is treated as the value at t = start + i + 0.5 (centre of its 1 s bin); edges are held.
 */
export function sampleTimeline(tl: Timeline, t: number, out: Float32Array): void {
  const x = (t - tl.start) * tl.rate - 0.5;
  const i0 = Math.min(Math.max(Math.floor(x), 0), tl.T - 1);
  const i1 = Math.min(i0 + 1, tl.T - 1);
  const w = Math.min(Math.max(x - i0, 0), 1);
  const a = i0 * tl.P;
  const b = i1 * tl.P;
  const d = tl.data;
  for (let p = 0; p < tl.P; p++) out[p] = d[a + p] * (1 - w) + d[b + p] * w;
}

/**
 * Per-song display range: 2nd..98th percentile of all values in the song's timeline.
 * Clipping outliers keeps the animation readable; it is a display choice only and never
 * feeds back into similarity or the network bars.
 */
export function displayRange(tl: Timeline, lo = 0.02, hi = 0.98): [number, number] {
  const n = tl.data.length;
  const stride = Math.max(1, Math.floor(n / 50000)); // subsample: percentiles don't need every value
  const vals: number[] = [];
  for (let i = 0; i < n; i += stride) vals.push(tl.data[i]);
  vals.sort((x, y) => x - y);
  const q = (f: number) => vals[Math.min(vals.length - 1, Math.max(0, Math.round(f * (vals.length - 1))))];
  const a = q(lo);
  const b = q(hi);
  return b > a ? [a, b] : [a - 1, a + 1];
}

// Inferno (matplotlib), perceptually uniform. 9 anchor colours, linearly interpolated into a 256-entry LUT.
const INFERNO_ANCHORS: [number, number, number][] = [
  [0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106], [186, 54, 85],
  [227, 89, 51], [249, 140, 10], [249, 201, 50], [252, 255, 164],
];

function buildLut(anchors: [number, number, number][], size = 256): Float32Array {
  const lut = new Float32Array(size * 3);
  for (let i = 0; i < size; i++) {
    const x = (i / (size - 1)) * (anchors.length - 1);
    const k = Math.min(Math.floor(x), anchors.length - 2);
    const f = x - k;
    for (let c = 0; c < 3; c++) {
      lut[i * 3 + c] = (anchors[k][c] * (1 - f) + anchors[k + 1][c] * f) / 255;
    }
  }
  return lut;
}

export const INFERNO = buildLut(INFERNO_ANCHORS);
export const INFERNO_CSS = `linear-gradient(90deg, ${INFERNO_ANCHORS.map(([r, g, b]) => `rgb(${r},${g},${b})`).join(", ")})`;

/** Map parcel values to RGB per parcel (into `out` [P*3]) using range [lo, hi] and the LUT. */
export function colorParcels(values: Float32Array, lo: number, hi: number, out: Float32Array, lut = INFERNO): void {
  const n = lut.length / 3;
  const scale = (n - 1) / (hi - lo);
  for (let p = 0; p < values.length; p++) {
    let k = Math.round((values[p] - lo) * scale);
    k = k < 0 ? 0 : k >= n ? n - 1 : k;
    out[p * 3] = lut[k * 3];
    out[p * 3 + 1] = lut[k * 3 + 1];
    out[p * 3 + 2] = lut[k * 3 + 2];
  }
}

/** Per-network mean of parcel values. netParcels[j] = parcel indices of network j. */
export function networkMeans(values: Float32Array, netParcels: number[][]): number[] {
  return netParcels.map((ps) => {
    let s = 0;
    for (const p of ps) s += values[p];
    return ps.length ? s / ps.length : 0;
  });
}
