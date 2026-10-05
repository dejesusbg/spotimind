// Typed client for the FastAPI backend. Wire formats: docs/DATA_CONTRACT.md.

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000").replace(/\/$/, "");

export type NetworkInfo = { id: string; name: string; description: string };

export type Meta = {
  mock: boolean;
  n_songs: number;
  mesh_space: string;
  parcellation: string;
  n_parcels: number;
  trim_start_s: number;
  trim_end_s: number;
  max_seconds: number;
  window_mode: boolean;
  networks: NetworkInfo[];
  network_range: { min: number; max: number };
  explain_available: boolean;
  explain_model: string | null;
  created_utc: string;
};

export type Song = {
  id: string;
  title: string;
  artist: string | null;
  filename: string;
  duration_s: number;
  processed_duration_s: number;
  bin_count: number;
  timeline_start_s: number;
};

export type SongDetail = Song & { networks: Record<string, number> };

export type Neighbor = { id: string; title: string; artist: string | null; score: number };

export type Compare = {
  id: string;
  k: number;
  brain: Neighbor[];
  audio: Neighbor[];
  overlap: { shared: number; shared_ids: string[]; jaccard: number };
};

export type Explain =
  | { available: false; text: null; message: string }
  | {
      available: true;
      text: string;
      model: string;
      label: string;
      inputs: { similarity: { brain: number; audio: number } };
    };

export type NetworksDoc = {
  n_parcels: number;
  networks: (NetworkInfo & { parcels: number[] })[];
};

/** Parcel timeline [T, P] plus timing. Row i covers audio time start + i .. start + i + 1. */
export type Timeline = { T: number; P: number; start: number; rate: number; data: Float32Array };

export type Mesh = {
  positions: Float32Array;
  faces: Uint32Array;
  parcel: Int16Array;
  nVertices: number;
  nParcels: number;
};

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text();
    let detail = body;
    try {
      detail = (JSON.parse(body) as { detail?: string }).detail ?? body; // FastAPI error shape
    } catch {}
    throw new Error(res.status >= 500 ? detail : `${path}: ${res.status} ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  meta: () => getJson<Meta>("/api/meta"),
  songs: () => getJson<Song[]>("/api/songs"),
  song: (id: string) => getJson<SongDetail>(`/api/songs/${encodeURIComponent(id)}`),
  networks: () => getJson<NetworksDoc>("/api/networks"),
  compare: (id: string, k = 10) => getJson<Compare>(`/api/compare/${encodeURIComponent(id)}?k=${k}`),
  explain: (id: string, otherId: string) =>
    getJson<Explain>(`/api/songs/${encodeURIComponent(id)}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ other_id: otherId }),
    }),
  audioUrl: (id: string) => `${API_BASE}/api/audio/${encodeURIComponent(id)}`,
  coverUrl: (id: string) => `${API_BASE}/api/songs/${encodeURIComponent(id)}/cover`,

  async timeline(id: string): Promise<Timeline> {
    const res = await fetch(`${API_BASE}/api/songs/${encodeURIComponent(id)}/timeline`);
    if (!res.ok) throw new Error(`timeline ${id}: ${res.status}`);
    const buf = await res.arrayBuffer();
    const head = new DataView(buf, 0, 16);
    const T = head.getUint32(0, true);
    const P = head.getUint32(4, true);
    return {
      T,
      P,
      start: head.getFloat32(8, true),
      rate: head.getFloat32(12, true),
      data: new Float32Array(buf, 16, T * P),
    };
  },

  async mesh(): Promise<Mesh> {
    type Buf = { offset: number; count: number };
    const desc = await getJson<{ n_vertices: number; n_parcels: number; buffers: Record<string, Buf> }>("/api/mesh");
    const res = await fetch(`${API_BASE}/api/mesh/bin`);
    if (!res.ok) throw new Error(`mesh: ${res.status}`);
    const buf = await res.arrayBuffer();
    const b = desc.buffers;
    return {
      positions: new Float32Array(buf, b.positions.offset, b.positions.count),
      faces: new Uint32Array(buf, b.faces.offset, b.faces.count),
      parcel: new Int16Array(buf, b.parcel.offset, b.parcel.count),
      nVertices: desc.n_vertices,
      nParcels: desc.n_parcels,
    };
  },
};

export function songLabel(s: { title: string; artist: string | null }): string {
  return s.artist ? `${s.artist} · ${s.title}` : s.title;
}

export function formatTime(t: number): string {
  if (!Number.isFinite(t) || t < 0) return "0:00";
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
