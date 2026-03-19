const BASE = "/api";

export interface QueryResponse {
  answer: string;
  timestamp: number;
  confidence: number;
  citations: Citation[];
}

export interface Citation {
  ref: string;
  timestamp: string;
  start_seconds?: number;
  visual_summary?: string;
  diagram_type?: string;
}

export interface StatusResponse {
  status: string;
  progress: number;
  filename?: string;
  error?: string;
}

export async function uploadVideo(file: File): Promise<{ video_id: string; filename: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getStatus(videoId: string): Promise<StatusResponse> {
  const res = await fetch(`${BASE}/status/${videoId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function askQuestion(videoId: string, query: string): Promise<QueryResponse> {
  const res = await fetch(`${BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_id: videoId, query }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
