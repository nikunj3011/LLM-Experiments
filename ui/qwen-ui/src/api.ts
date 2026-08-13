import type { GalleryAsset, Message, ModelOption, SessionOption } from "./types";

const API_BASE = "http://127.0.0.1:8000/api";

const request = async <T>(path: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) throw new Error(`Request failed: ${response.status}`);
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
};

export const api = {
  models: () => request<{ models: ModelOption[] }>("/models"),
  sessions: async (): Promise<SessionOption[]> => {
    const { sessions = [] } = await request<{ sessions: string[] }>("/sessions");
    return sessions.map((id) => ({ id, title: id.replace(/^chat_/, "").replace(/_/g, " ") }));
  },
  createSession: () => request<{ session_id: string }>("/create_session"),
  loadSession: (sessionId: string) => request<{ history: Message[] }>("/load_session", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: sessionId }),
  }),
  deleteSession: (sessionId: string) => request<void>(`/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" }),
  clearVram: () => request<void>("/clear_vram", { method: "POST" }),
  gallery: () => request<{ images?: GalleryAsset[]; videos?: GalleryAsset[] }>("/gallery"),
  stream: (formData: FormData) => fetch(`${API_BASE}/stream`, { method: "POST", body: formData }),
};
