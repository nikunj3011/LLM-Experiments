export type GenerationMode = "flash" | "thinking";

export interface ModelOption {
  id: string;
  name: string;
}

export interface SessionOption {
  id: string;
  title: string;
}

export interface Message {
  role: "user" | "assistant" | "system";
  content: string | Record<string, unknown> | Array<unknown>;
  filePreview?: string;
  fileName?: string;
  modelUsed?: string;
  modeUsed?: GenerationMode;
}

export interface GalleryAsset {
  filename: string;
  url: string;
}

export type CreativeKind = "image" | "video" | "audio";

export interface AgentConnection {
  id: string;
  name: string;
  detail: string;
  status: "connected" | "ready" | "offline";
  icon: string;
}
