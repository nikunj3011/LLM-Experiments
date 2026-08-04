import React, { useState, useEffect, useRef, type ChangeEvent, type KeyboardEvent } from "react";
import { Plus, RefreshCw, Download, BarChart2, Trash2, Send, Paperclip, X, FileText, Cpu, Sparkles, Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = "http://127.0.0.1:8000/api";

export interface ModelOption {
  id: string;
  name: string;
  backend_type?: string;
  modality?: string;
  supports_vision?: boolean;
  description?: string;
}

export interface Message {
  role: "user" | "assistant" | "system";
  content: string | Record<string, unknown> | Array<unknown>;
  filePreview?: string;
  fileName?: string;
  modelUsed?: string;
}

export interface SessionsResponse {
  sessions: string[];
}

export interface LoadSessionResponse {
  history?: Message[];
  messages?: Message[];
}

export interface CreateSessionResponse {
  session_id: string;
}

export interface SummarizePromptResponse {
  prompt: string | null;
}

export interface ClearVramResponse {
  status: string;
  new_session_id: string;
}

export interface StreamTokenPayload {
  token?: string;
  done?: boolean;
  session_id?: string;
  metrics?: {
    elapsed_sec: number;
    tokens: number;
    tps: number;
  };
}

const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = (error) => reject(error);
  });
};

const sanitizeImageFile = (file: File, maxDimension: number = 1024): Promise<File> => {
  return new Promise((resolve) => {
    if (!file.type.startsWith("image/")) {
      const cleanName = `file_${Date.now()}.${file.name.split(".").pop() || "bin"}`;
      resolve(new File([file], cleanName, { type: file.type }));
      return;
    }

    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(url);

      let width = img.naturalWidth;
      let height = img.naturalHeight;

      if (width > maxDimension || height > maxDimension) {
        if (width > height) {
          height = Math.round((height * maxDimension) / width);
          width = maxDimension;
        } else {
          width = Math.round((width * maxDimension) / height);
          height = maxDimension;
        }
      }

      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        resolve(new File([file], `upload_${Date.now()}.jpg`, { type: file.type }));
        return;
      }

      ctx.drawImage(img, 0, 0, width, height);

      canvas.toBlob(
        (blob) => {
          if (!blob) {
            resolve(new File([file], `upload_${Date.now()}.jpg`, { type: file.type }));
            return;
          }
          const cleanFile = new File([blob], `image_${Date.now()}.jpg`, {
            type: "image/jpeg",
          });
          resolve(cleanFile);
        },
        "image/jpeg",
        0.85
      );
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(new File([file], `upload_${Date.now()}.${file.name.split(".").pop() || "jpg"}`, { type: file.type }));
    };

    img.src = url;
  });
};

export default function QwenAssistantUI(): React.JSX.Element {
  const [availableModels, setAvailableModels] = useState<ModelOption[]>([
    { id: "qwen", name: "Qwen2.5 Coder 7B" }
  ]);
  const [sessions, setSessions] = useState<string[]>([]);
  const [selectedSession, setSelectedSession] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("qwen");
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputPrompt, setInputPrompt] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreviewUrl, setFilePreviewUrl] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isGenerating]);

  useEffect(() => {
    return () => {
      if (filePreviewUrl && filePreviewUrl.startsWith("blob:")) {
        URL.revokeObjectURL(filePreviewUrl);
      }
    };
  }, [filePreviewUrl]);

  useEffect(() => {
    fetchAvailableModels();
    fetchSessions();
    handleNewChat();
  }, []);

  const fetchAvailableModels = async (): Promise<void> => {
    try {
      const res = await fetch(`${API_BASE}/models`);
      if (!res.ok) throw new Error("Failed to fetch models");
      const data = await res.json();
      if (data.models && Array.isArray(data.models) && data.models.length > 0) {
        setAvailableModels(data.models);
        if (!data.models.some((m: ModelOption) => m.id === selectedModel)) {
          setSelectedModel(data.models[0].id);
        }
      }
    } catch (e) {
      console.warn("Could not load backend model list dynamically, using fallback list:", e);
    }
  };

  const handleTextareaChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInputPrompt(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  };

  const fetchSessions = async (): Promise<void> => {
    try {
      const res = await fetch(`${API_BASE}/sessions`);
      if (!res.ok) throw new Error("Failed to fetch sessions");
      const data: SessionsResponse = await res.json();
      setSessions(data.sessions || []);
      if (data.sessions && data.sessions.length > 0 && !selectedSession) {
        setSelectedSession(data.sessions[0]);
      }
    } catch (e) {
      console.error("Failed to load session list:", e);
    }
  };

  const handleNewChat = async (): Promise<void> => {
    try {
      const res = await fetch(`${API_BASE}/create_session`);
      if (!res.ok) throw new Error("Failed to create session");
      const data: CreateSessionResponse = await res.json();
      setSelectedSession(data.session_id);
      setMessages([]);
      handleRemoveFile();
      fetchSessions();
    } catch (e) {
      console.error("Failed to create new session:", e);
    }
  };

  const handleLoadChat = async (sessionToLoad?: string): Promise<void> => {
    const targetSession = sessionToLoad || selectedSession;
    if (!targetSession || targetSession === "No Saved Chats") return;
    try {
      const res = await fetch(`${API_BASE}/load_session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: targetSession }),
      });
      if (!res.ok) throw new Error("Failed to load session");
      const data: LoadSessionResponse = await res.json();
      const historyMessages = data.messages || data.history || [];
      setMessages(historyMessages);
      setSelectedSession(targetSession);
      handleRemoveFile();
    } catch (e) {
      console.error("Failed to load chat history:", e);
    }
  };

  const handleClearVram = async (): Promise<void> => {
    try {
      const res = await fetch(`${API_BASE}/clear_vram`, { method: "POST" });
      if (!res.ok) throw new Error("Failed to clear VRAM");
      const data: ClearVramResponse = await res.json();
      setMessages([]);
      setSelectedSession(data.new_session_id);
      handleRemoveFile();
      fetchSessions();
    } catch (e) {
      console.error("Failed to clear VRAM:", e);
    }
  };

  const handleSummarizeAll = async (): Promise<void> => {
    try {
      const res = await fetch(`${API_BASE}/summarize_prompt`);
      if (!res.ok) throw new Error("Failed to get summary prompt");
      const data: SummarizePromptResponse = await res.json();
      if (!data.prompt) {
        setMessages([
          { role: "user", content: "Summarize all my past chats." },
          {
            role: "assistant",
            content: "⚠️ No previous chat history found to summarize.",
          },
        ]);
        return;
      }
      await handleNewChat();
      sendMessage(data.prompt);
    } catch (e) {
      console.error("Failed to compile summary:", e);
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>): void => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);

      if (filePreviewUrl && filePreviewUrl.startsWith("blob:")) {
        URL.revokeObjectURL(filePreviewUrl);
      }

      if (file.type.startsWith("image/")) {
        setFilePreviewUrl(URL.createObjectURL(file));
      } else {
        setFilePreviewUrl(null);
      }
    }
  };

  const handleRemoveFile = (): void => {
    setSelectedFile(null);
    if (filePreviewUrl && filePreviewUrl.startsWith("blob:")) {
      URL.revokeObjectURL(filePreviewUrl);
    }
    setFilePreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const sendMessage = async (overridePrompt?: string): Promise<void> => {
    const text = overridePrompt || inputPrompt;
    if ((!text.trim() && !selectedFile) || isGenerating) return;

    let currentFile = selectedFile;
    const currentFileName = selectedFile?.name;

    let base64ImagePreview: string | undefined = undefined;
    if (currentFile && currentFile.type.startsWith("image/")) {
      try {
        base64ImagePreview = await fileToBase64(currentFile);
      } catch (e) {
        console.error("Failed to convert file to Base64", e);
      }
    }

    const userMessage: Message = {
      role: "user",
      content: text,
      filePreview: base64ImagePreview,
      fileName: currentFileName,
    };

    const updatedMessages: Message[] = [...messages, userMessage];

    setInputPrompt("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    handleRemoveFile();
    setIsGenerating(true);

    setMessages([...updatedMessages, { role: "assistant", content: "", modelUsed: selectedModel }]);

    try {
      const formData = new FormData();
      formData.append("prompt", text);
      formData.append("model", selectedModel);
      if (selectedSession) {
        formData.append("session_id", selectedSession);
      }
      formData.append("messages", JSON.stringify(updatedMessages));

      if (currentFile) {
        currentFile = await sanitizeImageFile(currentFile);
        formData.append("file", currentFile);
      }

      const response = await fetch(`${API_BASE}/stream`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error(`Server returned HTTP ${response.status}`);
      if (!response.body) throw new Error("ReadableStream not supported on response body.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const jsonStr = line.replace(/^data:\s*/, "").trim();
              if (jsonStr) {
                const parsed: StreamTokenPayload = JSON.parse(jsonStr);

                if (parsed.session_id) {
                  setSelectedSession(parsed.session_id);
                }

                if (parsed.token) {
                  assistantText += parsed.token;
                  setMessages((prev) => {
                    const next = [...prev];
                    next[next.length - 1] = {
                      role: "assistant",
                      content: assistantText,
                      modelUsed: selectedModel,
                    };
                    return next;
                  });
                }
              }
            } catch {
              // Ignore partial frames
            }
          }
        }
      }
      fetchSessions();
    } catch (err: any) {
      console.error("Stream generation failed:", err);
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = {
          role: "assistant",
          content: `❌ Stream Connection Error: ${err.message || "Failed to generate stream."}`,
        };
        return next;
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-[#f4f4f5] text-[#18181b] flex flex-col items-center p-4 sm:p-8 font-sans">
      <div className="w-full max-w-5xl flex flex-col gap-4">
        
        <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-[#e4e4e7]">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold flex items-center gap-2 text-[#18181b]">
              <Cpu className="text-indigo-600" size={28} /> Dynamic VRAM Assistant &amp; OCR
            </h1>
            <p className="text-sm text-[#71717a]">
              Multi-model text, OCR, and vision support with dynamic VRAM swapping
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1 text-xs font-semibold text-indigo-700 bg-indigo-50 border border-indigo-200 px-3 py-1 rounded-full">
              <Sparkles size={12} /> Dynamic Engine Ready
            </span>
          </div>
        </header>

        <div className="grid grid-cols-12 gap-2 bg-white p-2.5 rounded-xl border border-[#e4e4e7] shadow-sm">
          <button
            onClick={handleNewChat}
            className="col-span-12 sm:col-span-2 bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 text-white font-medium py-2 px-3 rounded-lg flex items-center justify-center gap-1.5 transition-colors text-sm shadow-xs cursor-pointer"
          >
            <Plus size={16} /> New Chat
          </button>

          <select
            value={selectedModel}
            onChange={(e: ChangeEvent<HTMLSelectElement>) => setSelectedModel(e.target.value)}
            className="col-span-12 sm:col-span-3 bg-white border border-[#d4d4d8] text-[#18181b] text-sm rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all font-medium"
          >
            {availableModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>

          <select
            value={selectedSession}
            onChange={(e: ChangeEvent<HTMLSelectElement>) => {
              setSelectedSession(e.target.value);
              handleLoadChat(e.target.value);
            }}
            className="col-span-12 sm:col-span-3 bg-white border border-[#d4d4d8] text-[#18181b] text-sm rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all"
          >
            {sessions.length === 0 ? (
              <option value="">No Saved Chats</option>
            ) : (
              sessions.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))
            )}
          </select>

          <button
            onClick={() => handleLoadChat(selectedSession)}
            className="col-span-6 sm:col-span-1 bg-[#fafafa] hover:bg-[#f4f4f5] border border-[#e4e4e7] text-[#18181b] font-medium py-2 px-2 rounded-lg flex items-center justify-center gap-1 transition-colors text-xs cursor-pointer"
            title="Load Chat"
          >
            <Download size={14} /> Load
          </button>

          <button
            onClick={handleSummarizeAll}
            className="col-span-6 sm:col-span-2 bg-[#fafafa] hover:bg-[#f4f4f5] border border-[#e4e4e7] text-[#18181b] font-medium py-2 px-2 rounded-lg flex items-center justify-center gap-1 transition-colors text-xs cursor-pointer"
          >
            <BarChart2 size={14} /> Summarize
          </button>

          <button
            onClick={fetchSessions}
            className="col-span-12 sm:col-span-1 bg-[#fafafa] hover:bg-[#f4f4f5] border border-[#e4e4e7] text-[#18181b] font-medium py-2 px-3 rounded-lg flex items-center justify-center transition-colors text-sm cursor-pointer"
            title="Refresh List"
          >
            <RefreshCw size={16} />
          </button>
        </div>

        <div className="bg-white border border-[#e4e4e7] rounded-xl h-[550px] flex flex-col shadow-sm overflow-hidden">
          <div className="bg-[#fafafa] px-4 py-2.5 border-b border-[#e4e4e7] text-xs font-semibold text-[#71717a] uppercase tracking-wider flex justify-between items-center">
            <span>Conversation History</span>
            <span className="text-[10px] text-indigo-600 bg-indigo-50 border border-indigo-100 px-2 py-0.5 rounded-md">
              Auto VRAM Swapping Enabled
            </span>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-[#71717a] text-sm gap-2">
                <Cpu size={32} className="text-[#a1a1aa] stroke-[1.5]" />
                <p>No conversation history. Attach an image/document or type a prompt below.</p>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex flex-col w-full ${
                    msg.role === "user" ? "items-end" : "items-start"
                  }`}
                >
                  <div
                    className={`max-w-[85%] rounded-xl p-4 text-sm border shadow-xs text-left ${
                      msg.role === "user"
                        ? "border-indigo-200 bg-indigo-50/40 text-[#18181b]"
                        : "border-[#e4e4e7] bg-white text-[#18181b]"
                    }`}
                  >
                    {msg.filePreview && (
                      <div className="mb-3">
                        <img
                          src={msg.filePreview}
                          alt={msg.fileName || "User upload"}
                          className="max-h-56 rounded-lg border border-[#e4e4e7] object-contain bg-[#fafafa]"
                        />
                        {msg.fileName && (
                          <span className="text-[11px] text-[#71717a] block mt-1 font-mono">
                            {msg.fileName}
                          </span>
                        )}
                      </div>
                    )}
                    {!msg.filePreview && msg.fileName && (
                      <div className="mb-3 flex items-center gap-2 p-2.5 bg-[#f4f4f5] rounded-lg border border-[#e4e4e7] text-xs font-mono">
                        <FileText size={16} className="text-indigo-600 flex-shrink-0" />
                        <span className="truncate">{msg.fileName}</span>
                      </div>
                    )}

                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        p: ({ children }) => <p className="mb-2 last:mb-0 text-left block leading-relaxed">{children}</p>,
                        ul: ({ children }) => <ul className="list-disc pl-5 my-2 text-left space-y-1">{children}</ul>,
                        ol: ({ children }) => <ol className="list-decimal pl-5 my-2 text-left space-y-1">{children}</ol>,
                        li: ({ children }) => <li className="text-left">{children}</li>,
                        h1: ({ children }) => <h1 className="text-xl font-bold my-2 text-left">{children}</h1>,
                        h2: ({ children }) => <h2 className="text-lg font-bold my-2 text-left">{children}</h2>,
                        h3: ({ children }) => <h3 className="text-md font-bold my-1 text-left">{children}</h3>,
                        img: ({ src, alt }) => (
                          <div className="relative group my-3 inline-block max-w-full">
                            <img
                              src={src}
                              alt={alt || "AI Generated Image"}
                              className="max-h-[500px] w-auto rounded-xl border border-[#e4e4e7] object-contain shadow-sm bg-[#fafafa] transition-transform duration-200 group-hover:scale-[1.01]"
                              loading="lazy"
                            />
                            {src && (
                              <a
                                href={src}
                                target="_blank"
                                rel="noopener noreferrer"
                                download="generated_image.png"
                                className="absolute bottom-3 right-3 bg-[#18181b]/80 hover:bg-[#18181b] text-white p-2 rounded-lg backdrop-blur-md opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center gap-1.5 text-xs font-medium shadow-md"
                                title="Open / Download High-Res Image"
                              >
                                <Download size={14} />
                                <span>Save</span>
                              </a>
                            )}
                          </div>
                        ),
                        code({ className, children, ...props }) {
                          const match = /language-(\w+)/.exec(className || "");
                          const isInline = !match && !String(children).includes("\n");
                          return isInline ? (
                            <code className="bg-[#f4f4f5] border border-[#e4e4e7] px-1.5 py-0.5 rounded text-xs font-mono text-indigo-600">
                              {children}
                            </code>
                          ) : (
                            <pre className="bg-[#18181b] text-white p-3.5 rounded-lg overflow-x-auto my-2 text-xs font-mono text-left leading-relaxed">
                              <code className={className} {...props}>
                                {children}
                              </code>
                            </pre>
                          );
                        },
                        blockquote: ({ children }) => (
                          <blockquote className="border-l-4 border-indigo-500 pl-3 my-2 text-[#71717a] italic text-left">
                            {children}
                          </blockquote>
                        ),
                      }}
                    >
                      {typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content)}
                    </ReactMarkdown>
                  </div>

                  {msg.modelUsed && (
                    <span className="text-[10px] text-[#a1a1aa] mt-1 px-1 flex items-center gap-1 font-mono">
                      <Cpu size={10} /> Engine: {msg.modelUsed}
                    </span>
                  )}
                </div>
              ))
            )}
            <div ref={chatEndRef} />
          </div>
        </div>

        {selectedFile && (
          <div className="flex items-center gap-3 bg-white p-2.5 rounded-xl border border-indigo-200 shadow-sm w-fit animate-in fade-in duration-200">
            {filePreviewUrl ? (
              <img src={filePreviewUrl} alt="Preview" className="w-12 h-12 object-cover rounded-lg border border-[#e4e4e7]" />
            ) : (
              <div className="w-12 h-12 bg-indigo-50 rounded-lg border border-indigo-100 flex items-center justify-center text-indigo-600">
                <FileText size={22} />
              </div>
            )}
            <div className="flex flex-col">
              <span className="text-xs font-medium text-[#18181b] max-w-[200px] truncate">
                {selectedFile.name}
              </span>
              <span className="text-[10px] text-indigo-600 font-semibold flex items-center gap-1">
                <Sparkles size={10} /> Target: {selectedModel}
              </span>
            </div>
            <button
              onClick={handleRemoveFile}
              className="p-1.5 hover:bg-[#f4f4f5] rounded-full text-[#71717a] hover:text-red-500 transition-colors ml-2 cursor-pointer"
              title="Remove File"
            >
              <X size={16} />
            </button>
          </div>
        )}

        <div className="flex gap-2 items-end bg-white p-2 rounded-xl border border-[#e4e4e7] shadow-sm">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*,audio/*,video/*,.pdf,.txt,.py,.js,.json"
            className="hidden"
          />

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="bg-[#fafafa] hover:bg-[#f4f4f5] border border-[#d4d4d8] text-[#71717a] hover:text-indigo-600 p-3 rounded-lg flex items-center justify-center transition-colors cursor-pointer self-stretch"
            title="Attach Media or Document"
          >
            <Paperclip size={20} />
          </button>

          <textarea
            ref={textareaRef}
            value={inputPrompt}
            onChange={handleTextareaChange}
            onKeyDown={handleKeyDown}
            placeholder={
              selectedFile
                ? `Add instructions for analyzing ${selectedFile.name}...`
                : "Type a prompt or paste code... (Shift+Enter for new line)"
            }
            rows={1}
            className="flex-1 bg-transparent border-0 rounded-lg p-2.5 text-sm text-[#18181b] placeholder-[#71717a] outline-none resize-none max-h-[180px] min-h-[44px]"
          />

          <button
            onClick={() => sendMessage()}
            disabled={isGenerating || (!inputPrompt.trim() && !selectedFile)}
            className="bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 disabled:opacity-50 text-white font-medium px-5 rounded-lg flex items-center justify-center gap-2 transition-colors shadow-xs cursor-pointer disabled:cursor-not-allowed self-stretch min-w-[90px]"
          >
            {isGenerating ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <>
                <Send size={18} />
                <span className="hidden sm:inline">Send</span>
              </>
            )}
          </button>
        </div>

        <div className="flex justify-start">
          <button
            onClick={handleClearVram}
            className="bg-white hover:bg-[#fafafa] border border-[#e4e4e7] text-[#18181b] font-medium py-2 px-4 rounded-lg flex items-center gap-2 transition-colors text-sm shadow-xs cursor-pointer"
          >
            <Trash2 size={16} className="text-red-500" />
            Clear Conversation &amp; Free VRAM
          </button>
        </div>

      </div>
    </div>
  );
}