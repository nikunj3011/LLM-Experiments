import React, { useState, useEffect, useRef, type ChangeEvent, type KeyboardEvent } from "react";
import { Plus, RefreshCw, Download, BarChart2, Trash2, Send } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = "http://127.0.0.1:8000/api";

// TypeScript Interfaces
export interface Message {
  role: "user" | "assistant" | "system";
  content: string | Record<string, unknown> | Array<unknown>;
}

export interface SessionsResponse {
  sessions: string[];
}

export interface LoadSessionResponse {
  history: Message[];
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
}

export default function QwenAssistantUI(): React.JSX.Element {
  const [sessions, setSessions] = useState<string[]>([]);
  const [selectedSession, setSelectedSession] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputPrompt, setInputPrompt] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  const chatEndRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll chat window
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Initial Load
  useEffect(() => {
    fetchSessions();
    handleNewChat();
  }, []);

  const fetchSessions = async (): Promise<void> => {
    try {
      const res = await fetch(`${API_BASE}/sessions`);
      const data: SessionsResponse = await res.json();
      setSessions(data.sessions);
      if (data.sessions.length > 0 && !selectedSession) {
        setSelectedSession(data.sessions[0]);
      }
    } catch (e) {
      console.error("Failed to load session list:", e);
    }
  };

  const handleNewChat = async (): Promise<void> => {
    try {
      const res = await fetch(`${API_BASE}/create_session`);
      const data: CreateSessionResponse = await res.json();
      setSelectedSession(data.session_id);
      setMessages([]);
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
      const data: LoadSessionResponse = await res.json();
      setMessages(data.history || []);
      setSelectedSession(targetSession);
    } catch (e) {
      console.error("Failed to load chat history:", e);
    }
  };

  const handleClearVram = async (): Promise<void> => {
    try {
      const res = await fetch(`${API_BASE}/clear_vram`, { method: "POST" });
      const data: ClearVramResponse = await res.json();
      setMessages([]);
      setSelectedSession(data.new_session_id);
      fetchSessions();
    } catch (e) {
      console.error("Failed to clear VRAM:", e);
    }
  };

  const handleSummarizeAll = async (): Promise<void> => {
    try {
      const res = await fetch(`${API_BASE}/summarize_prompt`);
      const data: SummarizePromptResponse = await res.json();
      if (!data.prompt) {
        setMessages([
          { role: "user", content: "Summarize all my past chats." },
          {
            role: "assistant",
            content: "⚠️ No previous chat history found in `./chat_history` to summarize.",
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

  const sendMessage = async (overridePrompt?: string): Promise<void> => {
    const text = overridePrompt || inputPrompt;
    if (!text.trim() || isGenerating) return;

    const updatedMessages: Message[] = [...messages, { role: "user", content: text }];

    // Add temporary empty assistant message for streaming
    setMessages([...updatedMessages, { role: "assistant", content: "" }]);
    setInputPrompt("");
    setIsGenerating(true);

    try {
      const response = await fetch(`${API_BASE}/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: selectedSession,
          messages: updatedMessages,
        }),
      });

      if (!response.body) {
        throw new Error("ReadableStream not supported on response body.");
      }

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
                if (parsed.token) {
                  assistantText += parsed.token;
                  setMessages((prev) => {
                    const next = [...prev];
                    next[next.length - 1] = { role: "assistant", content: assistantText };
                    return next;
                  });
                }
              }
            } catch {
              // Ignore partial JSON frame errors
            }
          }
        }
      }
      fetchSessions();
    } catch (err) {
      console.error("Stream generation failed:", err);
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
        
        {/* Title Header */}
        <header className="mb-2">
          <h1 className="text-2xl sm:text-3xl font-bold flex items-center gap-2 text-[#18181b]">
            🤖 Qwen2.5-Coder-7B Local Assistant
          </h1>
          <p className="text-sm text-[#71717a] italic">
            Running locally via PyTorch &amp; BitsAndBytes 4-Bit Quantization
          </p>
        </header>

        {/* Top Controls Bar */}
        <div className="grid grid-cols-12 gap-2 bg-white p-2 rounded-lg border border-[#e4e4e7] shadow-sm">
          <button
            onClick={handleNewChat}
            className="col-span-12 sm:col-span-2 bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 px-3 rounded-md flex items-center justify-center gap-1.5 transition-colors text-sm"
          >
            <Plus size={16} /> New Chat
          </button>

          <select
            value={selectedSession}
            onChange={(e: ChangeEvent<HTMLSelectElement>) => {
              setSelectedSession(e.target.value);
              handleLoadChat(e.target.value);
            }}
            className="col-span-12 sm:col-span-4 bg-white border border-[#d4d4d8] text-[#18181b] text-sm rounded-md p-2 outline-none focus:border-indigo-500"
          >
            {sessions.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>

          <button
            onClick={() => handleLoadChat(selectedSession)}
            className="col-span-6 sm:col-span-2 bg-[#fafafa] hover:bg-[#f4f4f5] border border-[#e4e4e7] text-[#18181b] font-medium py-2 px-3 rounded-md flex items-center justify-center gap-1.5 transition-colors text-sm"
          >
            <Download size={16} /> Load Chat
          </button>

          <button
            onClick={handleSummarizeAll}
            className="col-span-6 sm:col-span-3 bg-[#fafafa] hover:bg-[#f4f4f5] border border-[#e4e4e7] text-[#18181b] font-medium py-2 px-3 rounded-md flex items-center justify-center gap-1.5 transition-colors text-sm"
          >
            <BarChart2 size={16} /> Summarize All
          </button>

          <button
            onClick={fetchSessions}
            className="col-span-12 sm:col-span-1 bg-[#fafafa] hover:bg-[#f4f4f5] border border-[#e4e4e7] text-[#18181b] font-medium py-2 px-3 rounded-md flex items-center justify-center transition-colors text-sm"
            title="Refresh List"
          >
            <RefreshCw size={16} />
          </button>
        </div>

        {/* Chat Log Box */}
        <div className="bg-white border border-[#e4e4e7] rounded-lg h-[550px] flex flex-col shadow-sm overflow-hidden">
          <div className="bg-[#fafafa] px-4 py-2 border-b border-[#e4e4e7] text-xs font-semibold text-[#71717a] uppercase tracking-wider">
            Conversation History
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="h-full flex items-center justify-center text-[#71717a] text-sm">
                No conversation history. Type a prompt below to start.
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex flex-col w-full ${
                    msg.role === "user" ? "items-end" : "items-start"
                  }`}
                >
                  {/* 2. Message Bubble (Ensure text-left and items-start) */}
                  <div
                    className={`max-w-[85%] rounded-lg p-3 text-sm border border-[#e4e4e7] bg-white text-[#18181b] shadow-sm text-left ${
                      msg.role === "user" ? "border-indigo-200 bg-indigo-50/30" : ""
                    }`}
                  >
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        // Ensure paragraphs align top-left and reset line margins
                        p: ({ children }) => <p className="mb-2 last:mb-0 text-left block">{children}</p>,
                        ul: ({ children }) => <ul className="list-disc pl-5 my-2 text-left">{children}</ul>,
                        ol: ({ children }) => <ol className="list-decimal pl-5 my-2 text-left">{children}</ol>,
                        li: ({ children }) => <li className="mb-1 text-left">{children}</li>,
                        h1: ({ children }) => <h1 className="text-xl font-bold my-2 text-left">{children}</h1>,
                        h2: ({ children }) => <h2 className="text-lg font-bold my-2 text-left">{children}</h2>,
                        h3: ({ children }) => <h3 className="text-md font-bold my-1 text-left">{children}</h3>,
                        code({ inline, className, children, ...props }: any) {
                          return inline ? (
                            <code className="bg-[#f4f4f5] border border-[#e4e4e7] px-1 py-0.5 rounded text-xs font-mono text-indigo-600">
                              {children}
                            </code>
                          ) : (
                            <pre className="bg-[#18181b] text-white p-3 rounded-md overflow-x-auto my-2 text-xs font-mono text-left">
                              <code {...props}>{children}</code>
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
                </div>
              ))
            )}
            <div ref={chatEndRef} />
          </div>
        </div>

        {/* Prompt Input Area */}
        <div className="flex gap-2">
          <textarea
            value={inputPrompt}
            onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setInputPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a prompt or paste large code blocks/context here... (Shift+Enter for new line)"
            rows={3}
            className="flex-1 bg-white border border-[#d4d4d8] rounded-lg p-3 text-sm text-[#18181b] placeholder-[#71717a] outline-none focus:border-indigo-500 shadow-sm resize-y"
          />
          <button
            onClick={() => sendMessage()}
            disabled={isGenerating || !inputPrompt.trim()}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-medium px-6 rounded-lg flex items-center justify-center gap-2 transition-colors shadow-sm cursor-pointer disabled:cursor-not-allowed"
          >
            <Send size={18} />
            <span>Send</span>
          </button>
        </div>

        {/* VRAM Clear Footer */}
        <div className="flex justify-start">
          <button
            onClick={handleClearVram}
            className="bg-[#fafafa] hover:bg-[#f4f4f5] border border-[#e4e4e7] text-[#18181b] font-medium py-2 px-4 rounded-md flex items-center gap-2 transition-colors text-sm shadow-sm cursor-pointer"
          >
            <Trash2 size={16} className="text-red-500" />
            Clear Conversation &amp; Free VRAM
          </button>
        </div>

      </div>
    </div>
  );
}