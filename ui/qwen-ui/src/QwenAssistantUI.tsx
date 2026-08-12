import React, { useState, useEffect, useRef, type ChangeEvent, type KeyboardEvent } from "react";
import {
  Plus,
  Trash2,
  Send,
  Paperclip,
  X,
  FileText,
  Cpu,
  Sparkles,
  Loader2,
  MessageSquare,
  LayoutGrid,
  Moon,
  Sun,
  ChevronDown,
  Copy,
  Check,
  Zap,
  Brain,
  Square,
  Menu
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
// ============================================================================
// CONFIGURATION & INTERFACES
// ============================================================================
const API_BASE = "http://127.0.0.1:8000/api";
const API_ROOT = "http://127.0.0.1:8000";
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
  modeUsed?: "flash" | "thinking";
}

// Utility to convert file to Base64 for image previews
const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = (error) => reject(error);
  });
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================
export default function QwenAssistantUI(): React.JSX.Element {
  // --- State Management ---
  const [isDarkMode, setIsDarkMode] = useState<boolean>(false);
  const [availableModels, setAvailableModels] = useState<ModelOption[]>([{ id: "qwen", name: "Qwen2.5 Coder 7B" }]);
  const [sessions, setSessions] = useState<SessionOption[]>([]);
  const [selectedSession, setSelectedSession] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("qwen");
  const [generationMode, setGenerationMode] = useState<"flash" | "thinking">("flash");
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputPrompt, setInputPrompt] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreviewUrl, setFilePreviewUrl] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
// --- New State for Navigation & Gallery ---
  const [activeTab, setActiveTab] = useState<"chat" | "images" | "videos">("chat");
  const [galleryImages, setGalleryImages] = useState<{filename: string, url: string}[]>([]);
  const [galleryVideos, setGalleryVideos] = useState<{filename: string, url: string}[]>([]);
const [selectedImage, setSelectedImage] = useState<{ filename: string; url: string } | null>(null);
  const fetchGallery = async () => {
    try {
      const res = await fetch(`${API_BASE}/gallery`);
      if (res.ok) {
        const data = await res.json();
        setGalleryImages(data.images || []);
        setGalleryVideos(data.videos || []);
      }
    } catch (e) {
      console.error("Failed to fetch gallery:", e);
    }
  };

  // Fetch gallery when the tab changes
  useEffect(() => {
    if (activeTab === "images" || activeTab === "videos") {
      fetchGallery();
    }
  }, [activeTab]);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // --- Theme Initialization (Dark/Light Mode) ---
  useEffect(() => {
    const savedTheme = localStorage.getItem("theme");
    const systemPrefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    
    if (savedTheme === "dark" || (!savedTheme && systemPrefersDark)) {
      setIsDarkMode(true);
      document.documentElement.classList.add("dark");
    } else {
      setIsDarkMode(false);
      document.documentElement.classList.remove("dark");
    }
  }, []);

  // --- Theme Toggle Handler ---
  const toggleTheme = () => {
    setIsDarkMode((prev) => {
      const newTheme = !prev;
      if (newTheme) {
        document.documentElement.classList.add("dark");
        localStorage.setItem("theme", "dark");
      } else {
        document.documentElement.classList.remove("dark");
        localStorage.setItem("theme", "light");
      }
      return newTheme;
    });
  };

  // --- Auto-scroll to bottom of chat ---
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isGenerating]);

  // --- Cleanup Object URLs to prevent memory leaks ---
  useEffect(() => {
    return () => {
      if (filePreviewUrl && filePreviewUrl.startsWith("blob:")) {
        URL.revokeObjectURL(filePreviewUrl);
      }
    };
  }, [filePreviewUrl]);

  // --- Initial Data Fetching ---
  useEffect(() => {
    fetchAvailableModels();
    fetchSessions();
    handleNewChat();
  }, []);

  // --- API Calls ---
  const fetchAvailableModels = async () => {
    try {
      const res = await fetch(`${API_BASE}/models`);
      const data = await res.json();
      if (data.models && data.models.length > 0) {
        setAvailableModels(data.models);
        setSelectedModel(data.models[0].id);
      }
    } catch (e) {
      console.warn("Could not load backend model list dynamically.");
    }
  };

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`);

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json();

      // Backend returns:
      // {
      //   "sessions": [
      //     "chat_2026-08-07_22-41-12",
      //     "chat_2026-08-07_22-38-54"
      //   ]
      // }

      const normalizedSessions: SessionOption[] = (data.sessions || []).map(
        (sessionId: string) => ({
          id: sessionId,
          title: sessionId
            .replace(/^chat_/, "")
            .replace(/_/g, " "),
        })
      );

      setSessions(normalizedSessions);

    } catch (e) {
      console.error("Failed to load session list:", e);
      setSessions([]);
    }
  };
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);

  const handleSelectSession = async (sessionId: string) => {
    try {
      const res = await fetch(`${API_BASE}/load_session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId })
      });
      const data = await res.json();
      console.error("session history name:", sessionId);

      if (data.history) {
        setMessages(data.history);
        setSelectedSession(sessionId);
        setIsSidebarOpen(false); 
      }
    } catch (e) {
      console.error("Failed to load session history:", e);
    }
  };

  const handleNewChat = async () => {
    try {
      const res = await fetch(`${API_BASE}/create_session`);

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json();

      setSelectedSession(data.session_id);
      setMessages([]);
      handleRemoveFile();

      await fetchSessions();

    } catch (e) {
      console.error(
        "Failed to create new session:",
        e
      );
    }
  };

  const handleClearVram = async () => {
    try {
      await fetch(`${API_BASE}/clear_vram`, { method: "POST" });
      handleNewChat();
    } catch (e) {
      console.error("Failed to clear VRAM:", e);
    }
  };

  // --- Copy Text Functionality ---
  const handleCopyText = (content: string | Record<string, unknown> | Array<unknown>, index: number) => {
    const textToCopy = typeof content === "string" ? content : JSON.stringify(content);
    navigator.clipboard.writeText(textToCopy);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  // --- Input Handlers ---
  const handleTextareaChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInputPrompt(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  };
  const [deleteSessionTarget, setDeleteSessionTarget] =
    useState<SessionOption | null>(null);

  const [isDeleting, setIsDeleting] = useState(false);

  const handleDeleteSession = (
    session: SessionOption,
    e?: React.MouseEvent
  ) => {
    e?.stopPropagation();
    setDeleteSessionTarget(session);
  };

  const confirmDeleteSession = async () => {
    if (!deleteSessionTarget || isDeleting) return;

    setIsDeleting(true);

    try {
      const res = await fetch(
        `${API_BASE}/sessions/${encodeURIComponent(deleteSessionTarget.id)}`,
        {
          method: "DELETE",
        }
      );

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw new Error(
          data?.detail || `HTTP ${res.status}`
        );
      }

      const deletedId = deleteSessionTarget.id;

      setSessions((prev) =>
        prev.filter((session) => session.id !== deletedId)
      );

      setDeleteSessionTarget(null);

      if (selectedSession === deletedId) {
        await handleNewChat();
      }

    } catch (error) {
      console.error("Failed to delete session:", error);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      if (file.type.startsWith("image/")) {
        setFilePreviewUrl(URL.createObjectURL(file));
      } else {
        setFilePreviewUrl(null);
      }
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    if (filePreviewUrl) URL.revokeObjectURL(filePreviewUrl);
    setFilePreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // --- Send Message Functionality ---
  const sendMessage = async (overridePrompt?: string) => {
    const text = overridePrompt || inputPrompt;
    if ((!text.trim() && !selectedFile) || isGenerating) return;

    let base64ImagePreview: string | undefined = undefined;
    if (selectedFile && selectedFile.type.startsWith("image/")) {
      base64ImagePreview = await fileToBase64(selectedFile);
    }

    const userMessage: Message = {
      role: "user",
      content: text,
      filePreview: base64ImagePreview,
      fileName: selectedFile?.name,
    };

    const updatedMessages = [...messages, userMessage];
    setMessages([...updatedMessages, { role: "assistant", content: "", modelUsed: selectedModel, modeUsed: generationMode }]);
    
    setInputPrompt("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    handleRemoveFile();
    setIsGenerating(true);

    try {
      const formData = new FormData();
      formData.append("prompt", text);
      formData.append("model", selectedModel);
      formData.append("mode", generationMode);
      if (selectedSession) formData.append("session_id", selectedSession);
      formData.append("messages", JSON.stringify(updatedMessages));
      if (selectedFile) formData.append("file", selectedFile);

      const response = await fetch(`${API_BASE}/stream`, { method: "POST", body: formData });
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let assistantText = "";

      if (reader) {
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();

          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split("\n");

          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;

            try {
              const jsonStr = line.slice(6).trim();

              if (!jsonStr) continue;

              const parsed = JSON.parse(jsonStr);

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
                    modeUsed: generationMode,
                  };

                  return next;
                });
              }
            } catch (err) {
              console.warn("Invalid SSE data:", line);
            }
          }
        }
      }
      fetchSessions();
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = { role: "assistant", content: `❌ Error: Failed to generate response.` };
        return next;
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const hasMessages = messages.length > 0;

  // ============================================================================
  // RENDER UI
  // ============================================================================
  return (
    <div className="flex h-screen w-full overflow-hidden bg-white text-gray-900 dark:bg-gray-950 dark:text-gray-100">
        
      {/* Inline styles to completely hide scrollbars cleanly across browsers */}
     <style>{`
        /* Modern thin scrollbar */
        .modern-scrollbar {
          scrollbar-width: thin;
          scrollbar-color: rgba(156, 163, 175, 0.45) transparent;
        }
        .stream-dot {
          width: 6px;
          height: 6px;
          border-radius: 999px;
          background: rgb(99 102 241);
          animation: streamBounce 1.2s infinite ease-in-out;
          box-shadow: 0 0 8px rgba(99, 102, 241, 0.5);
        }

        .animation-delay-150 {
          animation-delay: 150ms;
        }

        .animation-delay-300 {
          animation-delay: 300ms;
        }

        @keyframes streamBounce {
          0%,
          60%,
          100% {
            transform: translateY(0);
            opacity: 0.4;
          }

          30% {
            transform: translateY(-5px);
            opacity: 1;
          }
        }
        .modern-scrollbar::-webkit-scrollbar {
          width: 7px;
          height: 7px;
        }

        .modern-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }

        .modern-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(156, 163, 175, 0.35);
          border-radius: 999px;
        }

        .modern-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(99, 102, 241, 0.65);
        }

        .dark .modern-scrollbar {
          scrollbar-color: rgba(107, 114, 128, 0.55) transparent;
        }

        .dark .modern-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(107, 114, 128, 0.45);
        }

        .dark .modern-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(129, 140, 248, 0.7);
        }

        /* Markdown styling */
        .chat-markdown {
          line-height: 1.75;
        }

        .chat-markdown p {
          margin: 0 0 0.9rem;
        }

        .chat-markdown p:last-child {
          margin-bottom: 0;
        }

        .chat-markdown ul,
        .chat-markdown ol {
          margin: 0.75rem 0;
          padding-left: 1.5rem;
        }

        .chat-markdown li {
          margin: 0.3rem 0;
        }

        .chat-markdown h1,
        .chat-markdown h2,
        .chat-markdown h3 {
          font-weight: 700;
          margin-top: 1.4rem;
          margin-bottom: 0.65rem;
        }

        .chat-markdown h1 {
          font-size: 1.5rem;
        }

        .chat-markdown h2 {
          font-size: 1.25rem;
        }

        .chat-markdown h3 {
          font-size: 1.1rem;
        }

        .chat-markdown blockquote {
          border-left: 3px solid rgb(99 102 241);
          padding-left: 1rem;
          color: rgb(107 114 128);
          margin: 1rem 0;
        }

        .dark .chat-markdown blockquote {
          color: rgb(156 163 175);
        }

        .chat-markdown table {
          width: 100%;
          border-collapse: collapse;
          margin: 1rem 0;
          overflow: hidden;
          border-radius: 0.75rem;
        }

        .chat-markdown th,
        .chat-markdown td {
          border: 1px solid rgb(229 231 235);
          padding: 0.6rem 0.8rem;
          text-align: left;
        }

        .dark .chat-markdown th,
        .dark .chat-markdown td {
          border-color: rgb(55 65 81);
        }

        .chat-markdown th {
          background: rgb(243 244 246);
          font-weight: 600;
        }

        .dark .chat-markdown th {
          background: rgb(31 41 55);
        }

        /* Prevent long code/text from expanding the page */
        .chat-markdown pre {
          max-width: 100%;
          overflow-x: auto;
        }

        .chat-markdown img {
          max-width: 100%;
          border-radius: 0.75rem;
        }
          .streaming-response {
  position: relative;
}

.streaming-response::before {
  content: "";
  position: absolute;
  left: -10px;
  top: 8px;
  bottom: 8px;
  width: 2px;
  border-radius: 999px;
  background: linear-gradient(
    to bottom,
    transparent,
    rgba(99, 102, 241, 0.7),
    rgba(168, 85, 247, 0.7),
    transparent
  );
  animation: streamingGlow 1.8s ease-in-out infinite;
}

@keyframes streamingGlow {
  0%,
  100% {
    opacity: 0.25;
  }

  50% {
    opacity: 1;
  }
}
  @keyframes streamLine {
  0% {
    transform: translateX(-150%);
  }

  100% {
    transform: translateX(450%);
  }
}
      `}</style>

      {/* --- SIDEBAR --- */}
      <aside
        className={`
          ${isSidebarOpen ? "translate-x-0" : "-translate-x-full"}
          md:translate-x-0
          fixed md:relative
          z-30
          flex flex-col
          w-[280px]
          h-full
          min-h-0
          shrink-0
          bg-gray-50/95 dark:bg-gray-950/95
          backdrop-blur-xl
          border-r border-gray-200/80 dark:border-gray-800/80
          transition-transform duration-300 ease-out
        `}
      >
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-2 font-bold text-lg">
            <div className="w-8 h-8 rounded-full bg-gray-900 text-white dark:bg-gray-100 dark:text-gray-900 flex items-center justify-center text-white">
              <Cpu size={18} />
            </div>
            <span>OB</span>
          </div>
          <button className="md:hidden p-1 rounded-md hover:bg-gray-200 dark:hover:bg-gray-800" onClick={() => setIsSidebarOpen(false)}>
            <X size={20} />
          </button>
        </div>

        {/* Sidebar New Chat Action */}
        <div className="px-3 mb-2">
          <button onClick={handleNewChat} className="w-full flex items-center justify-center gap-2 bg-gray-900 text-white dark:bg-gray-100 dark:text-gray-900 text-white font-medium py-2 px-4 rounded-xl transition-colors shadow-sm">
            <Plus size={18} /> New Chat
          </button>
        </div>

        {/* Sidebar Sessions & Chat List with scrollbar hidden via 'no-scrollbar' */}
        <nav className="flex-1 min-h-0 overflow-y-auto modern-scrollbar px-3 py-3 space-y-1">
          {/* Navigation Tabs */}
        <div className="px-3 my-4 space-y-1">
          <div className="text-xs font-semibold text-gray-400 px-3 mb-2 uppercase tracking-wider">
            Navigation
          </div>
          {/* <button 
            onClick={() => setActiveTab("chat")}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl transition-colors ${activeTab === "chat" ? "bg-gray-200 dark:bg-gray-800 text-indigo-600 dark:text-indigo-400 font-medium" : "text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800/60"}`}
          >
            <MessageSquare size={18} /> Chat
          </button> */}
          <button 
            onClick={() => setActiveTab("images")}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl transition-colors ${activeTab === "images" ? "bg-gray-200 dark:bg-gray-800 text-indigo-600 dark:text-indigo-400 font-medium" : "text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800/60"}`}
          >
            <LayoutGrid size={18} /> Generated Images
          </button>
          <button 
            onClick={() => setActiveTab("videos")}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl transition-colors ${activeTab === "videos" ? "bg-gray-200 dark:bg-gray-800 text-indigo-600 dark:text-indigo-400 font-medium" : "text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800/60"}`}
          >
            <Square size={18} /> Generated Videos
          </button>
        </div>
          <div className="text-xs font-semibold text-gray-400 px-3 my-2 uppercase tracking-wider">
            Recent Conversations
          </div>
          {sessions.length === 0 ? (
            <div className="text-sm text-gray-400 px-3 py-2 italic">No chats available</div>
          ) : (
            sessions.map((session) => (
              <div
                key={session.id}
                className={`
                  group
                  flex items-center
                  w-full
                  rounded-xl
                  transition-colors
                  ${
                    selectedSession === session.id
                      ? "bg-gray-200 dark:bg-gray-800"
                      : "hover:bg-gray-100 dark:hover:bg-gray-800/60"
                  }
                `}
              >
                <button
                  onClick={() => handleSelectSession(session.id)}
                  className={`
                    flex-1
                    min-w-0
                    flex items-center
                    gap-3
                    px-3 py-2.5
                    text-left
                    text-sm
                    font-medium
                    ${
                      selectedSession === session.id
                        ? "text-indigo-600 dark:text-indigo-400"
                        : "text-gray-600 dark:text-gray-300"
                    }
                  `}
                >
                  <MessageSquare
                    size={16}
                    className="shrink-0"
                  />

                  <span className="truncate">
                    {session.title}
                  </span>
                </button>

                <button
                  onClick={(e) => handleDeleteSession(session, e)}
                  title="Delete chat"
                  className="
                    shrink-0
                    mr-1.5
                    p-1.5
                    rounded-lg
                    text-gray-400
                    opacity-0
                    group-hover:opacity-100
                    hover:text-red-500
                    hover:bg-red-50
                    dark:hover:bg-red-950/40
                    transition-all
                  "
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))
          )}
        </nav>

        {/* Sidebar Footer Controls */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-800 space-y-2">
          <button onClick={handleClearVram} className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-900/30 dark:hover:text-red-400 transition-colors text-sm font-medium text-gray-600 dark:text-gray-300">
            <Trash2 size={18} /> Clear VRAM
          </button>
          
          <button onClick={toggleTheme} className="w-full flex items-center justify-between px-3 py-2.5 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-800 transition-colors text-sm font-medium">
            <div className="flex items-center gap-3">
              {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
              <span>{isDarkMode ? "Light Mode" : "Dark Mode"}</span>
            </div>
          </button>
        </div>
      </aside>

      {isSidebarOpen && (
        <div className="fixed inset-0 bg-black/50 z-20 md:hidden" onClick={() => setIsSidebarOpen(false)} />
      )}

      {/* --- MAIN CONTENT AREA --- */}
      <main className="flex-1 min-w-0 min-h-0 flex flex-col h-full w-full relative overflow-hidden bg-white dark:bg-gray-900">
        
        {/* Top Header Row */}
        <header
          className="
            flex
            items-center
            justify-between
            gap-1

            p-2
            sm:p-4

            absolute
            top-0
            left-0
            right-0

            z-30

            bg-white/80
            dark:bg-gray-900/80

            backdrop-blur-xl

            border-b
            border-gray-100/50
            dark:border-gray-800/50
          "
        >
          {/* ========================================================= */}
          {/* LEFT */}
          {/* ========================================================= */}

          <div className="flex items-center gap-1.5 min-w-0 flex-1">

            {/* MOBILE MENU */}
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="
                md:hidden

                w-10
                h-10
                shrink-0

                flex
                items-center
                justify-center

                rounded-xl

                text-gray-600
                dark:text-gray-300

                hover:bg-gray-100
                dark:hover:bg-gray-800

                active:scale-95
                transition-all
              "
              aria-label="Open menu"
            >
              <Menu size={21} strokeWidth={2.2} />
            </button>


            {/* ======================================================= */}
            {/* MODEL SELECTOR */}
            {/* ======================================================= */}

            <div className="relative min-w-0 flex-1">

              <button
                onClick={() => setIsModelMenuOpen(prev => !prev)}
                className="
                  group

                  w-full
                  max-w-[210px]
                  sm:max-w-[240px]
                  md:max-w-[280px]

                  h-10
                  sm:h-11

                  px-2
                  sm:px-2.5

                  flex
                  items-center
                  gap-2

                  rounded-xl

                  border
                  border-gray-200/80
                  dark:border-gray-700/80

                  bg-white/90
                  dark:bg-gray-800/90

                  shadow-sm

                  hover:bg-gray-50
                  dark:hover:bg-gray-800

                  active:scale-[0.98]

                  transition-all
                "
              >

                {/* ICON */}
                <div
                  className="
                    w-8
                    h-8
                    shrink-0

                    rounded-lg

                    bg-gradient-to-br
                    from-indigo-500
                    to-purple-600

                    flex
                    items-center
                    justify-center

                    text-white
                  "
                >
                  <Cpu size={15} />
                </div>


                {/* MODEL INFO */}
                <div className="min-w-0 flex-1 text-left">

                  <div
                    className="
                      text-[8px]
                      sm:text-[9px]

                      uppercase
                      tracking-wider
                      font-bold

                      text-gray-400
                      dark:text-gray-500

                      leading-none
                      mb-1
                    "
                  >
                    Model
                  </div>

                  <div
                    className="
                      text-[11px]
                      sm:text-xs
                      md:text-sm

                      font-semibold

                      text-gray-800
                      dark:text-gray-100

                      truncate
                    "
                  >
                    {
                      availableModels.find(
                        m => m.id === selectedModel
                      )?.name || "Select model"
                    }
                  </div>

                </div>


                <ChevronDown
                  size={15}
                  className={`
                    shrink-0
                    text-gray-400

                    transition-transform

                    ${isModelMenuOpen ? "rotate-180" : ""}
                  `}
                />

              </button>


              {/* ===================================================== */}
              {/* MODEL MENU */}
              {/* ===================================================== */}

              {isModelMenuOpen && (
              <>
                {/* ========================================================= */}
                {/* MOBILE BACKDROP */}
                {/* ========================================================= */}

                <div
                  className="
                    fixed
                    inset-0
                    z-[60]

                    bg-black/30
                    dark:bg-black/60

                    backdrop-blur-[2px]

                    md:hidden
                  "
                  onClick={() => setIsModelMenuOpen(false)}
                />

                {/* ========================================================= */}
                {/* MOBILE BOTTOM SHEET */}
                {/* ========================================================= */}

                <div
                  className="
                    fixed
                    left-0
                    right-0
                    bottom-0

                    z-[70]

                    md:hidden

                    w-full

                    max-h-[75vh]

                    rounded-t-[28px]

                    border-t
                    border-gray-200
                    dark:border-gray-700

                    bg-white
                    dark:bg-gray-900

                    shadow-[0_-20px_60px_rgba(0,0,0,0.2)]
                    dark:shadow-[0_-20px_60px_rgba(0,0,0,0.6)]

                    overflow-hidden

                    animate-in
                    slide-in-from-bottom
                    duration-300
                  "
                  onClick={(e) => e.stopPropagation()}
                >

                  {/* ======================================================= */}
                  {/* MOBILE SHEET HANDLE */}
                  {/* ======================================================= */}

                  <div className="flex justify-center pt-3 pb-2">
                    <div
                      className="
                        w-10
                        h-1

                        rounded-full

                        bg-gray-300
                        dark:bg-gray-700
                      "
                    />
                  </div>


                  {/* ======================================================= */}
                  {/* HEADER */}
                  {/* ======================================================= */}

                  <div
                    className="
                      flex
                      items-center
                      justify-between

                      px-5
                      py-3

                      border-b
                      border-gray-100
                      dark:border-gray-800
                    "
                  >

                    <div className="flex items-center gap-3 min-w-0">

                      <div
                        className="
                          w-10
                          h-10
                          shrink-0

                          rounded-xl

                          bg-gradient-to-br
                          from-indigo-500
                          to-purple-600

                          flex
                          items-center
                          justify-center

                          text-white

                          shadow-[0_4px_15px_rgba(99,102,241,0.3)]
                        "
                      >
                        <Cpu size={18} />
                      </div>

                      <div className="min-w-0">

                        <div
                          className="
                            text-base
                            font-semibold

                            text-gray-900
                            dark:text-white
                          "
                        >
                          Choose Model
                        </div>

                        <div
                          className="
                            text-xs

                            text-gray-400
                            dark:text-gray-500

                            truncate
                          "
                        >
                          Select the model for this chat
                        </div>

                      </div>

                    </div>


                    {/* CLOSE */}

                    <button
                      onClick={() => setIsModelMenuOpen(false)}
                      className="
                        w-9
                        h-9
                        shrink-0

                        flex
                        items-center
                        justify-center

                        rounded-full

                        text-gray-500
                        dark:text-gray-400

                        bg-gray-100
                        dark:bg-gray-800

                        active:scale-95

                        transition-all
                      "
                      aria-label="Close model menu"
                    >
                      <X size={17} />
                    </button>

                  </div>


                  {/* ======================================================= */}
                  {/* MODEL COUNT */}
                  {/* ======================================================= */}

                  <div
                    className="
                      px-5
                      pt-3
                      pb-2

                      text-[11px]
                      font-semibold
                      uppercase
                      tracking-wider

                      text-gray-400
                      dark:text-gray-500
                    "
                  >
                    {availableModels.length}{" "}
                    {availableModels.length === 1 ? "model" : "models"} available
                  </div>


                  {/* ======================================================= */}
                  {/* SCROLLABLE MODEL LIST */}
                  {/* ======================================================= */}

                  <div
                    className="
                      overflow-y-auto

                      max-h-[calc(75vh-145px)]

                      px-3
                      pb-5

                      modern-scrollbar

                      overscroll-contain
                    "
                  >

                    {availableModels.map((model) => {

                      const selected = model.id === selectedModel;

                      return (
                        <button
                          key={model.id}

                          onClick={() => {
                            setSelectedModel(model.id);
                            setIsModelMenuOpen(false);
                          }}

                          className={`
                            w-full

                            flex
                            items-center
                            gap-3

                            px-3
                            py-3.5

                            mb-1

                            rounded-2xl

                            text-left

                            transition-all
                            duration-200

                            active:scale-[0.98]

                            ${
                              selected
                                ? `
                                  bg-indigo-50
                                  dark:bg-indigo-950/50

                                  ring-1
                                  ring-indigo-200
                                  dark:ring-indigo-800
                                `
                                : `
                                  hover:bg-gray-50
                                  dark:hover:bg-gray-800

                                  active:bg-gray-100
                                  dark:active:bg-gray-800
                                `
                            }
                          `}
                        >

                          {/* MODEL ICON */}

                          <div
                            className={`
                              w-11
                              h-11
                              shrink-0

                              rounded-xl

                              flex
                              items-center
                              justify-center

                              transition-colors

                              ${
                                selected
                                  ? `
                                    bg-indigo-500
                                    text-white

                                    shadow-[0_4px_12px_rgba(99,102,241,0.3)]
                                  `
                                  : `
                                    bg-gray-100
                                    dark:bg-gray-800

                                    text-gray-500
                                    dark:text-gray-400
                                  `
                              }
                            `}
                          >
                            <Cpu size={18} />
                          </div>


                          {/* MODEL INFO */}

                          <div className="flex-1 min-w-0">

                            <div
                              className="
                                text-sm
                                font-semibold

                                text-gray-900
                                dark:text-gray-100

                                truncate
                              "
                            >
                              {model.name}
                            </div>

                            <div
                              className="
                                text-[11px]

                                text-gray-400
                                dark:text-gray-500

                                truncate

                                mt-0.5
                              "
                            >
                              {model.id}
                            </div>

                          </div>


                          {/* SELECTED CHECK */}

                          {selected && (
                            <div
                              className="
                                w-7
                                h-7
                                shrink-0

                                rounded-full

                                bg-indigo-500

                                flex
                                items-center
                                justify-center

                                text-white

                                shadow-sm
                              "
                            >
                              <Check size={14} strokeWidth={3} />
                            </div>
                          )}

                        </button>
                      );
                    })}

                  </div>

                </div>


                {/* ========================================================= */}
                {/* DESKTOP DROPDOWN */}
                {/* ========================================================= */}

                <div
                  className="
                    hidden
                    md:block

                    absolute

                    top-full
                    left-0

                    mt-2

                    z-[70]

                    w-[300px]

                    max-h-[50vh]

                    overflow-hidden

                    rounded-2xl

                    border
                    border-gray-200
                    dark:border-gray-700

                    bg-white/95
                    dark:bg-gray-900/95

                    backdrop-blur-xl

                    shadow-[0_20px_60px_rgba(0,0,0,0.18)]
                    dark:shadow-[0_20px_60px_rgba(0,0,0,0.5)]
                  "
                >

                  {/* DESKTOP HEADER */}

                  <div
                    className="
                      px-4
                      py-3

                      border-b
                      border-gray-100
                      dark:border-gray-800

                      bg-white/95
                      dark:bg-gray-900/95
                    "
                  >
                    <div
                      className="
                        text-[10px]
                        font-bold
                        uppercase
                        tracking-wider

                        text-gray-400
                        dark:text-gray-500
                      "
                    >
                      Available Models
                    </div>

                    <div
                      className="
                        text-xs
                        text-gray-500
                        dark:text-gray-400

                        mt-0.5
                      "
                    >
                      Choose a model to use
                    </div>
                  </div>


                  {/* DESKTOP LIST */}

                  <div
                    className="
                      overflow-y-auto

                      max-h-[calc(50vh-70px)]

                      p-1.5

                      modern-scrollbar
                    "
                  >

                    {availableModels.map((model) => {

                      const selected = model.id === selectedModel;

                      return (
                        <button
                          key={model.id}

                          onClick={() => {
                            setSelectedModel(model.id);
                            setIsModelMenuOpen(false);
                          }}

                          className={`
                            w-full

                            flex
                            items-center
                            gap-3

                            px-3
                            py-3

                            rounded-xl

                            text-left

                            transition-all

                            ${
                              selected
                                ? `
                                  bg-indigo-50
                                  dark:bg-indigo-950/40

                                  ring-1
                                  ring-indigo-200/70
                                  dark:ring-indigo-800/60
                                `
                                : `
                                  hover:bg-gray-100
                                  dark:hover:bg-gray-800
                                `
                            }
                          `}
                        >

                          <div
                            className={`
                              w-9
                              h-9
                              shrink-0

                              rounded-xl

                              flex
                              items-center
                              justify-center

                              ${
                                selected
                                  ? "bg-indigo-500 text-white"
                                  : "bg-gray-100 dark:bg-gray-800 text-gray-500"
                              }
                            `}
                          >
                            <Cpu size={16} />
                          </div>

                          <div className="flex-1 min-w-0">

                            <div
                              className="
                                text-sm
                                font-semibold

                                text-gray-800
                                dark:text-gray-100

                                truncate
                              "
                            >
                              {model.name}
                            </div>

                            <div
                              className="
                                text-[10px]

                                text-gray-400
                                dark:text-gray-500

                                truncate
                              "
                            >
                              {model.id}
                            </div>

                          </div>

                          {selected && (
                            <Check
                              size={16}
                              className="
                                shrink-0
                                text-indigo-500
                              "
                            />
                          )}

                        </button>
                      );
                    })}

                  </div>

                </div>
              </>
            )}

            </div>

          </div>


          {/* ========================================================= */}
          {/* RIGHT SIDE */}
          {/* ========================================================= */}

          <div className="flex items-center gap-1.5 shrink-0">

            {/* NEW CHAT */}
            <button
              onClick={handleNewChat}
              className="
                w-10
                h-10
                sm:w-auto
                sm:h-10

                px-0
                sm:px-3.5

                flex
                items-center
                justify-center
                gap-1.5

                rounded-xl
                sm:rounded-full

                bg-gray-900
                dark:bg-gray-100

                text-white
                dark:text-gray-900

                hover:bg-indigo-600
                dark:hover:bg-indigo-300

                active:scale-95

                transition-all

                shadow-sm
              "
              aria-label="New chat"
            >

              <Plus size={18} />

              <span className="hidden sm:inline text-sm font-medium">
                New Chat
              </span>

            </button>

          </div>

        </header>

        {/* --- SCROLLABLE VIEW (HOME OR CHAT) --- */}
        {/* --- SCROLLABLE VIEW (ROUTING) --- */}
        <div className="flex-1 min-h-0 w-full overflow-y-auto modern-scrollbar pt-20 pb-[230px] sm:pb-[260px] md:pb-[280px] flex flex-col items-center">
          {activeTab === "images" ? (
            <div className="w-full max-w-6xl mx-auto px-4 md:px-8 flex flex-col h-full">
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <LayoutGrid /> Image Gallery
              </h2>
              {galleryImages.length === 0 ? (
                <div className="text-gray-400 italic mt-10 text-center">
                  No images generated yet. Execute a ComfyUI image workflow to see them here.
                </div>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 pb-10">
                  {galleryImages.map((img, i) => (
                    <div
                      key={i}
                      onClick={() => setSelectedImage(img)}
                      className="group relative rounded-xl overflow-hidden border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900 shadow-sm hover:shadow-lg transition-all cursor-pointer"
                    >
                      <img
                        src={img.url}
                        alt={img.filename}
                        className="w-full h-auto object-cover aspect-square group-hover:scale-105 transition-transform duration-300"
                        loading="lazy"
                      />
                      <div className="p-2 text-xs text-gray-500 truncate" title={img.filename}>
                        {img.filename}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : activeTab === "videos" ? (
            <div className="w-full max-w-6xl mx-auto px-4 md:px-8 flex flex-col h-full">
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-2"><Square /> Video Gallery</h2>
              {galleryVideos.length === 0 ? (
                 <div className="text-gray-400 italic mt-10 text-center">No videos generated yet. Execute a ComfyUI video workflow to see them here.</div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-10">
                  {galleryVideos.map((vid, i) => (
                     <div key={i} className="rounded-xl overflow-hidden border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900 shadow-sm hover:shadow-md transition-shadow">
                       <video src={vid.url} controls loop className="w-full h-auto object-cover aspect-video bg-black" />
                       <div className="p-3 text-sm text-gray-500 truncate" title={vid.filename}>{vid.filename}</div>
                     </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            // ==========================================
            // EXISTING CHAT VIEW 
            // ==========================================
            <>
              {!hasMessages && (
                <div className="flex flex-col items-center justify-center h-full w-full max-w-4xl px-6 animate-in fade-in duration-500 my-auto">
                  <div className="w-16 h-16 rounded-full bg-gradient-to-br from-fuchsia-400 to-purple-600 shadow-[0_0_30px_rgba(168,85,247,0.5)] mb-6 animate-pulse" />
                  <h1 className="text-3xl md:text-5xl font-semibold mb-2 text-center">Good Afternoon, User</h1>
                  <p className="text-3xl md:text-5xl font-semibold text-gray-400 dark:text-gray-500 mb-10 text-center">What's on your mind?</p>
                </div>
              )}

              {hasMessages && (
            <div
              className="
                w-full
                max-w-4xl
                px-4
                md:px-8
                flex flex-col
                gap-8
                pt-6
                text-left
              "
            >
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`
                    flex w-full
                    ${msg.role === "user" ? "justify-end" : "justify-start"}
                  `}
                >
                  <div
                    className={`
                      w-full
                      ${msg.role === "user" ? "md:max-w-[78%]" : "md:max-w-[90%]"}
                      rounded-2xl
                      text-[15px]
                      leading-7
                      transition-all
                      duration-300
                      ${
                        msg.role === "user"
                          ? `
                            bg-gray-100
                            dark:bg-gray-800
                            px-4 py-3
                            border border-gray-200/70
                            dark:border-gray-700/70
                          `
                          : `
                            bg-transparent
                            px-1 py-1
                            ${
                              isGenerating && idx === messages.length - 1
                                ? "streaming-response"
                                : ""
                            }
                          `
                      }
                    `}
                  >
                    
                    {/* Assistant Header & Copy Action */}
                    {msg.role === "assistant" && (
                      <div className="flex items-center justify-between pb-3 mb-3">
                        <div className="flex items-center gap-2.5">
                          <div
                          className={`
                            relative
                            w-7 h-7
                            rounded-lg
                            bg-gradient-to-br
                            from-indigo-500
                            to-purple-600
                            flex items-center justify-center
                            text-white
                            shadow-sm
                            ${isGenerating && idx === messages.length - 1
                              ? "shadow-[0_0_18px_rgba(99,102,241,0.55)]"
                              : ""}
                          `}
                        >
                          <Sparkles
                            size={13}
                            className={
                              isGenerating && idx === messages.length - 1
                                ? "animate-spin"
                                : ""
                            }
                          />

                          {isGenerating && idx === messages.length - 1 && (
                            <span
                              className="
                                absolute
                                inset-0
                                rounded-lg
                                border
                                border-indigo-400/60
                                animate-ping
                              "
                            />
                          )}
                        </div>

                          <div className="flex flex-col">
                            <span className="font-semibold text-sm text-gray-800 dark:text-gray-100">
                              {msg.modeUsed === "thinking"
                                ? "Deep Thinking"
                                : msg.modelUsed || "Assistant"}
                            </span>

                            {msg.modeUsed === "thinking" && (
                              <span className="text-[11px] text-purple-500 dark:text-purple-400">
                                Reasoning mode
                              </span>
                            )}
                          </div>
                        </div>

                        
                      </div>
                    )}

                    {(msg.filePreview || msg.fileName) && (
                      <div className="mb-3">
                        <img
                          src={
                            msg.filePreview ||
                            `${API_ROOT}/temp_uploads/${msg.fileName}`
                          }
                          alt={msg.fileName || "Uploaded image"}
                          className="
                            max-w-full
                            max-h-80
                            rounded-2xl
                            border
                            border-gray-200
                            dark:border-gray-700
                            object-contain
                            shadow-sm
                          "
                        />
                      </div>
                    )}

                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeHighlight]}
                      components={{
                        code({ className, children, ...props }) {
                        const match = /language-(\w+)/.exec(className || "");
                        const code = String(children).replace(/\n$/, "");

                        if (!match) {
                          return (
                            <code
                              className="
                                px-1.5 py-0.5
                                rounded-md
                                bg-gray-200
                                dark:bg-gray-800
                                text-[13px]
                                font-mono
                                text-pink-600
                                dark:text-pink-400
                              "
                              {...props}
                            >
                              {children}
                            </code>
                          );
                        }

                        return (
                          <div
                            className="
                              my-5
                              overflow-hidden
                              rounded-xl
                              border
                              border-gray-200
                              dark:border-gray-700
                              shadow-sm
                            "
                          >
                            <div
                              className="
                                flex items-center justify-between
                                px-4 py-2.5
                                bg-gray-100
                                dark:bg-gray-800
                                border-b
                                border-gray-200
                                dark:border-gray-700
                              "
                            >
                              <div className="flex items-center gap-2">
                                <span className="w-2.5 h-2.5 rounded-full bg-red-400" />
                                <span className="w-2.5 h-2.5 rounded-full bg-yellow-400" />
                                <span className="w-2.5 h-2.5 rounded-full bg-green-400" />

                                <span className="ml-2 text-xs font-medium text-gray-500 dark:text-gray-400">
                                  {match[1]}
                                </span>
                              </div>

                              <button
                                onClick={() => navigator.clipboard.writeText(code)}
                                className="
                                  flex items-center gap-1.5
                                  text-xs
                                  text-gray-500
                                  hover:text-gray-900
                                  dark:hover:text-white
                                  transition-colors
                                "
                              >
                                <Copy size={13} />
                                Copy
                              </button>
                            </div>

                            <pre
                              className="
                                overflow-x-auto
                                bg-[#0d1117]
                                p-4
                                text-[13px]
                                leading-6
                                modern-scrollbar
                              "
                            >
                              <code className={className} {...props}>
                                {children}
                              </code>
                            </pre>
                          </div>
                        );
                      },
                      }}
                       className="
                        chat-markdown
                        max-w-none
                        w-full
                        text-left
                        text-[15px]
                        text-gray-800
                        dark:text-gray-200
                      "
                    >
                      {typeof msg.content === "string"
                        ? msg.content
                        : JSON.stringify(msg.content)}
                    </ReactMarkdown>
                    {isGenerating &&
                    idx === messages.length - 1 &&
                    msg.role === "assistant" &&
                    !msg.content && (
                      <div className="flex items-center gap-3 py-2">
                        <div className="flex items-center gap-1">
                          <span className="stream-dot" />
                          <span className="stream-dot animation-delay-150" />
                          <span className="stream-dot animation-delay-300" />
                        </div>

                        <span className="text-sm text-gray-400 dark:text-gray-500">
                          Thinking...
                        </span>
                      </div>
                    )}
                    {isGenerating &&
                        idx === messages.length - 1 &&
                        msg.role === "assistant" && (
                          <span
                            className="
                              inline-block
                              w-[2px]
                              h-[1.1em]
                              ml-1
                              align-middle
                              bg-indigo-500
                              dark:bg-indigo-400
                              rounded-full
                              animate-pulse
                              shadow-[0_0_8px_rgba(99,102,241,0.8)]
                            "
                          />
                        )}
                    <button
                          onClick={() => handleCopyText(msg.content, idx)}
                          className="
                            flex items-center gap-1.5
                            text-xs
                            text-gray-400
                            hover:text-gray-700
                            dark:hover:text-gray-200
                            px-2 py-1.5
                            rounded-lg
                            hover:bg-gray-100
                            dark:hover:bg-gray-800
                            transition-all
                          "
                        >
                          {copiedIndex === idx ? (
                            <>
                              <Check size={14} className="text-emerald-500" />
                              <span className="text-emerald-500 font-medium">Copied</span>
                            </>
                          ) : (
                            <>
                              <Copy size={14} />
                              <span>Copy</span>
                            </>
                          )}
                        </button>
                  </div>
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>
            )}
            </>
          )}
          
        </div>

        {/* --- FIXED BOTTOM INPUT AREA --- */}
        <div className="absolute bottom-0 w-full bg-gradient-to-t from-white via-white to-transparent dark:from-gray-900 dark:via-gray-900 pt-10 pb-6 px-4 flex flex-col items-center">          
          <div className="w-full max-w-4xl relative">
            
            {selectedFile && (
              <div className="absolute -top-12 left-0 flex items-center gap-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-1.5 shadow-sm">
                <FileText size={14} className="text-indigo-500" />
                <span className="text-xs font-medium truncate max-w-[150px]">{selectedFile.name}</span>
                <button onClick={handleRemoveFile} className="hover:text-red-500"><X size={14} /></button>
              </div>
            )}

            <div
              className={`
                bg-white
                dark:bg-gray-800/90
                border
                rounded-3xl
                p-2
                flex flex-col
                transition-all
                duration-500

                ${
                  isGenerating
                    ? `
                      border-indigo-400/70
                      dark:border-indigo-500/70
                      shadow-[0_8px_40px_rgba(99,102,241,0.18)]
                    `
                    : `
                      border-gray-200
                      dark:border-gray-700
                      shadow-[0_8px_35px_rgba(0,0,0,0.08)]
                      dark:shadow-[0_8px_35px_rgba(0,0,0,0.25)]
                    `
                }
              `}
            >
              {isGenerating && (
              <div
                className="
                  absolute
                  left-6
                  right-6
                  -top-[1px]
                  h-[2px]
                  overflow-hidden
                  rounded-full
                  pointer-events-none
                "
              >
                <div
                  className="
                    h-full
                    w-1/3
                    bg-gradient-to-r
                    from-transparent
                    via-indigo-500
                    to-transparent
                    animate-[streamLine_1.5s_ease-in-out_infinite]
                  "
                />
              </div>
            )}
              <textarea
                ref={textareaRef}
                value={inputPrompt}
                onChange={handleTextareaChange}
                onKeyDown={handleKeyDown}
                placeholder={
                  generationMode === "thinking"
                    ? "Ask anything · Deep Thinking enabled..."
                    : "Ask AI a question or make a request..."
                }
                rows={1}
                className="
                  w-full
                  bg-transparent
                  border-0
                  resize-none
                  outline-none
                  px-4
                  py-3
                  text-[15px]
                  leading-6
                  placeholder:text-gray-400
                  dark:placeholder:text-gray-500
                  max-h-[200px]
                "
              />
              
              <div className="flex items-center justify-between mt-2 px-1">
                <div className="flex items-center gap-1">
                  <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" />
                  <button onClick={() => fileInputRef.current?.click()} className="flex items-center gap-2 px-3 py-1.5 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors text-sm text-gray-600 dark:text-gray-300">
                    <Paperclip size={16} /> Attach
                  </button>
                </div>
                {/* FLASH VS DEEP THINKING TOGGLE */}
                <div className="flex items-center bg-gray-100 dark:bg-gray-800 p-1 rounded-xl border border-gray-200 dark:border-gray-700">
                  <button
                    onClick={() => setGenerationMode("flash")}
                    className={`
                      flex
                      items-center
                      justify-center
                      gap-1.5

                      px-2.5
                      sm:px-3

                      py-1.5

                      rounded-lg

                      text-xs
                      font-semibold

                      transition-all

                      ${
                        generationMode === "flash"
                          ? "bg-white dark:bg-gray-700 text-indigo-600 dark:text-indigo-300 shadow-sm"
                          : "text-gray-500 hover:text-gray-900 dark:hover:text-gray-200"
                      }
                    `}
                  >
                    <Zap size={13} />
                    <span>Flash</span>
                  </button>

                  <button
                    onClick={() => setGenerationMode("thinking")}
                    className={`
                      flex
                      items-center
                      justify-center
                      gap-1.5

                      px-2.5
                      sm:px-3

                      py-1.5

                      rounded-lg

                      text-xs
                      font-semibold

                      transition-all

                      ${
                        generationMode === "thinking"
                          ? "bg-white dark:bg-gray-700 text-purple-600 dark:text-purple-300 shadow-sm"
                          : "text-gray-500 hover:text-gray-900 dark:hover:text-gray-200"
                      }
                    `}
                  >
                    <Brain size={13} />
                    <span className="hidden xs:inline">
                      Deep Thinking
                    </span>
                    <span className="xs:hidden">
                      Think
                    </span>
                  </button>
                </div>
                <button
                  onClick={() => {
                    if (!isGenerating) {
                      sendMessage();
                    }
                  }}
                  disabled={!isGenerating && (!inputPrompt.trim() && !selectedFile)}
                  className={`w-9 h-9 inline-flex items-center justify-center rounded-full transition-all duration-200

                    ${
                      isGenerating
                        ? `
                          bg-red-500
                          hover:bg-red-600
                          text-white
                          shadow-[0_0_15px_rgba(239,68,68,0.35)]
                        `
                        : `
                          bg-gray-900
                          dark:bg-white
                          text-white
                          dark:text-gray-900
                          hover:bg-indigo-600
                          dark:hover:bg-indigo-400
                          dark:hover:text-white
                        `
                    }

                    disabled:opacity-40
                  `}
                >
                  {isGenerating ? (
                    <Square size={13} fill="currentColor" />
                  ) : (
                    <Send size={17}  strokeWidth={2}/>
                  )}
                </button>
              </div>
            </div>

            {!hasMessages && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-8">
                {[
                  { title: "Write a to-do list for a personal project", icon: <FileText size={16} /> },
                  { title: "Generate an email to reply to a job offer", icon: <MessageSquare size={16} /> },
                  { title: "Summarize this article in one paragraph", icon: <LayoutGrid size={16} /> },
                  { title: "How does AI work in a technical capacity", icon: <Cpu size={16} /> }
                ].map((card, idx) => (
                  <button 
                    key={idx} 
                    onClick={() => sendMessage(card.title)}
                    className="flex flex-col items-start text-left bg-gray-50 dark:bg-gray-800/40 hover:bg-gray-100 dark:hover:bg-gray-800 p-4 rounded-2xl border border-gray-100 dark:border-gray-700/50 transition-colors"
                  >
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-6">{card.title}</p>
                    <div className="text-gray-400 mt-auto">{card.icon}</div>
                  </button>
                ))}
              </div>
            )}
            
            <div className="text-center mt-3 text-xs text-gray-400 dark:text-gray-500">
              AI can make mistakes. Check important info.
            </div>
          </div>
        </div>
      </main>

      {deleteSessionTarget && (
        <div
          className="
            fixed inset-0
            z-[100]
            flex items-center justify-center
            p-4
          "
        >
          {/* Backdrop */}
          <div
            className="
              absolute inset-0
              bg-black/50
              backdrop-blur-sm
              animate-in
              fade-in
              duration-200
            "
            onClick={() =>
              !isDeleting && setDeleteSessionTarget(null)
            }
          />

          {/* Modal */}
          <div
            className="
              relative
              w-full
              max-w-md
              overflow-hidden
              rounded-3xl
              border
              border-gray-200/80
              dark:border-gray-700/80
              bg-white
              dark:bg-gray-900
              shadow-[0_25px_80px_rgba(0,0,0,0.25)]
              dark:shadow-[0_25px_80px_rgba(0,0,0,0.6)]
              animate-in
              zoom-in-95
              fade-in
              duration-200
            "
            onClick={(e) => e.stopPropagation()}
          >
            {/* Top glow */}
            <div
              className="
                absolute
                -top-20
                left-1/2
                -translate-x-1/2
                w-48
                h-48
                rounded-full
                bg-red-500/10
                dark:bg-red-500/15
                blur-3xl
                pointer-events-none
              "
            />

            {/* Content */}
            <div className="relative p-6">

              {/* Close */}
              <button
                onClick={() => setDeleteSessionTarget(null)}
                disabled={isDeleting}
                className="
                  absolute
                  top-4
                  right-4
                  p-2
                  rounded-xl
                  text-gray-400
                  hover:text-gray-700
                  dark:hover:text-gray-200
                  hover:bg-gray-100
                  dark:hover:bg-gray-800
                  transition-all
                  disabled:opacity-40
                "
              >
                <X size={18} />
              </button>

              {/* Icon */}
              <div
                className="
                  w-14
                  h-14
                  rounded-2xl
                  flex
                  items-center
                  justify-center
                  bg-red-50
                  dark:bg-red-950/40
                  text-red-500
                  dark:text-red-400
                  mb-5
                  shadow-sm
                "
              >
                <Trash2 size={24} />
              </div>

              {/* Title */}
              <h2
                className="
                  text-xl
                  font-semibold
                  text-gray-900
                  dark:text-white
                  tracking-tight
                "
              >
                Delete conversation?
              </h2>

              {/* Description */}
              <p
                className="
                  mt-2
                  text-sm
                  leading-6
                  text-gray-500
                  dark:text-gray-400
                "
              >
                This will permanently delete this conversation.
                This action cannot be undone.
              </p>

              {/* Session preview */}
              <div
                className="
                  mt-5
                  flex
                  items-center
                  gap-3
                  rounded-2xl
                  border
                  border-gray-200
                  dark:border-gray-700
                  bg-gray-50
                  dark:bg-gray-800/70
                  px-4
                  py-3
                "
              >
                <div
                  className="
                    w-9
                    h-9
                    shrink-0
                    rounded-xl
                    bg-white
                    dark:bg-gray-700
                    border
                    border-gray-200
                    dark:border-gray-600
                    flex
                    items-center
                    justify-center
                    text-indigo-500
                  "
                >
                  <MessageSquare size={16} />
                </div>

                <div className="min-w-0">
                  <p
                    className="
                      text-xs
                      font-medium
                      text-gray-400
                      dark:text-gray-500
                      mb-0.5
                    "
                  >
                    Conversation
                  </p>

                  <p
                    className="
                      text-sm
                      font-medium
                      text-gray-700
                      dark:text-gray-200
                      truncate
                    "
                  >
                    {deleteSessionTarget.title}
                  </p>
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center gap-3 mt-6">

                {/* Cancel */}
                <button
                  onClick={() => setDeleteSessionTarget(null)}
                  disabled={isDeleting}
                  className="
                    flex-1
                    h-11
                    rounded-xl
                    border
                    border-gray-200
                    dark:border-gray-700
                    bg-white
                    dark:bg-gray-800
                    text-sm
                    font-medium
                    text-gray-700
                    dark:text-gray-200
                    hover:bg-gray-50
                    dark:hover:bg-gray-700
                    transition-all
                    disabled:opacity-50
                  "
                >
                  Cancel
                </button>

                {/* Delete */}
                <button
                  onClick={confirmDeleteSession}
                  disabled={isDeleting}
                  className="
                    flex-1
                    h-11
                    rounded-xl
                    bg-red-500
                    hover:bg-red-600
                    text-white
                    text-sm
                    font-semibold
                    flex
                    items-center
                    justify-center
                    gap-2
                    shadow-sm
                    hover:shadow-[0_5px_20px_rgba(239,68,68,0.3)]
                    transition-all
                    disabled:opacity-60
                    disabled:cursor-not-allowed
                  "
                >
                  {isDeleting ? (
                    <>
                      <Loader2
                        size={16}
                        className="animate-spin"
                      />
                      Deleting...
                    </>
                  ) : (
                    <>
                      <Trash2 size={16} />
                      Delete
                    </>
                  )}
                </button>

              </div>
            </div>
          </div>
        </div>
      )}
    {/* --- IMAGE LIGHTBOX / ZOOM MODAL --- */}
      {selectedImage && (
        <div
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 animate-in fade-in duration-200"
          onClick={() => setSelectedImage(null)} // Click background to close
        >
          <div
            className="relative max-w-5xl max-h-[90vh] flex flex-col items-center"
            onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside content
          >
            {/* Close Button */}
            <button
              onClick={() => setSelectedImage(null)}
              className="absolute -top-12 right-0 text-white/80 hover:text-white bg-black/40 hover:bg-black/70 p-2 rounded-full transition-colors"
              title="Close (Esc)"
            >
              <X size={24} />
            </button>

            {/* Full Resolution Image View */}
            <img
              src={selectedImage.url}
              alt={selectedImage.filename}
              className="max-h-[80vh] w-auto max-w-full rounded-lg shadow-2xl object-contain cursor-zoom-in"
            />

            {/* Image Filename Footer */}
            <div className="mt-3 px-4 py-1.5 bg-black/60 rounded-full text-xs text-gray-200 backdrop-blur-md">
              {selectedImage.filename}
            </div>
          </div>
        </div>
      )}
    </div>
    
  );
}