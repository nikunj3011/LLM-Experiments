import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Sparkles } from "lucide-react";
import { api } from "./api";
import { ChatComposer } from "./components/ChatComposer";
import { ChatMessage } from "./components/ChatMessage";
import { CreativeStudio } from "./components/CreativeStudio";
import { Header } from "./components/Header";
import { SettingsPage } from "./components/SettingsPage";
import { Sidebar } from "./components/Sidebar";
import { useTheme } from "./hooks/useTheme";
import type { GenerationMode, Message, ModelOption, SessionOption } from "./types";

const fileToDataUrl = (file: File) => new Promise<string>((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.onerror = reject; reader.readAsDataURL(file); });
const validPath = (path: string) => ["/", "/image", "/video", "/audio", "/agents", "/settings"].includes(path) ? path : "/";

export default function App() {
  const { isDark, toggleTheme } = useTheme();
  const [pathname, setPathname] = useState(() => validPath(window.location.pathname));
  const [models, setModels] = useState<ModelOption[]>([{ id: "qwen", name: "Qwen2.5 Coder 7B" }]);
  const [selectedModel, setSelectedModel] = useState("qwen");
  const [sessions, setSessions] = useState<SessionOption[]>([]);
  const [selectedSession, setSelectedSession] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<GenerationMode>("flash");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const refreshSessions = useCallback(async () => { try { setSessions(await api.sessions()); } catch { setSessions([]); } }, []);
  const createChat = useCallback(async () => { try { const { session_id } = await api.createSession(); setSelectedSession(session_id); setMessages([]); setPrompt(""); setFile(null); await refreshSessions(); } catch { /* The workspace remains usable until the runtime returns. */ } }, [refreshSessions]);
  useEffect(() => { void refreshSessions(); void createChat(); void api.models().then(({ models: available }) => { if (available.length) { setModels(available); setSelectedModel(available[0].id); } }).catch(() => undefined); }, [createChat, refreshSessions]);
  useEffect(() => { const listener = () => setPathname(validPath(window.location.pathname)); window.addEventListener("popstate", listener); return () => window.removeEventListener("popstate", listener); }, []);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, isGenerating]);

  const navigate = useCallback((path: string) => { if (path !== window.location.pathname) window.history.pushState({}, "", path); setPathname(path); setIsSidebarOpen(false); }, []);
  const selectSession = useCallback(async (sessionId: string) => { try { const { history } = await api.loadSession(sessionId); setMessages(history ?? []); setSelectedSession(sessionId); navigate("/"); } catch { /* Keep the visible conversation on connection failure. */ } }, [navigate]);
  const deleteSession = useCallback(async (session: SessionOption) => { if (!window.confirm(`Delete “${session.title}”? This cannot be undone.`)) return; try { await api.deleteSession(session.id); await refreshSessions(); if (selectedSession === session.id) void createChat(); } catch { /* Session remains visible until backend confirms deletion. */ } }, [createChat, refreshSessions, selectedSession]);
  const clearVram = useCallback(async () => { try { await api.clearVram(); await createChat(); } catch { /* Runtime handles resource status. */ } }, [createChat]);
  const copy = useCallback((content: Message["content"], index: number) => { void navigator.clipboard.writeText(typeof content === "string" ? content : JSON.stringify(content)); setCopiedIndex(index); window.setTimeout(() => setCopiedIndex(null), 1800); }, []);
  const sendMessage = useCallback(async () => {
    if (isGenerating || (!prompt.trim() && !file)) return;
    const preview = file?.type.startsWith("image/") ? await fileToDataUrl(file) : undefined;
    const userMessage: Message = { role: "user", content: prompt, fileName: file?.name, filePreview: preview };
    const history = [...messages, userMessage];
    setMessages([...history, { role: "assistant", content: "", modelUsed: selectedModel, modeUsed: mode }]); setPrompt(""); setFile(null); setIsGenerating(true);
    try { const formData = new FormData(); formData.append("prompt", userMessage.content as string); formData.append("model", selectedModel); formData.append("mode", mode); formData.append("messages", JSON.stringify(history)); if (selectedSession) formData.append("session_id", selectedSession); if (file) formData.append("file", file); const response = await api.stream(formData); if (!response.ok || !response.body) throw new Error("Streaming unavailable"); const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = ""; let answer = ""; while (true) { const { done, value } = await reader.read(); if (done) break; buffer += decoder.decode(value, { stream: true }); const lines = buffer.split("\n"); buffer = lines.pop() ?? ""; for (const line of lines) { if (!line.startsWith("data: ")) continue; try { const event = JSON.parse(line.slice(6)); if (event.session_id) setSelectedSession(event.session_id); if (event.token) { answer += event.token; setMessages((current) => [...current.slice(0, -1), { role: "assistant", content: answer, modelUsed: selectedModel, modeUsed: mode }]); } } catch { /* Ignore incomplete stream events. */ } } } void refreshSessions(); } catch { setMessages((current) => [...current.slice(0, -1), { role: "assistant", content: "Sorry, I couldn't generate a response. Please check that the local backend is running." }]); } finally { setIsGenerating(false); }
  }, [file, isGenerating, messages, mode, prompt, refreshSessions, selectedModel, selectedSession]);
  const createCreativeTask = useCallback((creativePrompt: string) => { setPrompt(`Create a ${pathname.slice(1)} concept and execution plan for: ${creativePrompt}`); navigate("/"); }, [navigate, pathname]);
  const pageTitle = useMemo(() => pathname === "/image" ? "Image studio" : pathname === "/video" ? "Motion studio" : pathname === "/audio" ? "Sound lab" : pathname === "/agents" ? "Agents" : pathname === "/settings" ? "Settings" : "Chat", [pathname]);
  const isCreative = pathname === "/image" || pathname === "/video" || pathname === "/audio";

  return <div className="flex h-dvh overflow-hidden bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100"><Sidebar pathname={pathname} sessions={sessions} selectedSession={selectedSession} isDark={isDark} isOpen={isSidebarOpen} isCollapsed={isSidebarCollapsed} onNavigate={navigate} onNewChat={() => { void createChat(); navigate("/"); }} onSelectSession={(id) => void selectSession(id)} onDeleteSession={(session) => void deleteSession(session)} onToggleTheme={toggleTheme} onClearVram={() => void clearVram()} onClose={() => setIsSidebarOpen(false)} onToggleCollapse={() => setIsSidebarCollapsed((collapsed) => !collapsed)} /><main className="flex min-w-0 flex-1 flex-col overflow-hidden"><Header models={models} selectedModel={selectedModel} isOpen={isModelMenuOpen} title={pageTitle} onModelChange={(id) => { setSelectedModel(id); setIsModelMenuOpen(false); }} onToggleModels={() => setIsModelMenuOpen((open) => !open)} onMenu={() => setIsSidebarOpen(true)} onNewChat={() => void createChat()} />
    {pathname === "/agents" || pathname === "/settings" ? <SettingsPage /> : isCreative ? <CreativeStudio kind={pathname.slice(1) as "image" | "video" | "audio"} onCreate={createCreativeTask} /> : <><section className="flex-1 overflow-y-auto px-4 py-8"><div className="mx-auto flex max-w-4xl flex-col gap-6">{messages.length === 0 ? <div className="grid min-h-[55vh] place-items-center text-center"><div><div className="mx-auto mb-5 grid size-16 place-items-center rounded-3xl bg-gradient-to-br from-indigo-500 to-fuchsia-500 text-white shadow-2xl shadow-indigo-500/30"><Sparkles size={29} /></div><p className="mb-3 text-xs font-bold uppercase tracking-[.2em] text-indigo-500">Agent workspace</p><h1 className="text-3xl font-bold tracking-tight sm:text-5xl">What will we make?</h1><p className="mt-3 text-slate-500">Research with agents, create in the studio, or attach a file to begin.</p><div className="mt-6 flex flex-wrap justify-center gap-2"><button onClick={() => navigate("/image")} className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-600 shadow-sm hover:border-indigo-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300">Create an image</button><button onClick={() => navigate("/agents")} className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-600 shadow-sm hover:border-indigo-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300">Configure agents</button></div></div></div> : messages.map((message, index) => <ChatMessage key={`${message.role}-${index}`} message={message} copied={copiedIndex === index} onCopy={() => copy(message.content, index)} />)}{isGenerating && <p className="text-sm text-indigo-500">Qwen is thinking…</p>}<div ref={chatEndRef} /></div></section><ChatComposer prompt={prompt} file={file} mode={mode} isGenerating={isGenerating} onPromptChange={setPrompt} onFileChange={setFile} onModeChange={setMode} onSubmit={() => void sendMessage()} /></>}
  </main></div>;
}
