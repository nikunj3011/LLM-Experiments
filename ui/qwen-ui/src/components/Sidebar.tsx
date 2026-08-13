import { Image, MessageSquare, Moon, Plus, Sun, Trash2, Video } from "lucide-react";
import type { SessionOption } from "../types";
import { AppIcon } from "./AppIcon";

interface SidebarProps {
  pathname: string; sessions: SessionOption[]; selectedSession: string; isDark: boolean; isOpen: boolean;
  onNavigate: (path: string) => void; onNewChat: () => void; onSelectSession: (id: string) => void; onDeleteSession: (session: SessionOption) => void; onToggleTheme: () => void; onClearVram: () => void; onClose: () => void;
}

export function Sidebar({ pathname, sessions, selectedSession, isDark, isOpen, onNavigate, onNewChat, onSelectSession, onDeleteSession, onToggleTheme, onClearVram, onClose }: SidebarProps) {
  const nav = [{ path: "/", label: "Chat", icon: MessageSquare }, { path: "/image", label: "Images", icon: Image }, { path: "/video", label: "Videos", icon: Video }];
  return <>
    {isOpen && <button aria-label="Close navigation" className="fixed inset-0 z-30 bg-slate-950/40 md:hidden" onClick={onClose} />}
    <aside className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-slate-200/80 bg-white/90 p-3 backdrop-blur-xl transition-transform dark:border-slate-800 dark:bg-slate-950/90 md:relative md:translate-x-0 ${isOpen ? "translate-x-0" : "-translate-x-full"}`}>
      <div className="flex items-center gap-3 px-2 py-3"><AppIcon /><div><p className="font-bold tracking-tight text-slate-900 dark:text-white">Qwen Studio</p><p className="text-xs text-slate-500">Local AI workspace</p></div></div>
      <button onClick={onNewChat} className="my-4 flex items-center justify-center gap-2 rounded-xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-slate-950/10 transition hover:bg-indigo-600 dark:bg-white dark:text-slate-950"><Plus size={17} /> New conversation</button>
      <nav className="space-y-1">{nav.map(({ path, label, icon: Icon }) => <button key={path} onClick={() => onNavigate(path)} className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition ${pathname === path ? "bg-indigo-50 text-indigo-700 dark:bg-indigo-500/15 dark:text-indigo-300" : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-900"}`}><Icon size={17} />{label}</button>)}</nav>
      <div className="app-scrollbar mt-6 min-h-0 flex-1 overflow-y-auto pr-1"><p className="px-3 pb-2 text-[11px] font-bold uppercase tracking-wider text-slate-400">Recent conversations</p>{sessions.map((session) => <div key={session.id} className={`group flex items-center rounded-xl ${selectedSession === session.id ? "bg-slate-100 dark:bg-slate-900" : "hover:bg-slate-50 dark:hover:bg-slate-900/60"}`}><button onClick={() => onSelectSession(session.id)} className="min-w-0 flex-1 truncate px-3 py-2.5 text-left text-sm text-slate-600 dark:text-slate-300">{session.title}</button><button aria-label={`Delete ${session.title}`} onClick={() => onDeleteSession(session)} className="mr-1 hidden rounded-lg p-1.5 text-slate-400 hover:bg-red-50 hover:text-red-500 group-hover:block dark:hover:bg-red-950/30"><Trash2 size={14} /></button></div>)}</div>
      <div className="space-y-1 border-t border-slate-200 pt-3 dark:border-slate-800"><button onClick={onClearVram} className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-slate-500 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/30"><Trash2 size={17} /> Clear VRAM</button><button onClick={onToggleTheme} className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-900">{isDark ? <Sun size={17} /> : <Moon size={17} />}{isDark ? "Light theme" : "Dark theme"}</button></div>
    </aside>
  </>;
}
