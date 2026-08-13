import { Check, Copy, FileText, UserRound } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import type { Message } from "../types";

export function ChatMessage({ message, onCopy, copied }: { message: Message; onCopy: () => void; copied: boolean }) {
  const content = typeof message.content === "string" ? message.content : JSON.stringify(message.content, null, 2);
  const assistant = message.role === "assistant";
  return <article className={`group flex gap-3 ${assistant ? "" : "flex-row-reverse"}`}><div className={`grid size-9 shrink-0 place-items-center rounded-xl ${assistant ? "bg-gradient-to-br from-indigo-500 to-violet-600 text-white" : "bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-200"}`}>{assistant ? "Q" : <UserRound size={17} />}</div><div className={`min-w-0 max-w-[85%] rounded-2xl px-4 py-3 shadow-sm ${assistant ? "bg-white dark:bg-slate-900" : "bg-slate-950 text-white dark:bg-indigo-600"}`}>{message.filePreview && <img src={message.filePreview} alt={message.fileName ?? "Attachment"} className="mb-3 max-h-80 rounded-xl object-contain" />}{message.fileName && !message.filePreview && <p className="mb-2 flex items-center gap-2 text-sm"><FileText size={15} />{message.fileName}</p>}<div className={assistant ? "chat-markdown text-slate-700 dark:text-slate-200" : "whitespace-pre-wrap"}>{assistant ? <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>{content || ""}</ReactMarkdown> : content}</div>{assistant && content && <button onClick={onCopy} className="mt-3 inline-flex items-center gap-1.5 text-xs text-slate-400 opacity-0 transition group-hover:opacity-100 hover:text-indigo-500">{copied ? <Check size={14} /> : <Copy size={14} />}{copied ? "Copied" : "Copy"}</button>}</div></article>;
}
