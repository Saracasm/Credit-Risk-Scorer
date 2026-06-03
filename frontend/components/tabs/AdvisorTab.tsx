"use client";

import { useState, useRef, useEffect } from "react";
import { ApplicantInput, askAdvisor } from "@/lib/api";

type Message = { role: "user" | "assistant"; content: string };

const QUICK_QUESTIONS = [
  { label: "💡 Why this score?", prompt: "Why did this applicant get this risk score? Explain the top factors." },
  { label: "📈 How to improve?", prompt: "What are the top 3 things this applicant can do to reduce their default risk?" },
  { label: "💰 Loan recommendation", prompt: "Based on this profile, what loan would you recommend? Include amount, rate, and conditions." },
];

const MODELS = [
  { id: "groq-llama33-70b", label: "🆓 Llama 3.3 70B (Groq)", provider: "groq", model: "llama-3.3-70b-versatile", requiresKey: false },
  { id: "groq-llama31-8b", label: "🆓 Llama 3.1 8B (Groq)", provider: "groq", model: "llama-3.1-8b-instant", requiresKey: false },
  { id: "openai-gpt4o", label: "💎 GPT-4o (OpenAI Pro)", provider: "openai", model: "gpt-4o", requiresKey: true, link: "https://platform.openai.com/api-keys" },
  { id: "openai-gpt4omini", label: "💎 GPT-4o Mini (OpenAI Pro)", provider: "openai", model: "gpt-4o-mini", requiresKey: true, link: "https://platform.openai.com/api-keys" },
];

export default function AdvisorTab({ applicant }: { applicant: ApplicantInput }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState(MODELS[0].id);
  const [apiKey, setApiKey] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send(text: string) {
    if (!text.trim() || loading) return;
    const userMsg: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const selectedModel = MODELS.find((m) => m.id === selectedModelId) || MODELS[0];
      const res = await askAdvisor(
        text,
        applicant,
        selectedModel.provider,
        selectedModel.model,
        apiKey,
        sessionId,
      );
      setSessionId(res.session_id);
      setMessages((prev) => [...prev, { role: "assistant", content: res.reply }]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `❌ ${e instanceof Error ? e.message : "Request failed"}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in-up">
      <div className="glass-elevated p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="font-display text-lg font-semibold text-white mb-1">
              🤖 Credit Advisor Agent
            </h2>
            <p className="text-sm text-on-muted">
              Chat with an AI advisor that can explain scores, run what-if scenarios,
              suggest improvements, and recommend loan terms.
            </p>
          </div>
          
          <div className="flex flex-col gap-2 min-w-[250px]">
            <select 
              className="px-3 py-2 text-sm rounded-lg bg-background border border-surface text-white focus:outline-none focus:ring-1 focus:ring-primary"
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
            >
              {MODELS.map(m => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
            
            {(() => {
              const model = MODELS.find(m => m.id === selectedModelId);
              if (model?.requiresKey) {
                return (
                  <div className="flex flex-col gap-2 mt-1 max-w-[300px]">
                    <div className="flex items-start gap-2 text-[11px] text-on-muted bg-surface/30 p-2 rounded border border-surface/50 leading-tight">
                      <span className="text-primary text-xs">🔒</span>
                      <span><b>Privacy First:</b> Your key is processed securely in-memory for this session only. It is <b>never</b> stored on our servers.</span>
                    </div>
                    <input 
                      type="password" 
                      placeholder={`Paste your ${model.provider} API Key`}
                      className="px-3 py-2 text-sm rounded-lg bg-background border border-surface text-white focus:outline-none focus:ring-1 focus:ring-primary"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                    />
                    <div className="flex justify-between items-center px-1">
                      <span className="text-[10px] text-on-muted/50 uppercase tracking-wider font-semibold">BYOK Supported</span>
                      <a href={model.link} target="_blank" rel="noreferrer" className="text-xs text-primary hover:underline text-right">
                        Get API key here ↗
                      </a>
                    </div>
                  </div>
                );
              }
              return null;
            })()}
          </div>
        </div>
      </div>

      {/* Chat area */}
      <div className="glass-elevated p-4 space-y-4 max-h-[500px] overflow-y-auto">
        {messages.length === 0 && (
          <div className="text-center py-12 text-on-muted text-sm">
            <p className="text-3xl mb-3">💬</p>
            <p>Ask anything about the applicant&apos;s credit risk profile.</p>
            <p className="text-outline mt-1">Try a quick question below to get started.</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`max-w-[80%] px-4 py-3 text-sm whitespace-pre-wrap ${
              msg.role === "user"
                ? "chat-user ml-auto"
                : "chat-assistant mr-auto"
            }`}
          >
            {msg.content}
          </div>
        ))}

        {loading && (
          <div className="chat-assistant mr-auto max-w-[80%] px-4 py-3 text-sm text-on-muted">
            <span className="inline-flex gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: "0ms" }} />
              <span className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: "150ms" }} />
              <span className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: "300ms" }} />
            </span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Quick questions */}
      <div className="flex flex-wrap gap-2">
        {QUICK_QUESTIONS.map((q) => (
          <button
            key={q.label}
            className="btn-ghost text-xs"
            onClick={() => send(q.prompt)}
            disabled={loading}
          >
            {q.label}
          </button>
        ))}
      </div>

      {/* Input bar */}
      <div className="flex gap-3">
        <input
          type="text"
          className="flex-1 px-4 py-3 text-sm rounded-lg"
          placeholder="Ask about this applicant..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send(input)}
          disabled={loading}
        />
        <button
          className="btn-primary"
          onClick={() => send(input)}
          disabled={loading || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}
