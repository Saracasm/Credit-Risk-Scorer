"use client";

import { useState, useRef, useEffect } from "react";
import { ApplicantInput, KnowledgeInfo, askAdvisor, getKnowledge } from "@/lib/api";

type Message = { role: "user" | "assistant"; content: string };

/* Quick questions grouped by intent — shows the advisor handles far more than
   loan sizing: risk explanation, improvement, eligibility, process, fairness. */
const QUICK_GROUPS: { category: string; icon: string; questions: { label: string; prompt: string }[] }[] = [
  {
    category: "Understand", icon: "🔍", questions: [
      { label: "Why this score?", prompt: "Why did this applicant get this risk score? Walk me through the top factors in plain English." },
      { label: "Biggest risk factor", prompt: "What is the single biggest factor raising this applicant's risk, and why?" },
    ],
  },
  {
    category: "Improve", icon: "📈", questions: [
      { label: "How to improve?", prompt: "What can this applicant do to reduce default risk? Show the combined impact of doing the top fixes together." },
      { label: "Fastest win", prompt: "What is the single fastest change that would most lower this applicant's risk?" },
    ],
  },
  {
    category: "Eligibility", icon: "🏦", questions: [
      { label: "Do I qualify?", prompt: "Based on the bank's policy, what risk tier and loan products would this applicant qualify for?" },
      { label: "Loan options", prompt: "What loan products could this applicant realistically get? Cite the bank's policy." },
      { label: "Rebuild credit", prompt: "If this applicant has weak credit, what does the bank's policy say about ways to rebuild it?" },
    ],
  },
  {
    category: "Process", icon: "📄", questions: [
      { label: "Documents needed", prompt: "What documents would this applicant need to provide to apply, per the bank's policy?" },
      { label: "If income drops", prompt: "What hardship or payment-assistance options does the bank offer if this applicant loses income?" },
    ],
  },
  {
    category: "Fairness", icon: "⚖️", questions: [
      { label: "Why declined?", prompt: "If declined, what specific adverse-action reasons would the bank give this applicant?" },
      { label: "Is this fair?", prompt: "What does the bank's fair-lending policy say about how this decision must be made?" },
    ],
  },
];

const MODELS = [
  { id: "groq-llama33-70b", label: "🆓 Llama 3.3 70B (Groq)", provider: "groq", model: "llama-3.3-70b-versatile", requiresKey: false },
  { id: "groq-llama31-8b", label: "🆓 Llama 3.1 8B (Groq)", provider: "groq", model: "llama-3.1-8b-instant", requiresKey: false },
  { id: "openai-gpt4o", label: "💎 GPT-4o (OpenAI Pro)", provider: "openai", model: "gpt-4o", requiresKey: true, link: "https://platform.openai.com/api-keys" },
  { id: "openai-gpt4omini", label: "💎 GPT-4o Mini (OpenAI Pro)", provider: "openai", model: "gpt-4o-mini", requiresKey: true, link: "https://platform.openai.com/api-keys" },
];

/* ---------- Lightweight markdown rendering (no external deps) ---------- */

function renderInline(text: string, keyBase: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((p, i) =>
    p.startsWith("**") && p.endsWith("**") ? (
      <strong key={`${keyBase}-${i}`} className="text-white font-semibold">{p.slice(2, -2)}</strong>
    ) : (
      <span key={`${keyBase}-${i}`}>{p}</span>
    ),
  );
}

function FormattedMessage({ content }: { content: string }) {
  const lines = content.split("\n");
  return (
    <div className="space-y-1 leading-relaxed">
      {lines.map((line, i) => {
        const trimmed = line.trim();
        if (!trimmed) return <div key={i} className="h-2" />;

        const heading = trimmed.match(/^#{1,6}\s+(.*)/);
        if (heading) return <p key={i} className="font-display font-semibold text-white mt-2">{renderInline(heading[1], `h${i}`)}</p>;

        const bullet = trimmed.match(/^[-*•]\s+(.*)/);
        if (bullet) return (
          <div key={i} className="flex gap-2 pl-1">
            <span className="text-primary mt-[3px] text-[10px]">●</span>
            <span className="flex-1">{renderInline(bullet[1], `b${i}`)}</span>
          </div>
        );

        const numbered = trimmed.match(/^(\d+)[.)]\s+(.*)/);
        if (numbered) return (
          <div key={i} className="flex gap-2 pl-1">
            <span className="text-primary font-semibold tabular-nums">{numbered[1]}.</span>
            <span className="flex-1">{renderInline(numbered[2], `n${i}`)}</span>
          </div>
        );

        return <p key={i}>{renderInline(line, `p${i}`)}</p>;
      })}
    </div>
  );
}

export default function AdvisorTab({ applicant }: { applicant: ApplicantInput }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState(MODELS[0].id);
  const [apiKey, setApiKey] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [kb, setKb] = useState<KnowledgeInfo | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    getKnowledge().then(setKb).catch(() => setKb(null));
  }, []);

  function resetChat() {
    setMessages([]);
    setSessionId(null);
    setInput("");
  }

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

  const model = MODELS.find((m) => m.id === selectedModelId);

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in-up">
      <div className="glass-elevated p-6">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 mb-2">
          <div>
            <div className="flex items-center gap-2 flex-wrap mb-1">
              <h2 className="font-display text-lg font-semibold text-white">
                🤖 Credit Advisor Agent
              </h2>
              {kb?.available && (
                <span
                  title={`Policy knowledge base: ${kb.docs.join(", ")}`}
                  className="inline-flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded-full"
                  style={{ color: "#00f2fe", background: "rgba(0,242,254,0.08)", border: "1px solid rgba(0,242,254,0.2)" }}
                >
                  📚 Policy-aware · {kb.doc_count} docs
                </span>
              )}
              {messages.length > 0 && (
                <span className="inline-flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded-full text-on-muted bg-surface/40 border border-surface/50">
                  🧠 Memory on
                </span>
              )}
            </div>
            <p className="text-sm text-on-muted">
              Ask anything — risk explanations, ways to improve, eligibility,
              required documents, hardship help, or your rights. Answers are
              grounded in the real model, SHAP, and the bank&apos;s policy.
            </p>
          </div>

          <div className="flex flex-col gap-2 min-w-[250px]">
            <div className="flex gap-2">
              <select
                className="flex-1 px-3 py-2 text-sm rounded-lg bg-background border border-surface text-white focus:outline-none focus:ring-1 focus:ring-primary"
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
              >
                {MODELS.map((m) => (
                  <option key={m.id} value={m.id}>{m.label}</option>
                ))}
              </select>
              <button
                className="btn-ghost text-xs whitespace-nowrap"
                onClick={resetChat}
                disabled={loading || messages.length === 0}
                title="Start a fresh conversation"
              >
                ＋ New
              </button>
            </div>

            {model?.requiresKey && (
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
            )}
          </div>
        </div>
      </div>

      {/* Chat area */}
      <div className="glass-elevated p-4 space-y-4 max-h-[500px] overflow-y-auto">
        {messages.length === 0 && (
          <div className="text-center py-10 text-on-muted text-sm">
            <p className="text-3xl mb-3 animate-float-icon">💬</p>
            <p>Ask anything about this applicant&apos;s credit — not just loans.</p>
            <p className="text-outline mt-1">Pick a topic below, or type your own question.</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`max-w-[85%] px-4 py-3 text-sm ${
              msg.role === "user"
                ? "chat-user ml-auto whitespace-pre-wrap"
                : "chat-assistant mr-auto"
            }`}
          >
            {msg.role === "assistant" ? <FormattedMessage content={msg.content} /> : msg.content}
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

      {/* Categorized quick questions */}
      <div className="glass-elevated p-4 space-y-3">
        {QUICK_GROUPS.map((group) => (
          <div key={group.category} className="flex flex-col sm:flex-row sm:items-center gap-2">
            <span className="text-[11px] font-bold uppercase tracking-wider text-on-muted min-w-[110px]">
              {group.icon} {group.category}
            </span>
            <div className="flex flex-wrap gap-2">
              {group.questions.map((q) => (
                <button
                  key={q.label}
                  className="btn-ghost text-xs normal-case tracking-normal"
                  onClick={() => send(q.prompt)}
                  disabled={loading}
                  title={q.prompt}
                >
                  {q.label}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Input bar */}
      <div className="flex gap-3">
        <input
          type="text"
          className="flex-1 px-4 py-3 text-sm rounded-lg"
          placeholder="Ask about risk, improving, eligibility, documents, hardship help…"
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
