"use client";

import { useState, useRef, useEffect } from "react";
import { ApplicantInput, KnowledgeInfo, askAdvisor, getKnowledge } from "@/lib/api";

type Message = { role: "user" | "assistant"; content: string };

/* ── Model tiers ──────────────────────────────────────────────────────────── */

const FREE_MODELS = [
  { id: "groq-llama33-70b", label: "Llama 3.3 70B", provider: "groq", model: "llama-3.3-70b-versatile" },
  { id: "groq-llama31-8b",  label: "Llama 3.1 8B (faster)", provider: "groq", model: "llama-3.1-8b-instant" },
];

const PRO_MODELS = [
  { id: "openai-gpt4o",     label: "GPT-4o",         provider: "openai", model: "gpt-4o",       link: "https://platform.openai.com/api-keys" },
  { id: "openai-gpt4omini", label: "GPT-4o Mini",    provider: "openai", model: "gpt-4o-mini",  link: "https://platform.openai.com/api-keys" },
  { id: "openai-gpt41",     label: "GPT-4.1",        provider: "openai", model: "gpt-4.1",      link: "https://platform.openai.com/api-keys" },
  { id: "openai-gpt41mini", label: "GPT-4.1 Mini",   provider: "openai", model: "gpt-4.1-mini", link: "https://platform.openai.com/api-keys" },
];

const ALL_MODELS = [...FREE_MODELS, ...PRO_MODELS];

/* ── Quick question groups ────────────────────────────────────────────────── */

const QUICK_GROUPS = [
  {
    category: "Understand", icon: "🔍", questions: [
      { label: "Why this score?",    prompt: "Why did this applicant get this risk score? Walk me through the top factors in plain English." },
      { label: "Biggest risk factor", prompt: "What is the single biggest factor raising this applicant's risk, and why?" },
    ],
  },
  {
    category: "Improve", icon: "📈", questions: [
      { label: "How to improve?", prompt: "What can this applicant do to reduce default risk? Show the combined impact of doing the top fixes together." },
      { label: "Fastest win",     prompt: "What is the single fastest change that would most lower this applicant's risk?" },
    ],
  },
  {
    category: "Eligibility", icon: "🏦", questions: [
      { label: "Do I qualify?",   prompt: "Based on the bank's policy, what risk tier and loan products would this applicant qualify for?" },
      { label: "Loan options",    prompt: "What loan products could this applicant realistically get? Cite the bank's policy." },
      { label: "Rebuild credit",  prompt: "If this applicant has weak credit, what does the bank's policy say about ways to rebuild it?" },
    ],
  },
  {
    category: "Process", icon: "📄", questions: [
      { label: "Documents needed", prompt: "What documents would this applicant need to provide to apply, per the bank's policy?" },
      { label: "If income drops",  prompt: "What hardship or payment-assistance options does the bank offer if this applicant loses income?" },
    ],
  },
  {
    category: "Fairness", icon: "⚖️", questions: [
      { label: "Why declined?",  prompt: "If declined, what specific adverse-action reasons would the bank give this applicant?" },
      { label: "Is this fair?",  prompt: "What does the bank's fair-lending policy say about how this decision must be made?" },
    ],
  },
];

/* ── Lightweight markdown renderer (no extra deps) ───────────────────────── */

function renderInline(text: string, key: string) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((p, i) =>
    p.startsWith("**") && p.endsWith("**")
      ? <strong key={`${key}-${i}`} className="text-white font-semibold">{p.slice(2, -2)}</strong>
      : <span key={`${key}-${i}`}>{p}</span>,
  );
}

function FormattedMessage({ content }: { content: string }) {
  return (
    <div className="space-y-1 leading-relaxed">
      {content.split("\n").map((line, i) => {
        const t = line.trim();
        if (!t) return <div key={i} className="h-2" />;
        const h = t.match(/^#{1,6}\s+(.*)/);
        if (h) return <p key={i} className="font-display font-semibold text-white mt-2">{renderInline(h[1], `h${i}`)}</p>;
        const b = t.match(/^[-*•]\s+(.*)/);
        if (b) return (
          <div key={i} className="flex gap-2 pl-1">
            <span className="text-primary mt-[3px] text-[10px]">●</span>
            <span className="flex-1">{renderInline(b[1], `b${i}`)}</span>
          </div>
        );
        const n = t.match(/^(\d+)[.)]\s+(.*)/);
        if (n) return (
          <div key={i} className="flex gap-2 pl-1">
            <span className="text-primary font-semibold tabular-nums">{n[1]}.</span>
            <span className="flex-1">{renderInline(n[2], `n${i}`)}</span>
          </div>
        );
        return <p key={i}>{renderInline(line, `p${i}`)}</p>;
      })}
    </div>
  );
}

/* ── Component ───────────────────────────────────────────────────────────── */

export default function AdvisorTab({ applicant }: { applicant: ApplicantInput }) {
  const [messages, setMessages]     = useState<Message[]>([]);
  const [input, setInput]           = useState("");
  const [loading, setLoading]       = useState(false);
  const [tier, setTier]             = useState<"free" | "pro">("free");
  const [freeModelId, setFreeModelId] = useState(FREE_MODELS[0].id);
  const [proModelId, setProModelId]   = useState(PRO_MODELS[0].id);
  const [apiKey, setApiKey]         = useState("");
  const [keyVisible, setKeyVisible] = useState(false);
  const [sessionId, setSessionId]   = useState<string | null>(null);
  const [kb, setKb]                 = useState<KnowledgeInfo | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);
  useEffect(() => { getKnowledge().then(setKb).catch(() => setKb(null)); }, []);

  function resetChat() { setMessages([]); setSessionId(null); setInput(""); }

  /* Derive the active model from the selected tier */
  const activeModel = tier === "free"
    ? FREE_MODELS.find((m) => m.id === freeModelId)!
    : PRO_MODELS.find((m) => m.id === proModelId)!;

  /* Pro tier is ready only when a key has been pasted */
  const proReady = tier === "free" || (tier === "pro" && apiKey.trim().length > 10);

  async function send(text: string) {
    if (!text.trim() || loading || !proReady) return;
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setLoading(true);

    try {
      const res = await askAdvisor(
        text,
        applicant,
        activeModel.provider,
        activeModel.model,
        /* key: send only for pro tier; free tier uses server env key */
        tier === "pro" ? apiKey.trim() : undefined,
        sessionId,
      );
      setSessionId(res.session_id);
      setMessages((prev) => [...prev, { role: "assistant", content: res.reply }]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Request failed";
      const friendly = msg.includes("401") || msg.includes("invalid_api_key")
        ? "Invalid API key. Please check your key and try again."
        : msg.includes("429") || msg.includes("rate")
        ? "Rate limit reached. Wait a moment and try again."
        : msg.includes("No API key")
        ? "Free tier is not configured on this server. Use the Pro tier with your own key."
        : msg;
      setMessages((prev) => [...prev, { role: "assistant", content: `❌ ${friendly}` }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-5 animate-fade-in-up">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="glass-elevated p-6">
        <div className="flex items-center gap-2 flex-wrap mb-1">
          <h2 className="font-display text-lg font-semibold text-white">🤖 Credit Advisor Agent</h2>
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
          documents, hardship help, or your rights.
        </p>
      </div>

      {/* ── Tier selector ──────────────────────────────────────────────────── */}
      <div className="grid sm:grid-cols-2 gap-3">

        {/* Free tier */}
        <button
          onClick={() => setTier("free")}
          className={`glass-elevated p-4 text-left transition-all duration-200 ${
            tier === "free" ? "ring-2 ring-primary/60" : "opacity-70 hover:opacity-90"
          }`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold uppercase tracking-wider text-on-muted">Free Tier</span>
            <span
              className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full"
              style={{ color: "#10b981", background: "rgba(16,185,129,0.12)", border: "1px solid rgba(16,185,129,0.25)" }}
            >
              No key needed
            </span>
          </div>
          <p className="text-sm text-white font-semibold mb-1">Groq · Llama 3.3 70B</p>
          <p className="text-xs text-on-muted">Powered by the app's Groq key. Fast, capable, completely free for you.</p>
          {tier === "free" && (
            <div className="mt-3">
              <select
                className="w-full px-3 py-1.5 text-xs rounded-lg bg-background border border-surface text-white focus:outline-none focus:ring-1 focus:ring-primary"
                value={freeModelId}
                onChange={(e) => setFreeModelId(e.target.value)}
                onClick={(e) => e.stopPropagation()}
              >
                {FREE_MODELS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
            </div>
          )}
        </button>

        {/* Pro tier */}
        <button
          onClick={() => setTier("pro")}
          className={`glass-elevated p-4 text-left transition-all duration-200 ${
            tier === "pro" ? "ring-2 ring-primary/60" : "opacity-70 hover:opacity-90"
          }`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold uppercase tracking-wider text-on-muted">Pro Tier</span>
            <span
              className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full"
              style={{ color: "#b9c3ff", background: "rgba(185,195,255,0.1)", border: "1px solid rgba(185,195,255,0.2)" }}
            >
              Your API key
            </span>
          </div>
          <p className="text-sm text-white font-semibold mb-1">OpenAI GPT-4o / GPT-4.1</p>
          <p className="text-xs text-on-muted">Use your own OpenAI key. Your key is used in-memory only — never stored or logged.</p>
          {tier === "pro" && (
            <div className="mt-3 space-y-2" onClick={(e) => e.stopPropagation()}>
              <select
                className="w-full px-3 py-1.5 text-xs rounded-lg bg-background border border-surface text-white focus:outline-none focus:ring-1 focus:ring-primary"
                value={proModelId}
                onChange={(e) => setProModelId(e.target.value)}
              >
                {PRO_MODELS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
              {/* Key input */}
              <div className="relative">
                <input
                  type={keyVisible ? "text" : "password"}
                  placeholder="sk-… paste your OpenAI key"
                  className="w-full px-3 py-2 pr-10 text-xs rounded-lg bg-background border border-surface text-white focus:outline-none focus:ring-1 focus:ring-primary font-mono"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  autoComplete="off"
                  spellCheck={false}
                />
                <button
                  type="button"
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-on-muted hover:text-white text-xs"
                  onClick={() => setKeyVisible((v) => !v)}
                  tabIndex={-1}
                >
                  {keyVisible ? "hide" : "show"}
                </button>
              </div>
              {/* Safety callout */}
              <div className="flex items-start gap-2 p-2 rounded text-[11px] text-on-muted leading-snug bg-surface/30 border border-surface/50">
                <span>🔒</span>
                <span>
                  Your key goes directly to OpenAI for this request only.
                  It is <strong className="text-white">never written to disk, logged, or stored</strong> on this server.
                  {" "}<a href={PRO_MODELS.find((m) => m.id === proModelId)?.link} target="_blank" rel="noreferrer" className="text-primary underline">Get a key ↗</a>
                </span>
              </div>
              {apiKey.trim().length > 0 && apiKey.trim().length <= 10 && (
                <p className="text-[11px] text-yellow-400/80">Key looks too short — double-check it.</p>
              )}
            </div>
          )}
        </button>

      </div>

      {/* ── Chat area ──────────────────────────────────────────────────────── */}
      <div className="glass-elevated p-4 space-y-4 max-h-[500px] overflow-y-auto">
        {messages.length === 0 && (
          <div className="text-center py-10 text-on-muted text-sm">
            <p className="text-3xl mb-3 animate-float-icon">💬</p>
            <p>Ask anything about this applicant&apos;s credit — not just loans.</p>
            <p className="text-outline mt-1">Pick a topic below or type your own question.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`max-w-[85%] px-4 py-3 text-sm ${
              msg.role === "user" ? "chat-user ml-auto whitespace-pre-wrap" : "chat-assistant mr-auto"
            }`}
          >
            {msg.role === "assistant" ? <FormattedMessage content={msg.content} /> : msg.content}
          </div>
        ))}
        {loading && (
          <div className="chat-assistant mr-auto px-4 py-3 text-sm text-on-muted">
            <span className="inline-flex gap-1">
              {[0, 150, 300].map((d) => (
                <span key={d} className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: `${d}ms` }} />
              ))}
            </span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* ── Quick questions ────────────────────────────────────────────────── */}
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
                  disabled={loading || !proReady}
                  title={!proReady ? "Paste your API key above first" : q.prompt}
                >
                  {q.label}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* ── Input bar ──────────────────────────────────────────────────────── */}
      <div className="flex gap-3">
        <input
          type="text"
          className="flex-1 px-4 py-3 text-sm rounded-lg"
          placeholder={proReady ? "Ask about risk, improving, eligibility, documents, hardship…" : "Paste your API key in the Pro tier card above first"}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send(input)}
          disabled={loading || !proReady}
        />
        <div className="flex flex-col items-end gap-1">
          <button
            className="btn-primary"
            onClick={() => send(input)}
            disabled={loading || !input.trim() || !proReady}
          >
            Send
          </button>
          {messages.length > 0 && (
            <button
              className="text-[10px] text-on-muted hover:text-white transition-colors px-2"
              onClick={resetChat}
              disabled={loading}
            >
              + New chat
            </button>
          )}
        </div>
      </div>

    </div>
  );
}
