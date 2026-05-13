"use client";

import { useState } from "react";
import { ApplicantInput, PredictResult, predictRisk, DEFAULT_APPLICANT } from "@/lib/api";
import ApplicantForm from "@/components/ApplicantForm";
import RiskGauge from "@/components/RiskGauge";
import ShapBars from "@/components/ShapBars";
import MetricCard from "@/components/MetricCard";

export default function ScorerTab({
  form,
  setForm,
  result,
  setResult,
}: {
  form: ApplicantInput;
  setForm: (f: ApplicantInput) => void;
  result: PredictResult | null;
  setResult: (r: PredictResult | null) => void;
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runAnalysis() {
    setLoading(true);
    setError(null);
    try {
      const r = await predictRisk(form);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const debtStatus = form.debt_ratio < 0.36 ? "OPTIMAL" : "HIGH";
  const utilPct = Math.round(form.revolving_util * 100);
  const utilStatus = form.revolving_util < 0.3 ? "HEALTHY" : "ELEVATED";

  return (
    <div className="grid lg:grid-cols-[1fr_320px] gap-6">
      {/* Main column */}
      <div className="space-y-6">
        <ApplicantForm form={form} onChange={setForm} />

        <button
          onClick={runAnalysis}
          disabled={loading}
          className="btn-primary w-full sm:w-auto"
        >
          {loading ? "Scoring…" : "🎯 Run Risk Analysis"}
        </button>

        {error && (
          <div className="glass border-error/30 p-4 text-error text-sm font-mono animate-fade-in-up">
            {error}
          </div>
        )}

        {/* Risk gauge */}
        <div className="glass-elevated p-6 animate-fade-in-up">
          <RiskGauge probability={result?.probability ?? null} />
        </div>

        {/* Metric cards */}
        <div className="grid sm:grid-cols-3 gap-4">
          <MetricCard
            label="Monthly Income"
            value={`$${form.monthly_income.toLocaleString()}`}
            icon="💰"
          />
          <MetricCard
            label="Debt-to-Income"
            value={form.debt_ratio.toFixed(2)}
            icon="⚖️"
            status={debtStatus}
            statusColor={debtStatus === "OPTIMAL" ? "#10b981" : "#ffb964"}
          />
          <MetricCard
            label="Credit Usage"
            value={`${utilPct}%`}
            icon="📈"
            status={utilStatus}
            statusColor={utilStatus === "HEALTHY" ? "#10b981" : "#ffb964"}
          />
        </div>

        {/* SHAP bars */}
        {result && <ShapBars items={result.shap} />}
      </div>

      {/* Sidebar */}
      <aside className="space-y-4">
        <div className="glass-elevated p-5">
          <div className="flex justify-between items-center mb-3">
            <div>
              <div className="text-[10px] font-bold tracking-[0.1em] text-primary uppercase">
                Quick Actions
              </div>
              <div className="text-[10px] font-bold tracking-[0.08em] text-outline uppercase mt-0.5">
                Institutional Grade
              </div>
            </div>
            <span className="text-lg">⚡</span>
          </div>
          <div className="h-px bg-white/5 my-3" />
          <div className="flex items-center gap-2.5">
            <span className="w-2 h-2 rounded-full bg-ok animate-pulse-glow" style={{ color: "#10b981" }} />
            <span className="text-[10px] font-bold text-outline uppercase tracking-[0.06em]">
              System Status: Nominal
            </span>
          </div>
        </div>

        {result && (
          <div className="glass p-5 space-y-3">
            <p className="text-[10px] font-bold uppercase tracking-widest text-primary">
              Score Summary
            </p>
            <div className="space-y-2 text-xs text-on-muted">
              <div className="flex justify-between">
                <span>Probability</span>
                <span className="font-mono text-white">{(result.probability * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span>Risk Level</span>
                <span
                  className="font-mono font-bold"
                  style={{
                    color:
                      result.risk_label === "HIGH" ? "#ffb4ab" :
                      result.risk_label === "MEDIUM" ? "#ffb964" : "#10b981",
                  }}
                >
                  {result.risk_label}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Classification</span>
                <span className="font-mono text-white">
                  {result.predicted_class === 1 ? "DEFAULT" : "NO DEFAULT"}
                </span>
              </div>
              <div className="flex justify-between">
                <span>SHAP Base</span>
                <span className="font-mono text-white">{result.shap_base_value.toFixed(4)}</span>
              </div>
            </div>
          </div>
        )}

        <div className="glass p-5 text-xs text-on-muted">
          <p className="font-bold text-white mb-2">How it works</p>
          <p className="leading-relaxed">
            Your XGBoost model scores each applicant. SHAP values explain
            which features pushed risk up (red) or down (green). Use the
            other tabs for deeper analysis.
          </p>
        </div>
      </aside>
    </div>
  );
}
