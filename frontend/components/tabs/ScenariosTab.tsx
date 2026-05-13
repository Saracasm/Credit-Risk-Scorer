"use client";

import { useState } from "react";
import { ApplicantInput, ScenarioResponse, runScenarios } from "@/lib/api";
import ShapBars from "@/components/ShapBars";

const SCENARIO_LABELS = [
  "Scenario A — Higher Income",
  "Scenario B — Lower Debt",
  "Scenario C — No Late Payments",
];

export default function ScenariosTab({ baseline }: { baseline: ApplicantInput }) {
  const [result, setResult] = useState<ScenarioResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const scenarios: ApplicantInput[] = [
    { ...baseline, monthly_income: baseline.monthly_income * 1.5 },
    { ...baseline, debt_ratio: Math.max(0.1, baseline.debt_ratio * 0.5) },
    { ...baseline, n_30_59_late: 0, n_60_89_late: 0, n_90_late: 0 },
  ];

  async function compare() {
    setLoading(true);
    setError(null);
    try {
      const r = await runScenarios(baseline, scenarios);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6 animate-fade-in-up">
      <div className="glass-elevated p-6">
        <h2 className="font-display text-lg font-semibold text-white mb-1">
          🔄 What-If Scenario Comparison
        </h2>
        <p className="text-sm text-on-muted">
          See how changes to income, debt, or payment history would affect this applicant&apos;s risk score.
        </p>
        <p className="text-xs text-outline mt-2 font-mono">
          Baseline: Age {baseline.age}, Income ${baseline.monthly_income.toLocaleString()},
          Debt Ratio {baseline.debt_ratio.toFixed(2)}
        </p>
      </div>

      <button onClick={compare} disabled={loading} className="btn-primary">
        {loading ? "Comparing…" : "🔍 Compare Scenarios"}
      </button>

      {error && (
        <div className="glass border-error/30 p-4 text-error text-sm font-mono">
          {error}
        </div>
      )}

      {result && (
        <>
          <div className="glass p-4 text-center">
            <span className="text-xs font-bold text-outline uppercase tracking-widest">
              Baseline Risk
            </span>
            <p className="font-mono text-3xl font-bold text-white mt-1">
              {Math.round(result.baseline_probability * 100)}%
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            {result.results.map((sc, i) => {
              const changePct = Math.round(sc.change * 100);
              const improved = sc.change < 0;
              return (
                <div
                  key={i}
                  className="glass-elevated p-5 space-y-4 animate-fade-in-up"
                  style={{ animationDelay: `${i * 0.1}s` }}
                >
                  <h3 className="text-xs font-bold text-primary uppercase tracking-wider">
                    {SCENARIO_LABELS[i]}
                  </h3>

                  <div className="flex items-baseline gap-3">
                    <span className="font-mono text-3xl font-bold text-white">
                      {Math.round(sc.probability * 100)}%
                    </span>
                    <span
                      className="text-sm font-mono font-bold"
                      style={{ color: improved ? "#10b981" : "#ffb4ab" }}
                    >
                      {changePct > 0 ? "+" : ""}{changePct}pp
                    </span>
                  </div>

                  <div
                    className="text-[10px] font-bold uppercase tracking-wider inline-block px-3 py-1 rounded-full"
                    style={{
                      color:
                        sc.risk_label === "HIGH" ? "#ffb4ab" :
                        sc.risk_label === "MEDIUM" ? "#ffb964" : "#10b981",
                      background:
                        sc.risk_label === "HIGH" ? "rgba(255,180,171,0.1)" :
                        sc.risk_label === "MEDIUM" ? "rgba(255,185,100,0.1)" : "rgba(16,185,129,0.1)",
                    }}
                  >
                    {sc.risk_label} Risk
                  </div>

                  <div className="border-t border-white/5 pt-3">
                    <p className="text-[10px] font-bold text-outline uppercase tracking-wider mb-2">
                      Top Factors
                    </p>
                    {sc.shap.slice(0, 5).map((s) => (
                      <div key={s.feature} className="flex justify-between text-xs py-0.5">
                        <span className="text-on-muted truncate mr-2">
                          {s.feature.replace(/_/g, " ")}
                        </span>
                        <span
                          className="font-mono"
                          style={{ color: s.value > 0 ? "#ffb4ab" : "#10b981" }}
                        >
                          {s.value > 0 ? "+" : ""}{s.value.toFixed(3)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
