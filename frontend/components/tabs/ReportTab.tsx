"use client";

import { useState } from "react";
import { ApplicantInput, PredictResult, downloadReport } from "@/lib/api";

export default function ReportTab({
  applicant,
  result,
}: {
  applicant: ApplicantInput;
  result: PredictResult | null;
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function generate() {
    setLoading(true);
    setError(null);
    try {
      const blob = await downloadReport(applicant);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "credit_assessment_report.pdf";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Report generation failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-fade-in-up">
      <div className="glass-elevated p-6">
        <h2 className="font-display text-lg font-semibold text-white mb-1">
          📄 Credit Assessment Report
        </h2>
        <p className="text-sm text-on-muted">
          Generate a professional PDF report with risk score, SHAP analysis,
          and loan recommendations.
        </p>
      </div>

      {!result ? (
        <div className="glass p-8 text-center">
          <span className="text-3xl">📝</span>
          <p className="text-on-muted mt-3">
            Score an applicant in the <span className="text-primary font-bold">Risk Scorer</span> tab
            first to generate a report.
          </p>
        </div>
      ) : (
        <>
          {/* Summary card */}
          <div className="glass-elevated p-6">
            <h3 className="text-xs font-bold text-outline uppercase tracking-wider mb-4">
              Report Preview
            </h3>
            <div className="grid sm:grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-on-muted">Applicant</span>
                <p className="font-mono text-white mt-1">
                  Age {applicant.age}, Income ${applicant.monthly_income.toLocaleString()}
                </p>
              </div>
              <div>
                <span className="text-on-muted">Default Risk</span>
                <p className="font-mono text-white mt-1">
                  {(result.probability * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <span className="text-on-muted">Risk Level</span>
                <p
                  className="font-mono font-bold mt-1"
                  style={{
                    color:
                      result.risk_label === "HIGH" ? "#ffb4ab" :
                      result.risk_label === "MEDIUM" ? "#ffb964" : "#10b981",
                  }}
                >
                  {result.risk_label}
                </p>
              </div>
            </div>

            <div className="border-t border-white/5 mt-4 pt-4">
              <p className="text-xs text-outline mb-2 font-bold uppercase tracking-wider">
                Report Includes
              </p>
              <ul className="text-xs text-on-muted space-y-1">
                <li>✓ Risk score with color-coded indicator</li>
                <li>✓ Applicant profile summary</li>
                <li>✓ SHAP factor analysis chart</li>
                <li>✓ Loan recommendation with terms</li>
                <li>✓ Professional formatting with disclaimer</li>
              </ul>
            </div>
          </div>

          <button
            onClick={generate}
            disabled={loading}
            className="btn-primary w-full sm:w-auto"
          >
            {loading ? "Generating…" : "⬇️ Generate & Download PDF"}
          </button>

          {error && (
            <div className="glass border-error/30 p-4 text-error text-sm font-mono">
              {error}
            </div>
          )}
        </>
      )}
    </div>
  );
}
