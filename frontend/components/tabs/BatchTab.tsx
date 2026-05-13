"use client";

import { useState, useCallback } from "react";
import { BatchResponse, batchScore } from "@/lib/api";

const TEMPLATE_CSV = `age,MonthlyIncome,DebtRatio,RevolvingUtilizationOfUnsecuredLines,NumberOfTime30-59DaysPastDueNotWorse,NumberOfTime60-89DaysPastDueNotWorse,NumberOfTimes90DaysLate,NumberOfOpenCreditLinesAndLoans,NumberRealEstateLoansOrLines,NumberOfDependents
45,5000,0.3,0.2,0,0,0,5,1,1
25,3500,0.35,0.45,0,0,0,3,0,0
38,2800,0.85,0.92,3,2,1,12,1,3`;

export default function BatchTab() {
  const [result, setResult] = useState<BatchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  function downloadTemplate() {
    const blob = new Blob([TEMPLATE_CSV], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "template.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  const handleFile = useCallback(async (file: File) => {
    setFileName(file.name);
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const text = await file.text();
      const lines = text.trim().split("\n");
      const headers = lines[0].split(",").map((h) => h.trim());
      const applicants = lines.slice(1).map((line) => {
        const vals = line.split(",");
        const obj: Record<string, unknown> = {};
        headers.forEach((h, i) => {
          obj[h] = parseFloat(vals[i]) || 0;
        });
        return obj;
      });

      const r = await batchScore(applicants);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Batch scoring failed");
    } finally {
      setLoading(false);
    }
  }, []);

  function downloadResults() {
    if (!result) return;
    const header = "index,probability,predicted_class,risk_label\n";
    const rows = result.rows.map((r) => `${r.index},${r.probability},${r.predicted_class},${r.risk_label}`).join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "scored_applicants.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6 animate-fade-in-up">
      <div className="glass-elevated p-6">
        <h2 className="font-display text-lg font-semibold text-white mb-1">
          📋 Batch CSV Scoring
        </h2>
        <p className="text-sm text-on-muted">
          Upload a CSV of applicants to score them all at once. Download results with risk levels.
        </p>
      </div>

      <div className="flex gap-3">
        <button onClick={downloadTemplate} className="btn-ghost">
          ⬇️ Download Template
        </button>
      </div>

      {/* Drop zone */}
      <label
        className="glass-elevated p-12 flex flex-col items-center justify-center cursor-pointer
                   border-2 border-dashed border-white/10 hover:border-primary/30 transition-colors"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const f = e.dataTransfer.files[0];
          if (f) handleFile(f);
        }}
      >
        <span className="text-4xl mb-3">📄</span>
        <p className="text-sm text-on-muted">
          Drag & drop a CSV file or <span className="text-primary underline">browse</span>
        </p>
        {fileName && <p className="text-xs text-outline mt-2 font-mono">{fileName}</p>}
        <input
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
          }}
        />
      </label>

      {loading && (
        <div className="glass p-6 text-center">
          <div className="shimmer h-4 w-48 mx-auto rounded mb-2" />
          <p className="text-sm text-on-muted">Scoring applicants…</p>
        </div>
      )}

      {error && (
        <div className="glass border-error/30 p-4 text-error text-sm font-mono">{error}</div>
      )}

      {result && (
        <>
          {/* Summary metrics */}
          <div className="grid sm:grid-cols-4 gap-4">
            <div className="glass p-4 text-center">
              <p className="text-[10px] font-bold text-outline uppercase tracking-wider">Total</p>
              <p className="font-mono text-2xl text-white mt-1">{result.total}</p>
            </div>
            <div className="glass p-4 text-center">
              <p className="text-[10px] font-bold text-outline uppercase tracking-wider">Avg Risk</p>
              <p className="font-mono text-2xl text-white mt-1">{Math.round(result.avg_risk * 100)}%</p>
            </div>
            <div className="glass p-4 text-center">
              <p className="text-[10px] font-bold text-outline uppercase tracking-wider">High Risk</p>
              <p className="font-mono text-2xl text-error mt-1">{result.high_risk_count}</p>
            </div>
            <div className="glass p-4 text-center">
              <p className="text-[10px] font-bold text-outline uppercase tracking-wider">Low Risk</p>
              <p className="font-mono text-2xl text-ok mt-1">{result.low_risk_count}</p>
            </div>
          </div>

          {/* Results table */}
          <div className="glass-elevated overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left px-4 py-3 text-[10px] font-bold text-outline uppercase tracking-wider">#</th>
                  <th className="text-left px-4 py-3 text-[10px] font-bold text-outline uppercase tracking-wider">Probability</th>
                  <th className="text-left px-4 py-3 text-[10px] font-bold text-outline uppercase tracking-wider">Class</th>
                  <th className="text-left px-4 py-3 text-[10px] font-bold text-outline uppercase tracking-wider">Risk Level</th>
                </tr>
              </thead>
              <tbody>
                {result.rows.map((row) => (
                  <tr key={row.index} className="border-b border-white/5 hover:bg-white/[0.02]">
                    <td className="px-4 py-2 font-mono text-outline">{row.index + 1}</td>
                    <td className="px-4 py-2 font-mono text-white">{(row.probability * 100).toFixed(1)}%</td>
                    <td className="px-4 py-2 font-mono text-white">
                      {row.predicted_class === 1 ? "DEFAULT" : "OK"}
                    </td>
                    <td className="px-4 py-2">
                      <span
                        className="text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-full"
                        style={{
                          color:
                            row.risk_label === "HIGH" ? "#ffb4ab" :
                            row.risk_label === "MEDIUM" ? "#ffb964" : "#10b981",
                          background:
                            row.risk_label === "HIGH" ? "rgba(255,180,171,0.1)" :
                            row.risk_label === "MEDIUM" ? "rgba(255,185,100,0.1)" : "rgba(16,185,129,0.1)",
                        }}
                      >
                        {row.risk_label}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <button onClick={downloadResults} className="btn-primary">
            ⬇️ Download Results CSV
          </button>
        </>
      )}
    </div>
  );
}
