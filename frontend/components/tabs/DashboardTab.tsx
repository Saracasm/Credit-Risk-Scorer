"use client";

import { useEffect, useState } from "react";
import { DashboardData, KnowledgeInfo, getDashboard, getKnowledge } from "@/lib/api";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  PieChart, Pie, Cell, ResponsiveContainer,
} from "recharts";

const PIE_COLORS: Record<string, string> = {
  LOW: "#10b981",
  MEDIUM: "#ffb964",
  HIGH: "#ffb4ab",
};

export default function DashboardTab() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [kb, setKb] = useState<KnowledgeInfo | null>(null);

  useEffect(() => {
    getDashboard()
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
    getKnowledge().then(setKb).catch(() => setKb(null));
  }, []);

  if (loading) {
    return (
      <div className="space-y-4 animate-fade-in-up">
        {[1, 2, 3].map((i) => (
          <div key={i} className="glass h-24 shimmer" />
        ))}
      </div>
    );
  }

  if (!data || data.total === 0) {
    return (
      <div className="glass-elevated p-12 text-center animate-fade-in-up">
        <span className="text-4xl">📊</span>
        <p className="text-on-muted mt-3">No predictions yet. Score some applicants to see analytics.</p>
      </div>
    );
  }

  // Build histogram data
  const bins = 20;
  const histData: { range: string; count: number }[] = [];
  for (let i = 0; i < bins; i++) {
    const lo = i / bins;
    const hi = (i + 1) / bins;
    const count = data.histogram.filter((v) => v >= lo && v < hi).length;
    histData.push({ range: `${Math.round(lo * 100)}%`, count });
  }

  // Pie data
  const pieData = [
    { name: "LOW", value: data.low_count },
    { name: "MEDIUM", value: data.medium_count },
    { name: "HIGH", value: data.high_count },
  ].filter((d) => d.value > 0);

  return (
    <div className="space-y-6 animate-fade-in-up">
      <div className="glass-elevated p-6">
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div>
            <h2 className="font-display text-lg font-semibold text-white mb-1">
              📊 Portfolio Risk Dashboard
            </h2>
            <p className="text-sm text-on-muted">
              Aggregate analytics across all scored applicants.
            </p>
          </div>
          {kb?.available && (
            <span
              title={`Advisor policy knowledge base: ${kb.docs.join(", ")}`}
              className="inline-flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider px-3 py-1.5 rounded-full whitespace-nowrap"
              style={{ color: "#00f2fe", background: "rgba(0,242,254,0.08)", border: "1px solid rgba(0,242,254,0.2)" }}
            >
              📚 Policy RAG · {kb.doc_count} docs · {kb.chunk_count} sections
            </span>
          )}
        </div>
      </div>

      {/* KPI cards */}
      <div className="grid sm:grid-cols-4 gap-4">
        {[
          { label: "Total Scored", value: data.total.toString(), color: "#b9c3ff" },
          { label: "Avg Risk", value: `${Math.round(data.avg_prob * 100)}%`, color: "#b9c3ff" },
          { label: "High Risk", value: data.high_count.toString(), color: "#ffb4ab" },
          { label: "Low Risk", value: data.low_count.toString(), color: "#10b981" },
        ].map((kpi) => (
          <div key={kpi.label} className="glass p-5 animate-fade-in-up">
            <p className="text-[10px] font-bold text-outline uppercase tracking-wider">{kpi.label}</p>
            <p className="font-mono text-3xl font-bold mt-1" style={{ color: kpi.color }}>
              {kpi.value}
            </p>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="glass-elevated p-5">
          <h3 className="text-xs font-bold text-outline uppercase tracking-wider mb-4">
            Risk Distribution
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={histData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="range" tick={{ fill: "#8f909e", fontSize: 10 }} />
              <YAxis tick={{ fill: "#8f909e", fontSize: 10 }} />
              <Tooltip
                contentStyle={{
                  background: "#1e1f26",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
              />
              <Bar dataKey="count" fill="#667eea" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="glass-elevated p-5">
          <h3 className="text-xs font-bold text-outline uppercase tracking-wider mb-4">
            Risk Level Breakdown
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={90}
                paddingAngle={3}
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {pieData.map((entry) => (
                  <Cell key={entry.name} fill={PIE_COLORS[entry.name] || "#666"} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: "#1e1f26",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent predictions table */}
      <div className="glass-elevated overflow-x-auto">
        <div className="p-4 border-b border-white/10">
          <h3 className="text-xs font-bold text-outline uppercase tracking-wider">
            Recent Predictions
          </h3>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/10">
              <th className="text-left px-4 py-3 text-[10px] font-bold text-outline uppercase tracking-wider">Timestamp</th>
              <th className="text-left px-4 py-3 text-[10px] font-bold text-outline uppercase tracking-wider">Age</th>
              <th className="text-left px-4 py-3 text-[10px] font-bold text-outline uppercase tracking-wider">Income</th>
              <th className="text-left px-4 py-3 text-[10px] font-bold text-outline uppercase tracking-wider">Probability</th>
              <th className="text-left px-4 py-3 text-[10px] font-bold text-outline uppercase tracking-wider">Risk</th>
            </tr>
          </thead>
          <tbody>
            {data.recent.map((row, i) => (
              <tr key={i} className="border-b border-white/5 hover:bg-white/[0.02]">
                <td className="px-4 py-2 text-xs text-outline font-mono">
                  {String(row.timestamp ?? "—").slice(0, 19)}
                </td>
                <td className="px-4 py-2 font-mono text-white">{String(row.age ?? "—")}</td>
                <td className="px-4 py-2 font-mono text-white">
                  ${Number(row.monthly_income ?? 0).toLocaleString()}
                </td>
                <td className="px-4 py-2 font-mono text-white">
                  {(Number(row.probability ?? 0) * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-2">
                  <span
                    className="text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-full"
                    style={{
                      color: PIE_COLORS[String(row.risk_level)] || "#666",
                      background:
                        row.risk_level === "HIGH" ? "rgba(255,180,171,0.1)" :
                        row.risk_level === "MEDIUM" ? "rgba(255,185,100,0.1)" : "rgba(16,185,129,0.1)",
                    }}
                  >
                    {String(row.risk_level ?? "—")}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
