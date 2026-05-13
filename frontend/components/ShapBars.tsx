"use client";

import { ShapItem } from "@/lib/api";

/** Horizontal SHAP factor bars with risk/positive coloring. */
export default function ShapBars({ items }: { items: ShapItem[] }) {
  const maxAbs = Math.max(...items.map((s) => Math.abs(s.value)), 0.001);

  return (
    <div className="glass-elevated p-6 animate-fade-in-up animate-fade-in-up-delay-2">
      <div className="flex justify-between items-center mb-5">
        <h3 className="font-display text-lg font-semibold text-white">
          Key Risk Factors
        </h3>
        <span className="text-[10px] font-bold text-outline tracking-[0.08em] uppercase">
          SHAP · XGBoost
        </span>
      </div>

      <div className="space-y-1">
        {items.slice(0, 10).map((s) => {
          const isRisk = s.value > 0;
          const widthPct = Math.min((Math.abs(s.value) / maxAbs) * 42, 42);
          const barColor = isRisk ? "rgba(255,180,171,0.6)" : "rgba(16,185,129,0.6)";
          const textColor = isRisk ? "#ffb4ab" : "#10b981";
          const tag = isRisk ? "Risk" : "Positive";

          return (
            <div
              key={s.feature}
              className="grid items-center gap-3 py-1.5"
              style={{ gridTemplateColumns: "140px 1fr 130px" }}
            >
              <span className="text-[11px] font-bold text-on-muted tracking-wide uppercase text-right truncate">
                {s.feature.replace(/_/g, " ")}
              </span>

              <div className="relative h-3 flex items-center">
                <div className="absolute left-1/2 w-px h-full bg-white/15" />
                {isRisk ? (
                  <div
                    className="absolute left-1/2 h-2 rounded-r-sm transition-all duration-700"
                    style={{ width: `${widthPct}%`, background: barColor }}
                  />
                ) : (
                  <div
                    className="absolute right-1/2 h-2 rounded-l-sm transition-all duration-700"
                    style={{ width: `${widthPct}%`, background: barColor }}
                  />
                )}
              </div>

              <span className="font-mono text-xs" style={{ color: textColor }}>
                {s.value > 0 ? "+" : ""}
                {s.value.toFixed(3)} ({tag})
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
