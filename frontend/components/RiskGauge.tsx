"use client";

/** Circular SVG risk gauge with animated ring and glow. */
export default function RiskGauge({
  probability,
  animated = true,
}: {
  probability: number | null;
  animated?: boolean;
}) {
  const pct = probability !== null ? Math.round(probability * 100) : null;

  let color: string, glow: string, label: string, labelBg: string;
  if (probability === null || probability === undefined) {
    color = "#8f909e";
    glow = "transparent";
    label = "AWAITING";
    labelBg = "transparent";
  } else if (probability > 0.6) {
    color = "#ffb4ab";
    glow = "rgba(255,180,171,0.15)";
    label = "HIGH RISK";
    labelBg = "rgba(255,180,171,0.1)";
  } else if (probability > 0.3) {
    color = "#ffb964";
    glow = "rgba(255,185,100,0.15)";
    label = "MEDIUM RISK";
    labelBg = "rgba(255,185,100,0.1)";
  } else {
    color = "#10b981";
    glow = "rgba(16,185,129,0.15)";
    label = "LOW RISK";
    labelBg = "rgba(16,185,129,0.1)";
  }

  const dash = probability !== null ? probability * 100 : 0;

  return (
    <div className="flex flex-col items-center py-8">
      <div
        className="relative flex items-center justify-center rounded-full"
        style={{
          width: 200,
          height: 200,
          border: probability !== null ? "4px solid rgba(255,255,255,0.04)" : "4px dashed rgba(255,255,255,0.12)",
          boxShadow: `0 0 50px ${glow}`,
        }}
      >
        <svg
          viewBox="0 0 36 36"
          className="absolute"
          style={{ width: 180, height: 180, transform: "rotate(-90deg)" }}
        >
          <path
            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none"
            stroke="rgba(255,255,255,0.04)"
            strokeWidth="2.5"
          />
          {probability !== null && (
            <path
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              fill="none"
              stroke={color}
              strokeWidth="2.5"
              strokeDasharray={`${dash}, 100`}
              strokeLinecap="round"
              className={animated ? "gauge-ring" : ""}
              style={{ filter: `drop-shadow(0 0 6px ${color})` }}
            />
          )}
        </svg>
        <div className="text-center z-10">
          <div className="font-mono text-5xl font-bold text-white leading-none tracking-tight">
            {pct !== null ? `${pct}%` : "—"}
          </div>
          <div className="text-[11px] font-bold text-outline tracking-[0.2em] uppercase mt-1">
            DEFAULT RISK
          </div>
        </div>
      </div>

      <div className="mt-5 flex flex-col items-center gap-2">
        <div
          className="flex items-center gap-2 px-4 py-1.5 rounded-full text-[11px] font-bold tracking-[0.05em]"
          style={{
            background: labelBg,
            border: `1px solid ${color}33`,
            color,
          }}
        >
          <span
            className="w-2 h-2 rounded-full animate-pulse-glow"
            style={{ background: color, color }}
          />
          {label}
        </div>
      </div>
    </div>
  );
}
