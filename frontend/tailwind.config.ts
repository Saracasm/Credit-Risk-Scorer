import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: "#060e14",
        "surface-raised": "#0a1620",
        "surface-container": "#0d1e2b",
        outline: "#8f909e",
        primary: "#00f2fe",
        on: { surface: "#e3e1eb", muted: "#a0aab2" },
        error: "#ffb4ab",
        ok: "#10b981",
        warn: "#ffb964",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        display: ["var(--font-hanken)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains)", "ui-monospace", "monospace"],
      },
      borderRadius: {
        glass: "12px",
      },
    },
  },
  plugins: [],
};

export default config;
