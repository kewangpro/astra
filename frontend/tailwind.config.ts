import type { Config } from "tailwindcss";

export default {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        obsidian: {
          950: "#070710",
          900: "#0d0d1a",
          800: "#12122a",
          700: "#1a1a35",
          600: "#222248",
        },
        teal: {
          DEFAULT: "#14b8a6",
          light: "#2dd4bf",
          dark: "#0d9488",
          glow: "rgba(20,184,166,0.15)",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      animation: {
        "pulse-teal": "pulse-teal 2s ease-in-out infinite",
        "slide-in": "slide-in 0.2s ease-out",
      },
      keyframes: {
        "pulse-teal": {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(20,184,166,0)" },
          "50%": { boxShadow: "0 0 20px 4px rgba(20,184,166,0.3)" },
        },
        "slide-in": {
          from: { opacity: "0", transform: "translateY(-8px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
