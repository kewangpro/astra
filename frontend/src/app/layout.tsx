"use client";

import "./globals.css";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: { queries: { staleTime: 5000, refetchInterval: 5000 } },
      })
  );

  return (
    <html lang="en">
      <head>
        <title>ASTRA — Autonomous Strategic Training Agent</title>
        <meta name="description" content="ML orchestration dashboard" />
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚡</text></svg>" />
      </head>
      <body>
        <QueryClientProvider client={queryClient}>
          <div className="min-h-screen bg-[#070710]">
            <nav className="border-b border-[rgba(20,184,166,0.15)] px-6 py-3 flex items-center gap-3">
              <span className="text-teal font-semibold tracking-widest text-sm">ASTRA</span>
              <span className="text-[#334155] text-xs">autonomous strategic training agent</span>
            </nav>
            <main>{children}</main>
          </div>
        </QueryClientProvider>
      </body>
    </html>
  );
}
