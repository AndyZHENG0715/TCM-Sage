"use client";

import React, { use } from "react";
import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";
import { useI18n } from "@/i18n/context";

function KGExplorerLoading() {
  const { t } = useI18n();

  return (
    <div className="h-screen w-full flex items-center justify-center bg-background-dark">
      <div className="flex flex-col items-center gap-3">
        <Loader2 className="animate-spin text-primary" size={32} />
        <p className="text-sm text-slate-400">{t.kgExplorer.loading}</p>
      </div>
    </div>
  );
}

const KGExplorerContent = dynamic(() => import("./KGExplorerContent"), {
  ssr: false,
  loading: () => <KGExplorerLoading />,
});

// Error boundary to suppress cytoscape renderer.notify SSR errors
// that don't affect functionality
class CytoscapeErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: false }; // Don't show fallback — the error is non-blocking
  }

  componentDidCatch(error: Error) {
    // Suppress renderer.notify errors silently — features work fine
    if (error.message?.includes("renderer") || error.message?.includes("notify")) {
      return;
    }
    // Re-throw non-cytoscape errors
    throw error;
  }

  render() {
    return this.props.children;
  }
}

interface PageProps {
  params: Promise<{ entityId: string }>;
}

export default function KGExplorerPage({ params }: PageProps) {
  const { entityId } = use(params);
  return (
    <CytoscapeErrorBoundary>
      <KGExplorerContent entityId={entityId} />
    </CytoscapeErrorBoundary>
  );
}
