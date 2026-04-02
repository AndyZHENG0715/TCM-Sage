"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { fetchArenaStats } from "@/lib/api";
import { cn } from "@/lib/utils";
import { ArrowLeft, RefreshCw, Download } from "lucide-react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar, Pie } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

type QueryResult = {
  query: string;
  winner: string;
  model: string;
  timestamp: string;
  session_id: string;
};

type TTestResult = {
  t_statistic: number;
  p_value: number;
  cohens_d: number;
  mean_rag_score: number;
  sample_size: number;
  significant: boolean;
  interpretation: string;
};

type ArenaStats = {
  total_votes: number;
  rag_wins: number;
  plain_wins: number;
  ties: number;
  rag_win_rate: number;
  plain_win_rate: number;
  tie_rate: number;
  t_test: TTestResult | null;
  query_results: QueryResult[];
};

export default function ArenaStatsPage() {
  const [stats, setStats] = useState<ArenaStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const barChartRef = useRef<ChartJS<"bar">>(null);
  const pieChartRef = useRef<ChartJS<"pie">>(null);

  const loadStats = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchArenaStats();
      setStats(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load statistics.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStats();
  }, []);

  const downloadChart = (
    chartRef: React.RefObject<ChartJS<"bar"> | ChartJS<"pie"> | null>,
    filename: string
  ) => {
    if (!chartRef.current) return;
    const base64 = chartRef.current.toBase64Image();
    const link = document.createElement("a");
    link.download = filename;
    link.href = base64;
    link.click();
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background-dark text-parchment">
        <RefreshCw className="animate-spin text-primary w-8 h-8" />
        <span className="ml-3 font-medium">Loading statistics...</span>
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background-dark text-parchment">
        <div className="text-center">
          <p className="text-red-400 mb-4">{error || "No data available."}</p>
          <button
            onClick={loadStats}
            className="flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 rounded-lg hover:bg-white/10 transition"
          >
            <RefreshCw size={16} /> Retry
          </button>
        </div>
      </div>
    );
  }

  // Chart Data Preparation
  const chartColors = {
    rag: "#19e6d4",
    plain: "#fbbf24",
    tie: "#6b7280",
  };

  const barData = {
    labels: ["RAG Wins", "Plain LLM Wins", "Ties"],
    datasets: [
      {
        label: "Win Rate (%)",
        data: [stats.rag_win_rate, stats.plain_win_rate, stats.tie_rate],
        backgroundColor: [chartColors.rag, chartColors.plain, chartColors.tie],
        borderWidth: 1,
        borderColor: "#112120",
      },
    ],
  };

  const barOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: "Win Rate Distribution (%)",
        color: "#F3EFE0",
        font: { size: 16, family: "serif" },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: { color: "#9ca3af" },
        grid: { color: "rgba(255, 255, 255, 0.1)" },
      },
      x: {
        ticks: { color: "#F3EFE0" },
        grid: { display: false },
      },
    },
  };

  const pieData = {
    labels: ["RAG", "Plain LLM", "Tie"],
    datasets: [
      {
        data: [stats.rag_wins, stats.plain_wins, stats.ties],
        backgroundColor: [chartColors.rag, chartColors.plain, chartColors.tie],
        borderColor: "#112120",
        borderWidth: 2,
      },
    ],
  };

  const pieOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "bottom" as const,
        labels: { color: "#F3EFE0" },
      },
      title: {
        display: true,
        text: "Total Vote Distribution",
        color: "#F3EFE0",
        font: { size: 16, family: "serif" },
      },
    },
  };

  return (
    <div className="min-h-screen bg-background-dark text-parchment p-8 font-sans">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div className="space-y-2">
            <Link
              href="/arena"
              className="inline-flex items-center text-sm text-primary hover:underline group"
            >
              <ArrowLeft size={16} className="mr-1 group-hover:-translate-x-1 transition-transform" />
              Back to Arena
            </Link>
            <h1 className="text-3xl font-serif font-bold tracking-tight">
              Arena Evaluation Results
            </h1>
            <div className="inline-flex items-center px-3 py-1 rounded-full bg-white/5 border border-white/10 text-sm">
              <span className="font-semibold">{stats.total_votes}</span>
              <span className="ml-1 text-gray-400">Total Votes</span>
            </div>
          </div>
          <button
            onClick={loadStats}
            className="flex items-center gap-2 px-4 py-2 bg-sidebar-dark border border-white/10 rounded-lg hover:bg-white/5 transition font-medium"
          >
            <RefreshCw size={18} /> Refresh Data
          </button>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-sidebar-dark border border-gray-700 rounded-lg p-6 shadow-sm">
            <h3 className="text-gray-400 text-sm font-semibold uppercase tracking-wider mb-2">
              RAG Wins
            </h3>
            <div className="flex items-end gap-3">
              <span className="text-4xl font-bold" style={{ color: chartColors.rag }}>
                {stats.rag_wins}
              </span>
              <span className="text-lg text-gray-300 mb-1">
                ({stats.rag_win_rate}%)
              </span>
            </div>
          </div>
          <div className="bg-sidebar-dark border border-gray-700 rounded-lg p-6 shadow-sm">
            <h3 className="text-gray-400 text-sm font-semibold uppercase tracking-wider mb-2">
              Plain LLM Wins
            </h3>
            <div className="flex items-end gap-3">
              <span className="text-4xl font-bold" style={{ color: chartColors.plain }}>
                {stats.plain_wins}
              </span>
              <span className="text-lg text-gray-300 mb-1">
                ({stats.plain_win_rate}%)
              </span>
            </div>
          </div>
          <div className="bg-sidebar-dark border border-gray-700 rounded-lg p-6 shadow-sm">
            <h3 className="text-gray-400 text-sm font-semibold uppercase tracking-wider mb-2">
              Ties
            </h3>
            <div className="flex items-end gap-3">
              <span className="text-4xl font-bold" style={{ color: chartColors.tie }}>
                {stats.ties}
              </span>
              <span className="text-lg text-gray-300 mb-1">
                ({stats.tie_rate}%)
              </span>
            </div>
          </div>
        </div>

        {/* Statistical Analysis Card (T-Test) */}
        <div className="bg-sidebar-dark border border-gray-700 rounded-lg p-6 shadow-sm">
          <h2 className="text-xl font-serif font-bold mb-4 border-b border-white/10 pb-2">
            Statistical Analysis (1-Sample T-Test)
          </h2>
          {!stats.t_test ? (
            <p className="text-gray-400 italic">
              Need at least 3 non-tie/tie-inclusive votes for statistical analysis.
            </p>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <p className="text-sm text-gray-400 mb-1">p-value</p>
                <p
                  className={cn(
                    "text-2xl font-bold",
                    stats.t_test.significant ? "text-green-400" : "text-red-400"
                  )}
                >
                  {stats.t_test.p_value.toFixed(4)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400 mb-1">t-statistic</p>
                <p className="text-2xl font-bold text-parchment">
                  {stats.t_test.t_statistic.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400 mb-1">Effect Size (d)</p>
                <p className="text-2xl font-bold text-parchment">
                  {stats.t_test.cohens_d.toFixed(2)}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  {Math.abs(stats.t_test.cohens_d) < 0.2
                    ? "Small"
                    : Math.abs(stats.t_test.cohens_d) < 0.8
                    ? "Medium"
                    : "Large"}{" "}
                  effect
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400 mb-1">Conclusion</p>
                <p
                  className={cn(
                    "text-sm font-medium leading-tight",
                    stats.t_test.significant ? "text-green-400" : "text-gray-300"
                  )}
                >
                  {stats.t_test.interpretation}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Bar Chart */}
          <div className="bg-sidebar-dark border border-gray-700 rounded-lg p-6 shadow-sm flex flex-col">
            <div className="h-64 relative mb-4">
              <Bar ref={barChartRef} data={barData} options={barOptions} />
            </div>
            <button
              onClick={() => downloadChart(barChartRef, "arena-win-rate.png")}
              className="mt-auto self-end flex items-center gap-2 px-3 py-1.5 text-sm bg-white/5 hover:bg-white/10 border border-white/10 rounded transition"
            >
              <Download size={14} /> Download PNG
            </button>
          </div>

          {/* Pie Chart */}
          <div className="bg-sidebar-dark border border-gray-700 rounded-lg p-6 shadow-sm flex flex-col">
            <div className="h-64 relative mb-4">
              <Pie ref={pieChartRef} data={pieData} options={pieOptions} />
            </div>
            <button
              onClick={() => downloadChart(pieChartRef, "arena-vote-distribution.png")}
              className="mt-auto self-end flex items-center gap-2 px-3 py-1.5 text-sm bg-white/5 hover:bg-white/10 border border-white/10 rounded transition"
            >
              <Download size={14} /> Download PNG
            </button>
          </div>
        </div>

        {/* Per-Query Results Table */}
        <div className="bg-sidebar-dark border border-gray-700 rounded-lg shadow-sm overflow-hidden flex flex-col">
          <div className="p-6 border-b border-white/10">
            <h2 className="text-xl font-serif font-bold">Query Log</h2>
          </div>
          <div className="overflow-x-auto max-h-96 overflow-y-auto scrollbar-thin scrollbar-thumb-white/10">
            <table className="w-full text-left text-sm">
              <thead className="bg-black/20 sticky top-0 z-10 backdrop-blur-sm">
                <tr>
                  <th className="px-6 py-3 font-semibold text-gray-400">Time</th>
                  <th className="px-6 py-3 font-semibold text-gray-400 w-1/2">Query</th>
                  <th className="px-6 py-3 font-semibold text-gray-400">Model</th>
                  <th className="px-6 py-3 font-semibold text-gray-400">Winner</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {(!stats.query_results || stats.query_results.length === 0) ? (
                  <tr>
                    <td colSpan={4} className="px-6 py-8 text-center text-gray-500 italic">
                      No query data available.
                    </td>
                  </tr>
                ) : (
                  stats.query_results.map((result, idx) => (
                    <tr key={idx} className="hover:bg-white/5 transition-colors">
                      <td className="px-6 py-4 text-gray-400 whitespace-nowrap">
                        {new Date(result.timestamp).toLocaleString(undefined, {
                          month: "short",
                          day: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </td>
                      <td className="px-6 py-4 truncate max-w-xs" title={result.query}>
                        {result.query.length > 40
                          ? result.query.substring(0, 40) + "..."
                          : result.query}
                      </td>
                      <td className="px-6 py-4 text-gray-300 whitespace-nowrap">
                        {result.model}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className={cn(
                            "px-2 py-1 rounded text-xs font-semibold uppercase tracking-wide",
                            result.winner === "rag"
                              ? "bg-[#19e6d4]/10 text-[#19e6d4]"
                              : result.winner === "plain"
                              ? "bg-amber-400/10 text-amber-400"
                              : "bg-gray-500/10 text-gray-300"
                          )}
                        >
                          {result.winner}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
