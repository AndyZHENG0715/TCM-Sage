"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { Bar, Pie } from "react-chartjs-2";
import {
  ArcElement,
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Title,
  Tooltip,
} from "chart.js";
import ChartDataLabels, { type Context as ChartDataLabelsContext } from "chartjs-plugin-datalabels";
import { ArrowLeft, Download, RefreshCw } from "lucide-react";
import { useI18n } from "@/i18n/context";
import { fetchArenaStats } from "@/lib/api";
import { cn } from "@/lib/utils";

ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Title, Tooltip, Legend, ChartDataLabels);

// Print mode: ?print=true switches chart colors for white-background export
const isPrintMode = typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('print') === 'true';

// White background plugin for print mode chart downloads
if (isPrintMode) {
  ChartJS.register({
    id: 'printBg',
    beforeDraw: (chart) => {
      const ctx = chart.canvas.getContext('2d');
      if (ctx) {
        ctx.save();
        ctx.globalCompositeOperation = 'destination-over';
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, chart.width, chart.height);
        ctx.restore();
      }
    },
  });
}

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
  const { t } = useI18n();
  const [stats, setStats] = useState<ArenaStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const barChartRef = useRef<ChartJS<"bar">>(null);
  const pieChartRef = useRef<ChartJS<"pie">>(null);

  const loadStats = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchArenaStats();
      setStats(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : t.arenaStats.noData);
    } finally {
      setLoading(false);
    }
  }, [t.arenaStats.noData]);

  useEffect(() => {
    void loadStats();
  }, [loadStats]);

  const downloadChart = (chartRef: React.RefObject<ChartJS<"bar"> | ChartJS<"pie"> | null>, filename: string) => {
    if (!chartRef.current) return;
    const link = document.createElement("a");
    link.download = filename;
    link.href = chartRef.current.toBase64Image();
    link.click();
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background-dark text-parchment">
        <RefreshCw className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-3 font-medium">{t.common.loading}</span>
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background-dark text-parchment">
        <div className="text-center">
          <p className="mb-4 text-red-400">{error || t.arenaStats.noData}</p>
          <button
            type="button"
            onClick={() => void loadStats()}
            className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-4 py-2 transition hover:bg-white/10"
          >
            <RefreshCw size={16} /> {t.common.retry}
          </button>
        </div>
      </div>
    );
  }

  const textColor = isPrintMode ? '#1a1a1a' : '#F3EFE0';
  const axisColor = isPrintMode ? '#444444' : '#9ca3af';
  const gridColor = isPrintMode ? 'rgba(0, 0, 0, 0.1)' : 'rgba(255, 255, 255, 0.1)';
  const borderCol = isPrintMode ? '#ffffff' : '#112120';
  const chartColors = { rag: '#19e6d4', plain: '#fbbf24', tie: '#6b7280' };

  const barData = {
    labels: [t.arenaStats.ragWins, t.arenaStats.llmWins, t.arenaStats.ties],
    datasets: [
      {
        label: t.arenaStats.winRate,
        data: [stats.rag_win_rate, stats.plain_win_rate, stats.tie_rate],
        backgroundColor: [chartColors.rag, chartColors.plain, chartColors.tie],
        borderWidth: 1,
        borderColor: borderCol,
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
        text: t.arenaStats.winRate,
        color: textColor,
        font: { size: 16, family: "serif" },
      },
      datalabels: {
        anchor: "end",
        align: "top",
        color: textColor,
        font: { weight: "bold", size: 14 },
        formatter: (value: number) => value.toFixed(1) + "%",
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        title: { display: true, text: "Percentage (%)", color: axisColor },
        ticks: { color: axisColor },
        grid: { color: gridColor },
      },
      x: {
        ticks: { color: textColor },
        grid: { display: false },
      },
    },
  };

  const pieData = {
    labels: [t.arenaStats.ragWins, t.arenaStats.llmWins, t.arenaStats.ties],
    datasets: [
      {
        data: [stats.rag_wins, stats.plain_wins, stats.ties],
        backgroundColor: [chartColors.rag, chartColors.plain, chartColors.tie],
        borderColor: borderCol,
        borderWidth: 2,
      },
    ],
  };

  const pieOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'bottom' as const, labels: { color: textColor } },
      title: {
        display: true,
        text: t.arenaStats.voteDistribution,
        color: textColor,
        font: { size: 16, family: "serif" },
      },
      datalabels: {
        color: textColor,
        font: { weight: "bold", size: 13 },
        formatter: (value: number, ctx: ChartDataLabelsContext) => {
          const data = ctx.dataset.data as number[];
          const total = data.reduce((a: number, b: number) => a + b, 0);
          const pct = ((value / total) * 100).toFixed(1);
          return value + " (" + pct + "%)";
        },
      },
    },
  };

  return (
    <div className="min-h-screen bg-background-dark p-8 font-sans text-parchment">
      <div className="mx-auto max-w-6xl space-y-8">
        <div className="flex flex-col items-start justify-between gap-4 md:flex-row md:items-center">
          <div className="space-y-2">
            <Link href="/" className="group inline-flex items-center text-sm text-primary hover:underline">
              <ArrowLeft size={16} className="mr-1 transition-transform group-hover:-translate-x-1" />
              {t.common.backToApp}
            </Link>
            <h1 className="text-3xl font-serif font-bold tracking-tight">{t.arenaStats.title}</h1>
          <div className="inline-flex items-center rounded-full border border-white/10 bg-white/5 px-3 py-1 text-sm">
              <span className="font-semibold">{stats.total_votes}</span>
              <span className="ml-1 text-gray-400">{t.arenaStats.totalVotes}</span>
            </div>
          </div>
          <button
            type="button"
            onClick={() => void loadStats()}
            className="flex items-center gap-2 rounded-lg border border-white/10 bg-sidebar-dark px-4 py-2 font-medium transition hover:bg-white/5"
          >
            <RefreshCw size={18} /> {t.common.retry}
          </button>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          {[
            { label: t.arenaStats.ragWins, value: stats.rag_wins, rate: stats.rag_win_rate, color: chartColors.rag },
            { label: t.arenaStats.llmWins, value: stats.plain_wins, rate: stats.plain_win_rate, color: chartColors.plain },
            { label: t.arenaStats.ties, value: stats.ties, rate: stats.tie_rate, color: chartColors.tie },
          ].map((item) => (
            <div key={item.label} className="rounded-lg border border-gray-700 bg-sidebar-dark p-6 shadow-sm">
              <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-gray-400">{item.label}</h3>
              <div className="flex items-end gap-3">
                <span className="text-4xl font-bold" style={{ color: item.color }}>
                  {item.value}
                </span>
                <span className="mb-1 text-lg text-gray-300">({item.rate}%)</span>
              </div>
            </div>
          ))}
        </div>

        <div className="rounded-lg border border-gray-700 bg-sidebar-dark p-6 shadow-sm">
          <h2 className="mb-4 border-b border-white/10 pb-2 text-xl font-serif font-bold">{t.arenaStats.statsTitle}</h2>
          {!stats.t_test ? (
            <p className="italic text-gray-400">{t.arenaStats.statsInsufficient}</p>
          ) : (
            <div className="grid grid-cols-2 gap-6 md:grid-cols-4">
              <div>
                <p className="mb-1 text-sm text-gray-400">{t.arenaStats.pValue}</p>
                <p className={cn("text-2xl font-bold", stats.t_test.significant ? "text-green-400" : "text-red-400")}>
                  {stats.t_test.p_value.toFixed(4)}
                </p>
              </div>
              <div>
                <p className="mb-1 text-sm text-gray-400">{t.arenaStats.tStatistic}</p>
                <p className="text-2xl font-bold text-parchment">{stats.t_test.t_statistic.toFixed(2)}</p>
              </div>
              <div>
                <p className="mb-1 text-sm text-gray-400">{t.arenaStats.effectSize}</p>
                <p className="text-2xl font-bold text-parchment">{stats.t_test.cohens_d.toFixed(2)}</p>
                <p className="mt-1 text-xs text-gray-500">
                  {Math.abs(stats.t_test.cohens_d) < 0.2
                    ? t.arenaStats.effectSmall
                    : Math.abs(stats.t_test.cohens_d) < 0.8
                      ? t.arenaStats.effectMedium
                      : t.arenaStats.effectLarge}{" "}
                  {t.arenaStats.effect}
                </p>
              </div>
              <div>
                <p className="mb-1 text-sm text-gray-400">{t.arenaStats.conclusion}</p>
                <p className={cn("text-sm font-medium leading-tight", stats.t_test.significant ? "text-green-400" : "text-gray-300")}>
                  {stats.t_test.interpretation}
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
          <div className="flex flex-col rounded-lg border border-gray-700 bg-sidebar-dark p-6 shadow-sm">
            <div className="relative mb-4 h-64">
              <Bar ref={barChartRef} data={barData} options={barOptions} />
            </div>
            <button
              type="button"
              onClick={() => downloadChart(barChartRef, "arena-win-rate.png")}
              className="mt-auto flex self-end rounded border border-white/10 bg-white/5 px-3 py-1.5 text-sm transition hover:bg-white/10"
            >
              <Download size={14} className="mr-2" /> {t.arenaStats.downloadPng}
            </button>
          </div>

          <div className="flex flex-col rounded-lg border border-gray-700 bg-sidebar-dark p-6 shadow-sm">
            <div className="relative mb-4 h-64">
              <Pie ref={pieChartRef} data={pieData} options={pieOptions} />
            </div>
            <button
              type="button"
              onClick={() => downloadChart(pieChartRef, "arena-vote-distribution.png")}
              className="mt-auto flex self-end rounded border border-white/10 bg-white/5 px-3 py-1.5 text-sm transition hover:bg-white/10"
            >
              <Download size={14} className="mr-2" /> {t.arenaStats.downloadPng}
            </button>
          </div>
        </div>

        <div className="flex flex-col overflow-hidden rounded-lg border border-gray-700 bg-sidebar-dark shadow-sm">
          <div className="border-b border-white/10 p-6">
            <h2 className="text-xl font-serif font-bold">{t.arenaStats.queryLog}</h2>
          </div>
          <div className="max-h-96 overflow-x-auto overflow-y-auto scrollbar-thin scrollbar-thumb-white/10">
            <table className="w-full text-left text-sm">
              <thead className="sticky top-0 z-10 bg-black/20 backdrop-blur-sm">
                <tr>
                  <th className="px-6 py-3 font-semibold text-gray-400">{t.arenaStats.colTime}</th>
                  <th className="w-1/2 px-6 py-3 font-semibold text-gray-400">{t.arenaStats.colQuery}</th>
                  <th className="px-6 py-3 font-semibold text-gray-400">{t.arenaStats.colModel}</th>
                  <th className="px-6 py-3 font-semibold text-gray-400">{t.arenaStats.colWinner}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {!stats.query_results || stats.query_results.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="px-6 py-8 text-center italic text-gray-500">
                      {t.arenaStats.noQueryData}
                    </td>
                  </tr>
                ) : (
                  stats.query_results.map((result) => (
                    <tr key={`${result.session_id}-${result.timestamp}-${result.query}`} className="transition-colors hover:bg-white/5">
                      <td className="whitespace-nowrap px-6 py-4 text-gray-400">
                        {new Date(result.timestamp).toLocaleString(undefined, {
                          month: "short",
                          day: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </td>
                      <td className="max-w-xs truncate px-6 py-4" title={result.query}>
                        {result.query.length > 40 ? `${result.query.substring(0, 40)}...` : result.query}
                      </td>
                      <td className="whitespace-nowrap px-6 py-4 text-gray-300">{result.model}</td>
                      <td className="whitespace-nowrap px-6 py-4">
                        <span
                          className={cn(
                            "rounded px-2 py-1 text-xs font-semibold uppercase tracking-wide",
                            result.winner === "rag"
                              ? "bg-[#19e6d4]/10 text-[#19e6d4]"
                              : result.winner === "plain"
                                ? "bg-amber-400/10 text-amber-400"
                                : "bg-gray-500/10 text-gray-300"
                          )}
                        >
                          {result.winner === "rag"
                            ? t.arenaStats.ragWins
                            : result.winner === "plain"
                              ? t.arenaStats.llmWins
                              : t.arenaStats.ties}
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
