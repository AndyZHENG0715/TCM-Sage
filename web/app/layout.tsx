import type { Metadata } from "next";
import "./globals.css";
import { NoticeBanner } from "@/components/NoticeBanner";
import { I18nProvider } from "@/i18n/context";

export const metadata: Metadata = {
  title: "TCM-Sage — classical texts, with sources",
  description: "A research prototype for exploring classical Chinese medicine texts, citations and recorded knowledge-graph relationships. Not medical advice.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        className="antialiased bg-background-dark text-parchment"
      >
        <I18nProvider>
          <NoticeBanner />
          {children}
        </I18nProvider>
      </body>
    </html>
  );
}
