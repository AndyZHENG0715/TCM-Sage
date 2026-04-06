import type { Metadata } from "next";
import { Inter, Noto_Serif_SC } from "next/font/google";
import "./globals.css";
import { NoticeBanner } from "@/components/NoticeBanner";
import { WelcomeModal } from "@/components/WelcomeModal";
import { I18nProvider } from "@/i18n/context";
import zh from "@/i18n/zh.json";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const notoSerifSC = Noto_Serif_SC({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-noto-serif-sc",
});

export const metadata: Metadata = {
  title: zh.common.appName,
  description: zh.common.appDescription,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh" className="dark" suppressHydrationWarning>
      <body
        className={`${inter.variable} ${notoSerifSC.variable} antialiased bg-background-dark text-parchment`}
      >
        <I18nProvider>
          <NoticeBanner />
          <WelcomeModal />
          {children}
        </I18nProvider>
      </body>
    </html>
  );
}
