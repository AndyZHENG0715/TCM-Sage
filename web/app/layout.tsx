import type { Metadata } from "next";
import { Inter, Noto_Serif_SC } from "next/font/google";
import "./globals.css";
import { NoticeBanner } from "@/components/NoticeBanner";
import { WelcomeModal } from "@/components/WelcomeModal";

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
  title: "TCM-Sage",
  description: "Traditional Chinese Medicine Research Assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        className={`${inter.variable} ${notoSerifSC.variable} antialiased bg-background-dark text-parchment`}
      >
        <NoticeBanner />
        <WelcomeModal />
        {children}
        {children}
      </body>
    </html>
  );
}
