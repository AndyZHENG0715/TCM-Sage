import type { Metadata } from "next";
import { Inter, Noto_Serif_SC } from "next/font/google";
import Link from "next/link";
import "./globals.css";

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
        <div className="fixed top-2 right-4 z-40">
          <Link
            href="/arena"
            className="text-xs bg-gray-800 border border-gray-700 px-3 py-1 rounded-full text-gray-400 hover:text-[#19e6d4] hover:border-[#19e6d4] transition-colors"
          >
            Arena 评测
          </Link>
        </div>
        {children}
      </body>
    </html>
  );
}
