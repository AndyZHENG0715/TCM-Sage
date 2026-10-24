"use client";

import { createContext, useContext, useSyncExternalStore, useCallback, type ReactNode } from "react";
import zh from "./zh.json";
import zhHant from "./zh-Hant.json";
import en from "./en.json";

type Locale = "zh" | "zh-Hant" | "en";
type Translations = typeof zh;

const translations: Record<Locale, Translations> = { zh, "zh-Hant": zhHant, en };
const nextLocale: Record<Locale, Locale> = { zh: "zh-Hant", "zh-Hant": "en", en: "zh" };

interface I18nContextValue {
  locale: Locale;
  t: Translations;
  setLocale: (locale: Locale) => void;
  toggleLocale: () => void;
}

const I18nContext = createContext<I18nContextValue | null>(null);

const STORAGE_KEY = "tcm-sage-locale";

function getInitialLocale(): Locale {
  if (typeof window === "undefined") return "en";
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "en" || stored === "zh" || stored === "zh-Hant") return stored;
  return "en";
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const locale = useSyncExternalStore(
    (notify) => {
      window.addEventListener("tcm-sage-locale-change", notify);
      window.addEventListener("storage", notify);
      return () => {
        window.removeEventListener("tcm-sage-locale-change", notify);
        window.removeEventListener("storage", notify);
      };
    },
    getInitialLocale,
    () => "en" as Locale,
  );

  const setLocale = useCallback((newLocale: Locale) => {
    if (typeof window !== "undefined") {
      localStorage.setItem(STORAGE_KEY, newLocale);
      document.documentElement.lang = newLocale === "zh-Hant" ? "zh-Hant" : newLocale;
      window.dispatchEvent(new Event("tcm-sage-locale-change"));
    }
  }, []);

  const toggleLocale = useCallback(() => {
    setLocale(nextLocale[locale]);
  }, [locale, setLocale]);

  return (
    <I18nContext.Provider value={{ locale, t: translations[locale], setLocale, toggleLocale }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n must be used within I18nProvider");
  return ctx;
}
