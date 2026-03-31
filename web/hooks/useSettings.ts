"use client";

import { useEffect, useState } from "react";
import {
    DEFAULT_SETTINGS,
    DEFAULT_SETTINGS_CAPABILITIES,
    Settings,
    SettingsCapabilities,
} from "@/lib/types";
import { fetchConfig } from "@/lib/api";

const STORAGE_KEY = "tcm-sage-settings";

function normalizeStoredSettings(
    settings: Settings,
    capabilities: SettingsCapabilities
): Settings {
    if (!capabilities.hybridAvailable) {
        return {
            ...settings,
            hybridRetrieval: false,
        };
    }

    return settings;
}

export function useSettings() {
    const [defaultSettings, setDefaultSettings] = useState<Settings>(DEFAULT_SETTINGS);
    const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
    const [capabilities, setCapabilities] = useState<SettingsCapabilities>(
        DEFAULT_SETTINGS_CAPABILITIES
    );
    const [isLoaded, setIsLoaded] = useState(false);

    useEffect(() => {
        let isCancelled = false;

        const loadSettings = async () => {
            try {
                const config = await fetchConfig();
                const nextCapabilities = config?.capabilities ?? DEFAULT_SETTINGS_CAPABILITIES;
                const nextDefaults = normalizeStoredSettings(
                    config?.defaultSettings ?? DEFAULT_SETTINGS,
                    nextCapabilities
                );

                let storedOverrides: Partial<Settings> = {};
                const stored = localStorage.getItem(STORAGE_KEY);
                if (stored) {
                    try {
                        storedOverrides = JSON.parse(stored) as Partial<Settings>;
                    } catch (error) {
                        console.error("Failed to parse settings", error);
                    }
                }

                const mergedSettings = normalizeStoredSettings(
                    { ...nextDefaults, ...storedOverrides },
                    nextCapabilities
                );

                if (!isCancelled) {
                    setCapabilities(nextCapabilities);
                    setDefaultSettings(nextDefaults);
                    setSettings(mergedSettings);
                }
            } catch (error) {
                console.error("Critical error loading settings; using defaults:", error);
                if (!isCancelled) {
                    setCapabilities(DEFAULT_SETTINGS_CAPABILITIES);
                    setDefaultSettings(DEFAULT_SETTINGS);
                    setSettings(DEFAULT_SETTINGS);
                }
            } finally {
                if (!isCancelled) {
                    setIsLoaded(true);
                }
            }
        };

        void loadSettings();

        return () => {
            isCancelled = true;
        };
    }, []);

    const updateSettings = (newSettings: Partial<Settings>) => {
        setSettings((prev) => {
            const updated = normalizeStoredSettings(
                { ...prev, ...newSettings },
                capabilities
            );
            localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
            return updated;
        });
    };

    const resetDefaults = () => {
        setSettings(defaultSettings);
        localStorage.removeItem(STORAGE_KEY);
    };

    return {
        defaultSettings,
        settings,
        capabilities,
        updateSettings,
        resetDefaults,
        isLoaded,
    };
}
