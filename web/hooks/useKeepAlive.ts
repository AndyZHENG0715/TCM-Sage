"use client";

import { useEffect } from "react";
import { healthCheck } from "@/lib/api";

const PING_INTERVAL_MS = 10 * 60 * 1000; // 10 minutes

export function useKeepAlive() {
    useEffect(() => {
        // Initial ping
        healthCheck();

        const interval = setInterval(() => {
            healthCheck();
        }, PING_INTERVAL_MS);

        return () => clearInterval(interval);
    }, []);
}
