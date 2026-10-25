"use client";

import { FullSystem } from "@/components/FullSystem";
import { PublicShowcase } from "@/components/PublicShowcase";
import { PRESENTATION_MODE } from "@/lib/showcase";

export default function Home() {
    return PRESENTATION_MODE ? <FullSystem /> : <PublicShowcase />;
}
