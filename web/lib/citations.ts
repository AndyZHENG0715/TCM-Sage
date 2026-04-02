const LOW_QUALITY_SOURCE_RE =
    /^卷[一二三四五六七八九十百千万0-9]+(?:第[一二三四五六七八九十百千万0-9]+)?(?:上编|中编|下编)?$/;
const KNOWN_OCR_ARTIFACT_RE = /\bKT\b/g;

export function cleanSourceLabel(source: string | null | undefined): string {
    if (!source) {
        return "";
    }

    return source
        .replace(/<[^>]+>/g, "")
        .replace(/[。、:：;；)\]）】」』]+$/g, "")
        .trim();
}

export function isLowQualitySourceLabel(source: string | null | undefined): boolean {
    const cleaned = cleanSourceLabel(source).replace(/\s+/g, "");
    return cleaned.length > 0 && LOW_QUALITY_SOURCE_RE.test(cleaned);
}

export function getDisplaySourceLabel(
    source: string | null | undefined,
    fallback?: string | null
): string {
    const cleanedFallback = cleanSourceLabel(fallback);
    if (cleanedFallback) {
        return cleanedFallback;
    }

    const cleanedSource = cleanSourceLabel(source);
    return isLowQualitySourceLabel(cleanedSource) ? "" : cleanedSource;
}

export function getOcrArtifacts(text: string | null | undefined): string[] {
    if (!text) {
        return [];
    }

    const matches = text.match(KNOWN_OCR_ARTIFACT_RE);
    return matches ? Array.from(new Set(matches)) : [];
}
