import { createElement, type ComponentPropsWithoutRef } from "react";

import type { Citation } from "@/lib/types";

export const CITE_PREFIX = "%%CITE_";
export const CITE_SUFFIX = "%%";

export function stripTrailingReferenceSection(content: string): string {
    const normalized = content.replace(/\r\n/g, "\n").trimEnd();
    const lines = normalized.split("\n");
    if (lines.length < 2) {
        return normalized;
    }

    const headingPattern =
        /^\s*(?:\*\*|__)?\s*(?:#{1,3}\s*)?(?:sources?|references?|citations?|bibliography|参考资料|参考文献|资料来源|引用来源|引用文献|参考来源|出处|参考)\s*[：:]*\s*(?:\*\*|__)?\s*$/i;
    const itemPattern =
        /^\s*(?:[-*•]|\d+[.)]|(?:\[\d+\]))\s+\S+/;

    for (let i = lines.length - 1; i >= 0; i -= 1) {
        if (!headingPattern.test(lines[i])) {
            continue;
        }

        let hasReferenceItems = false;
        for (let j = i + 1; j < lines.length; j += 1) {
            const candidate = lines[j].trim();
            if (!candidate) {
                continue;
            }
            if (itemPattern.test(candidate)) {
                hasReferenceItems = true;
                continue;
            }
            hasReferenceItems = false;
            break;
        }

        if (hasReferenceItems) {
            return lines.slice(0, i).join("\n").trimEnd();
        }
    }

    return normalized;
}

export function postProcessAssistantContent(content: string): string {
    // Fix malformed markdown: missing space after formatters
    const fixed = content
        // ## Title, ### Title etc (also handles Chinese characters after #)
        .replace(/(^|\n)(#{1,6})([^\s#])/gm, "$1$2 $3")
        // 1.Item → 1. Item (numbered lists)
        .replace(/(^|\n)(\d+)\.([^\s.])/gm, "$1$2. $3")
        // -Item → - Item (unordered lists)
        .replace(/(^|\n)-([^\s-])/gm, "$1- $2")
        // *Item → * Item (alternative unordered lists, but not **bold**)
        .replace(/(^|\n)\*([^\s*])/gm, "$1* $2");

    const withoutTrailingReferences = stripTrailingReferenceSection(fixed);
    const withCiteMarkers = withoutTrailingReferences.replace(
        /\[(\d+)\]/g,
        (_match, num) => `\`${CITE_PREFIX}${num}${CITE_SUFFIX}\``
    );
    return withCiteMarkers.replace(/([^\n])\n(?=(?:\d+\.\s|- ))/g, "$1\n\n");
}

export function createMarkdownComponents(
    citations: Citation[] | undefined,
    onCitationClick?: (citation: Citation) => void,
) {
    return {
        code({ children, ...props }: ComponentPropsWithoutRef<"code">) {
            const text = String(children).trim();
            if (text.startsWith(CITE_PREFIX) && text.endsWith(CITE_SUFFIX)) {
                const citationNumber = Number.parseInt(
                    text.slice(CITE_PREFIX.length, -CITE_SUFFIX.length),
                    10
                );
                const citation = citations?.find(
                    (item) => item.number === citationNumber
                );

                if (citation) {
                    return createElement(
                        "button",
                        {
                            onClick: () => onCitationClick?.(citation),
                            className: "inline-flex items-center justify-center mx-1 px-1.5 py-0.5 text-xs font-sans font-bold text-primary bg-primary/10 rounded-full hover:bg-primary/20 transition-colors cursor-pointer align-super",
                        },
                        citationNumber
                    );
                }

                return createElement("span", null, `[${citationNumber}]`);
            }

            return createElement(
                "code",
                {
                    className: "bg-black/5 px-1.5 py-0.5 rounded text-sm font-mono",
                    ...props,
                },
                children
            );
        },
        p({ children }: ComponentPropsWithoutRef<"p">) {
            return createElement("p", { className: "mb-3 last:mb-0 leading-relaxed" }, children);
        },
        h1({ children }: ComponentPropsWithoutRef<"h1">) {
            return createElement("h1", { className: "text-2xl font-bold mb-3 mt-4 first:mt-0" }, children);
        },
        h2({ children }: ComponentPropsWithoutRef<"h2">) {
            return createElement("h2", { className: "text-xl font-bold mb-2 mt-3 first:mt-0" }, children);
        },
        h3({ children }: ComponentPropsWithoutRef<"h3">) {
            return createElement("h3", { className: "text-lg font-semibold mb-2 mt-3 first:mt-0" }, children);
        },
        ul({ children }: ComponentPropsWithoutRef<"ul">) {
            return createElement("ul", { className: "flex flex-col gap-2 my-4" }, children);
        },
        ol({ children }: ComponentPropsWithoutRef<"ol">) {
            return createElement("ol", { className: "flex flex-col gap-2 my-4" }, children);
        },
        li({ children }: ComponentPropsWithoutRef<"li">) {
            return createElement("li", { className: "list-item ml-4" }, children);
        },
        blockquote({ children }: ComponentPropsWithoutRef<"blockquote">) {
            return createElement(
                "blockquote",
                { className: "border-l-4 border-primary/40 pl-4 italic my-3 text-parchment-text/80" },
                children
            );
        },
        strong({ children }: ComponentPropsWithoutRef<"strong">) {
            return createElement("strong", { className: "font-bold" }, children);
        },
        hr() {
            return createElement("hr", { className: "my-4 border-[#dcd3b8]" });
        },
        table({ children }: ComponentPropsWithoutRef<"table">) {
            return createElement("table", { className: "w-full border-collapse my-4 text-sm" }, children);
        },
        thead({ children }: ComponentPropsWithoutRef<"thead">) {
            return createElement("thead", { className: "bg-black/5 font-semibold" }, children);
        },
        th({ children }: ComponentPropsWithoutRef<"th">) {
            return createElement("th", { className: "border border-[#dcd3b8] px-3 py-1.5 text-left" }, children);
        },
        td({ children }: ComponentPropsWithoutRef<"td">) {
            return createElement("td", { className: "border border-[#dcd3b8] px-3 py-1.5" }, children);
        },
    };
}
