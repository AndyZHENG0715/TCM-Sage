"use client";

import {
    Background,
    Controls,
    Edge,
    MarkerType,
    Node,
    Position,
    ReactFlow,
    ReactFlowProvider,
    useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEffect, useMemo } from "react";
import type { SubgraphResponse } from "@/lib/api";
import { GraphCitation } from "@/lib/types";

interface KGViewerProps {
    citation: GraphCitation;
    subgraph?: SubgraphResponse | null;
}

function Flow({ nodes, edges }: { nodes: Node[]; edges: Edge[] }) {
    const { fitView } = useReactFlow();

    useEffect(() => {
        const timeout = setTimeout(() => {
            fitView({ padding: 0.25, duration: 300 });
        }, 40);
        return () => clearTimeout(timeout);
    }, [nodes, edges, fitView]);

    return (
        <ReactFlow
            nodes={nodes}
            edges={edges}
            fitView
            fitViewOptions={{ padding: 0.25 }}
            proOptions={{ hideAttribution: true }}
            nodesDraggable
            nodesConnectable={false}
            zoomOnScroll
            panOnDrag
        >
            <Background className="text-[#dcd3b8]" gap={16} size={1} />
            <Controls
                showInteractive={false}
                className="opacity-60 hover:opacity-100 transition-opacity [&_.react-flow__controls-button]:border-[#dcd3b8] [&_.react-flow__controls-button]:bg-[#ebe5d5] [&_.react-flow__controls-button]:text-[#5c5548] [&_.react-flow__controls-button:hover]:bg-[#dcd3b8]"
            />
        </ReactFlow>
    );
}

export function KGViewer({ citation, subgraph = null }: KGViewerProps) {
    const primaryColor = "#8c8578";

    const hasSubgraph = Boolean(subgraph && subgraph.nodes.length > 0);
    const citedIds = useMemo(() => new Set(subgraph?.cited_ids ?? []), [subgraph]);

    const { nodes, edges } = useMemo(() => {
        if (hasSubgraph && subgraph) {
            const baseNodes: Node[] = subgraph.nodes.map((node) => {
                const isCited = citedIds.has(node.id);
                return {
                    id: node.id,
                    position: { x: 0, y: 0 },
                    sourcePosition: Position.Right,
                    targetPosition: Position.Left,
                    data: { label: node.label },
                    style: {
                        background: isCited ? "#f59e0b" : "#3e382d",
                        color: isCited ? "var(--color-primary)" : "var(--color-parchment)",
                        border: "1px solid #8c8578",
                        borderRadius: "8px",
                        padding: "10px 15px",
                        fontWeight: isCited ? 700 : 500,
                        fontFamily: "var(--font-noto-serif-sc)",
                        fontSize: "13px",
                        boxShadow: isCited
                            ? "0 0 0 2px rgba(245, 158, 11, 0.28), 0 8px 16px rgba(245, 158, 11, 0.25)"
                            : "0 4px 6px rgba(0, 0, 0, 0.12)",
                    },
                } as Node;
            });

            const baseEdges: Edge[] = subgraph.edges.map((edge, index) => ({
                id: `${edge.source}-${edge.target}-${index}`,
                source: edge.source,
                target: edge.target,
                label: edge.label,
                animated: false,
                style: { stroke: primaryColor, strokeWidth: 1.8 },
                labelStyle: {
                    fill: "#5c5548",
                    fontWeight: 600,
                    fontFamily: "var(--font-inter)",
                    fontSize: 10,
                },
                labelBgStyle: { fill: "#ebe5d5", fillOpacity: 0.9, rx: 4, ry: 4 },
                markerEnd: {
                    type: MarkerType.ArrowClosed,
                    color: primaryColor,
                    width: 18,
                    height: 18,
                },
            }));

            // Radial/star layout: center cited node, fan others around it
            const centerX = 300;
            const centerY = 200;
            const radius = 160;
            const nonCited = baseNodes.filter(n => !citedIds.has(n.id));
            const cited = baseNodes.filter(n => citedIds.has(n.id));
            const angleStep = nonCited.length > 0 ? (2 * Math.PI) / nonCited.length : 0;

            cited.forEach(n => {
                n.position = { x: centerX, y: centerY };
            });

            nonCited.forEach((n, i) => {
                n.position = {
                    x: centerX + radius * Math.cos(angleStep * i - Math.PI / 2),
                    y: centerY + radius * Math.sin(angleStep * i - Math.PI / 2),
                };
            });

            return { nodes: [...cited, ...nonCited], edges: baseEdges };
        }

        const match = citation.fact.match(/^(.+?)\s*--(.+?)-->\s*(.+)$/);
        const parsed = match
            ? {
                source: match[1].trim(),
                relationship: match[2].trim(),
                target: match[3].trim(),
            }
            : {
                source: "Fact",
                relationship: "IS",
                target: citation.fact,
            };

        const fallbackNodes: Node[] = [
            {
                id: "source",
                position: { x: 0, y: 50 },
                data: { label: parsed.source },
                sourcePosition: Position.Right,
                targetPosition: Position.Left,
                style: {
                    background: "#3e382d",
                    color: "var(--color-parchment)",
                    border: "1px solid #8c8578",
                    borderRadius: "8px",
                    padding: "10px 15px",
                    fontWeight: 700,
                    fontFamily: "var(--font-noto-serif-sc)",
                    fontSize: "14px",
                    boxShadow: "0 4px 6px rgba(0, 0, 0, 0.12)",
                },
            },
            {
                id: "target",
                position: { x: 250, y: 50 },
                data: { label: parsed.target },
                sourcePosition: Position.Right,
                targetPosition: Position.Left,
                style: {
                    background: "#3e382d",
                    color: "var(--color-parchment)",
                    border: "1px solid #8c8578",
                    borderRadius: "8px",
                    padding: "10px 15px",
                    fontWeight: 700,
                    fontFamily: "var(--font-noto-serif-sc)",
                    fontSize: "14px",
                    boxShadow: "0 4px 6px rgba(0, 0, 0, 0.12)",
                },
            },
        ];

        const fallbackEdges: Edge[] = [
            {
                id: "edge1",
                source: "source",
                target: "target",
                label: parsed.relationship,
                animated: true,
                style: { stroke: primaryColor, strokeWidth: 2 },
                labelStyle: {
                    fill: "#5c5548",
                    fontWeight: 600,
                    fontFamily: "var(--font-inter)",
                    fontSize: 10,
                },
                labelBgStyle: { fill: "#ebe5d5", fillOpacity: 0.9, rx: 4, ry: 4 },
                markerEnd: {
                    type: MarkerType.ArrowClosed,
                    color: primaryColor,
                    width: 20,
                    height: 20,
                },
            },
        ];

        return { nodes: fallbackNodes, edges: fallbackEdges };
    }, [citation.fact, citedIds, hasSubgraph, primaryColor, subgraph]);

    const graphHeight = useMemo(() => {
        const nodeCount = hasSubgraph && subgraph ? subgraph.nodes.length : 2;
        return Math.min(500, Math.max(200, 140 + nodeCount * 3));
    }, [hasSubgraph, subgraph]);

    return (
        <div
            className="w-full border border-[#dcd3b8] rounded-xl overflow-hidden bg-[#f4ecd8] shadow-inner"
            style={{ height: `${graphHeight}px` }}
        >
            <ReactFlowProvider>
                <Flow nodes={nodes} edges={edges} />
            </ReactFlowProvider>
        </div>
    );
}
