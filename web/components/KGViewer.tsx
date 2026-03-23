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
import { useEffect } from "react";
import { GraphCitation } from "@/lib/types";

interface KGViewerProps {
    citation: GraphCitation;
}

function Flow({ nodes, edges }: { nodes: Node[]; edges: Edge[] }) {
    const { fitView } = useReactFlow();

    // Force fitView when nodes/edges change to ensure visibility in the sliding panel
    useEffect(() => {
        const timeout = setTimeout(() => {
            fitView({ padding: 0.3, duration: 400 });
        }, 50);
        return () => clearTimeout(timeout);
    }, [nodes, edges, fitView]);

    return (
        <ReactFlow
            nodes={nodes}
            edges={edges}
            fitView
            fitViewOptions={{ padding: 0.3 }}
            proOptions={{ hideAttribution: true }}
            nodesDraggable={false}
            nodesConnectable={false}
            zoomOnScroll={false}
            panOnDrag
        >
            <Background color="#dcd3b8" gap={16} size={1} />
            <Controls
                showInteractive={false}
                className="opacity-50 hover:opacity-100 transition-opacity [&_.react-flow__controls-button]:border-[#dcd3b8] [&_.react-flow__controls-button]:bg-[#ebe5d5] [&_.react-flow__controls-button]:text-[#5c5548] [&_.react-flow__controls-button:hover]:bg-[#dcd3b8]"
            />
        </ReactFlow>
    );
}

export function KGViewer({ citation }: KGViewerProps) {
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

    const primaryColor = "#8c8578";
    const nodeColor = "#3e382d";

    const nodes: Node[] = [
        {
            id: "source",
            position: { x: 0, y: 50 },
            data: { label: parsed.source },
            sourcePosition: Position.Right,
            targetPosition: Position.Left,
            style: {
                background: nodeColor,
                color: "#f4ecd8",
                border: `1px solid ${primaryColor}`,
                borderRadius: "8px",
                padding: "10px 15px",
                fontWeight: "bold",
                fontFamily: "serif",
                fontSize: "14px",
                boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
            },
        },
        {
            id: "target",
            position: { x: 250, y: 50 },
            data: { label: parsed.target },
            sourcePosition: Position.Right,
            targetPosition: Position.Left,
            style: {
                background: nodeColor,
                color: "#f4ecd8",
                border: `1px solid ${primaryColor}`,
                borderRadius: "8px",
                padding: "10px 15px",
                fontWeight: "bold",
                fontFamily: "serif",
                fontSize: "14px",
                boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
            },
        },
    ];

    const edges: Edge[] = [
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
                fontFamily: "sans-serif",
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

    return (
        <div className="w-full h-[200px] border border-[#dcd3b8] rounded-xl overflow-hidden bg-[#f4ecd8] shadow-inner">
            <ReactFlowProvider>
                <Flow nodes={nodes} edges={edges} />
            </ReactFlowProvider>
        </div>
    );
}

