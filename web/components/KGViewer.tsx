"use client";

import React, { useMemo } from 'react';
import { ReactFlow, Controls, Background, Node, Edge, Position, MarkerType } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { GraphCitation } from '@/lib/types';

interface KGViewerProps {
    citation: GraphCitation;
}

export function KGViewer({ citation }: KGViewerProps) {
    // Parse fact string like "陰 --CONTAINS--> 月"
    const parsed = useMemo(() => {
        const match = citation.fact.match(/^(.+?)\s*--(.+?)-->\s*(.+)$/);
        if (match) {
            return {
                source: match[1].trim(),
                relationship: match[2].trim(),
                target: match[3].trim(),
            };
        }
        return {
            source: 'Fact',
            relationship: 'IS',
            target: citation.fact,
        };
    }, [citation.fact]);

    const { nodes, edges } = useMemo(() => {
        const primaryColor = '#8c8578'; // Match parchment theme accent
        const nodeColor = '#3e382d'; // Dark background matching theme

        const initialNodes: Node[] = [
            {
                id: 'source',
                position: { x: 0, y: 50 },
                data: { label: parsed.source },
                sourcePosition: Position.Right,
                targetPosition: Position.Left,
                style: {
                    background: nodeColor,
                    color: '#f4ecd8', // parchment bg color
                    border: `1px solid ${primaryColor}`,
                    borderRadius: '8px',
                    padding: '10px 15px',
                    fontWeight: 'bold',
                    fontFamily: 'serif',
                    fontSize: '14px',
                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
                },
            },
            {
                id: 'target',
                position: { x: 250, y: 50 },
                data: { label: parsed.target },
                sourcePosition: Position.Right,
                targetPosition: Position.Left,
                style: {
                    background: nodeColor,
                    color: '#f4ecd8',
                    border: `1px solid ${primaryColor}`,
                    borderRadius: '8px',
                    padding: '10px 15px',
                    fontWeight: 'bold',
                    fontFamily: 'serif',
                    fontSize: '14px',
                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
                },
            },
        ];

        const initialEdges: Edge[] = [
            {
                id: 'edge1',
                source: 'source',
                target: 'target',
                label: parsed.relationship,
                animated: true,
                style: { stroke: primaryColor, strokeWidth: 2 },
                labelStyle: { fill: '#5c5548', fontWeight: 600, fontFamily: 'sans-serif', fontSize: 10 },
                labelBgStyle: { fill: '#ebe5d5', fillOpacity: 0.9, rx: 4, ry: 4 },
                markerEnd: {
                    type: MarkerType.ArrowClosed,
                    color: primaryColor,
                    width: 20,
                    height: 20,
                },
            },
        ];

        return { nodes: initialNodes, edges: initialEdges };
    }, [parsed]);

    return (
        <div className="w-full h-[200px] border border-[#dcd3b8] rounded-xl overflow-hidden bg-[#f4ecd8] shadow-inner">
            <ReactFlow 
                nodes={nodes} 
                edges={edges} 
                fitView
                fitViewOptions={{ padding: 0.3 }}
                proOptions={{ hideAttribution: true }}
                nodesDraggable={false}
                nodesConnectable={false}
                zoomOnScroll={false}
                panOnDrag={true}
            >
                <Background color="#dcd3b8" gap={16} size={1} />
                <Controls showInteractive={false} className="opacity-50 hover:opacity-100 transition-opacity [&_.react-flow__controls-button]:border-[#dcd3b8] [&_.react-flow__controls-button]:bg-[#ebe5d5] [&_.react-flow__controls-button]:text-[#5c5548] [&_.react-flow__controls-button:hover]:bg-[#dcd3b8]" />
            </ReactFlow>
        </div>
    );
}
