import dagre from "dagre";
import type { Edge, Node } from "@xyflow/react";

const NODE_WIDTH = 160;
const NODE_HEIGHT = 50;

export function getLayoutedElements(
    nodes: Node[],
    edges: Edge[],
    direction: "LR" | "TB" = "LR"
): { nodes: Node[]; edges: Edge[] } {
    const graph = new dagre.graphlib.Graph();
    graph.setDefaultEdgeLabel(() => ({}));
    graph.setGraph({ rankdir: direction, nodesep: 60, ranksep: 100 });

    nodes.forEach((node) => {
        graph.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
    });

    edges.forEach((edge) => {
        graph.setEdge(edge.source, edge.target);
    });

    dagre.layout(graph);

    const layoutedNodes = nodes.map((node) => {
        const position = graph.node(node.id);
        return {
            ...node,
            position: {
                x: position.x - NODE_WIDTH / 2,
                y: position.y - NODE_HEIGHT / 2,
            },
        };
    });

    return { nodes: layoutedNodes, edges };
}
