"use client";

import React, { useEffect, useState, useRef, useCallback } from "react";
import cytoscape, { Core, ElementDefinition, NodeSingular } from "cytoscape";
import cytoscapeDagre from "cytoscape-dagre";
import CytoscapeComponent from "react-cytoscapejs";
import { ArrowLeft, Search, Loader2, Network, GitBranch, Expand, ExternalLink } from "lucide-react";
import { useI18n } from "@/i18n/context";
import { useRouter } from "next/navigation";
import { fetchSubgraph, fetchGraphSearch, GraphSearchResult } from "@/lib/api";
import Link from "next/link";

// Safe to register at module level — this file is never loaded on the server
cytoscape.use(cytoscapeDagre);

const TYPE_COLORS: Record<string, string> = {
  Herb: "#4ade80",
  Formula: "#60a5fa",
  Symptom: "#f87171",
  Pattern: "#c084fc",
  Meridian: "#fbbf24",
  Acupoint: "#fb923c",
  Disease: "#f472b6",
  BodyPart: "#a78bfa",
  TreatmentMethod: "#34d399",
  Default: "#94a3b8"
};

interface KGExplorerContentProps {
  entityId: string;
}

export default function KGExplorerContent({ entityId }: KGExplorerContentProps) {
  const router = useRouter();
  const cyRef = useRef<Core | null>(null);
  const { t } = useI18n();

  const [elements, setElements] = useState<ElementDefinition[]>([]);
  const [layoutName, setLayoutName] = useState<"cose" | "dagre">("cose");
  const [loading, setLoading] = useState(true);
  const [expanding, setExpanding] = useState(false);
  
  // Search state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<GraphSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);

  // Selected node info
  const [selectedNode, setSelectedNode] = useState<{ id: string; label: string; type: string; degree: number } | null>(null);

  const decodedEntityId = decodeURIComponent(entityId);
  const typeLabels = t.kgExplorer.typeLabels as Record<string, string>;
  const getTypeLabel = useCallback((type: string) => typeLabels[type] || type, [typeLabels]);

  // Initial fetch
  useEffect(() => {
    let mounted = true;
    const loadInitialGraph = async () => {
      try {
        setLoading(true);
        const data = await fetchSubgraph(decodedEntityId, 2);
        if (!mounted) return;
        
        const cyElements: ElementDefinition[] = [
          ...data.nodes.map(n => ({
            data: { id: n.id, label: n.label, type: n.type }
          })),
          ...data.edges.map((e, i) => ({
            data: { id: `e-${i}`, source: e.source, target: e.target, label: e.label }
          }))
        ];
        
        setElements(cyElements);
      } catch (error) {
        console.error("Failed to fetch initial graph", error);
      } finally {
        if (mounted) setLoading(false);
      }
    };
    
    loadInitialGraph();
    return () => { mounted = false; };
  }, [decodedEntityId]);

  // Handle Search Debounce
  useEffect(() => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      setShowDropdown(false);
      return;
    }

    const timer = setTimeout(async () => {
      try {
        setIsSearching(true);
        const results = await fetchGraphSearch(searchQuery);
        setSearchResults(results);
        setShowDropdown(true);
      } catch (error) {
        console.error("Search failed", error);
      } finally {
        setIsSearching(false);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Define layout options
  const getLayoutOptions = useCallback((name: string) => {
    if (name === "cose") {
      return { name: "cose", animate: true, animationDuration: 500, nodeRepulsion: 8000, idealEdgeLength: 100 };
    }
    return { name: "dagre", rankDir: "LR", nodeSep: 60, rankSep: 100, animate: true, animationDuration: 500 };
  }, []);

  // Run layout when name changes
  useEffect(() => {
    if (cyRef.current && elements.length > 0) {
      cyRef.current.layout(getLayoutOptions(layoutName)).run();
    }
  }, [layoutName, getLayoutOptions, elements.length]);

  // Cytoscape Event Listeners
  const setupCyListeners = useCallback((cy: Core) => {
    cyRef.current = cy;
    
    cy.off("tap", "node");
    cy.off("tap");

    cy.on("tap", "node", (evt) => {
      const node = evt.target as NodeSingular;
      setSelectedNode({
        id: node.id(),
        label: node.data("label"),
        type: node.data("type"),
        degree: node.degree(false)
      });
    });

    cy.on("tap", (evt) => {
      if (evt.target === cy) {
        setSelectedNode(null);
      }
    });
  }, []);

  // Expand node
  const handleExpand = async () => {
    if (!selectedNode || !cyRef.current) return;
    try {
      setExpanding(true);
      const data = await fetchSubgraph(selectedNode.label, 1);
      
      const cy = cyRef.current;
      const existingNodes = new Set(cy.nodes().map(n => n.id()));
      const existingEdges = new Set(cy.edges().map(e => `${e.data("source")}-${e.data("target")}`));

      const newNodes = data.nodes
        .filter(n => !existingNodes.has(n.id))
        .map(n => ({ group: "nodes" as const, data: { id: n.id, label: n.label, type: n.type } }));
        
      const newEdges = data.edges
        .filter(e => !existingEdges.has(`${e.source}-${e.target}`))
        .map((e, i) => ({ 
          group: "edges" as const, 
          data: { id: `e-${Date.now()}-${i}`, source: e.source, target: e.target, label: e.label } 
        }));

      if (newNodes.length > 0 || newEdges.length > 0) {
        cy.add([...newNodes, ...newEdges]);
        cy.layout(getLayoutOptions(layoutName)).run();
        
        const tappedNode = cy.getElementById(selectedNode.id);
        cy.animate({
          center: { eles: tappedNode },
          zoom: 1.2,
          duration: 500
        });
        
        // Sync React state
        setElements(cy.elements().map(e => ({ data: e.data(), group: e.group() } as ElementDefinition)));
      }
    } catch (error) {
      console.error("Failed to expand node", error);
    } finally {
      setExpanding(false);
    }
  };

  const cytoscapeStylesheet: React.ComponentProps<typeof CytoscapeComponent>["stylesheet"] = [
    {
      selector: "node",
      style: {
        "label": "data(label)",
        "shape": "round-rectangle",
        "background-color": (node: NodeSingular) => TYPE_COLORS[node.data("type")] || TYPE_COLORS.Default,
        "color": "#ffffff",
        "text-valign": "center",
        "text-halign": "center",
        "font-size": "12px",
        "font-family": "var(--font-noto-serif-sc), serif",
        "width": "label",
        "height": "label",
        "padding": "8px",
      }
    },
    {
      selector: "edge",
      style: {
        "width": 1.5,
        "line-color": "#8c8578",
        "target-arrow-color": "#8c8578",
        "target-arrow-shape": "triangle",
        "curve-style": "bezier",
        "label": "data(label)",
        "font-size": "10px",
        "color": "#8c8578",
        "text-rotation": "autorotate",
        "text-margin-y": -10,
        "text-background-opacity": 1,
        "text-background-color": "#112120",
        "text-background-padding": "2px",
        "text-background-shape": "roundrectangle"
      }
    }
  ];

  return (
    <div className="h-screen w-full flex flex-col bg-background-dark text-parchment overflow-hidden font-sans">
      {/* Top Bar */}
      <div className="h-14 bg-sidebar-dark border-b border-[#1a3130] flex items-center justify-between px-4 z-10 shrink-0">
        <div className="flex items-center gap-4 flex-1">
          <Link href="/" className="flex items-center gap-2 text-sm text-slate-400 hover:text-parchment transition-colors">
            <ArrowLeft size={16} />
            <span className="hidden sm:inline">{t.common.backToApp}</span>
          </Link>
          
          <div className="h-6 w-px bg-[#1a3130] hidden sm:block"></div>
          
          <h1 className="font-serif text-lg font-medium truncate max-w-[150px] sm:max-w-xs">{decodedEntityId}</h1>
        </div>
        
        <div className="flex-1 max-w-md mx-4 relative hidden md:block">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
              <input 
               type="text"
               placeholder={t.kgExplorer.searchPlaceholder}
               value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onFocus={() => { if (searchResults.length > 0) setShowDropdown(true); }}
              onBlur={() => setTimeout(() => setShowDropdown(false), 200)}
              className="w-full bg-background-dark border border-[#1a3130] rounded-md py-1.5 pl-9 pr-4 text-sm text-parchment placeholder-slate-400 focus:outline-none focus:border-primary"
            />
            {isSearching && (
              <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 animate-spin text-primary" size={16} />
            )}
          </div>
          
          {/* Search Dropdown */}
          {showDropdown && searchResults.length > 0 && (
            <div className="absolute top-full left-0 right-0 mt-1 bg-sidebar-dark border border-[#1a3130] rounded-md shadow-lg overflow-hidden max-h-60 overflow-y-auto z-50">
              {searchResults.map((res) => (
                <button
                  key={res.id}
                  onClick={() => {
                    setSearchQuery("");
                    setShowDropdown(false);
                    router.push(`/kg/${encodeURIComponent(res.id)}`);
                  }}
                  className="w-full text-left px-4 py-2 text-sm hover:bg-background-dark flex items-center justify-between group"
                >
                  <span className="font-serif group-hover:text-primary transition-colors">{res.label}</span>
                  <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ backgroundColor: `${TYPE_COLORS[res.type] || TYPE_COLORS.Default}20`, color: TYPE_COLORS[res.type] || TYPE_COLORS.Default }}>
                    {getTypeLabel(res.type)}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
        
        <div className="flex items-center gap-1.5 bg-background-dark p-1 rounded-md border border-[#1a3130] flex-1 justify-end max-w-fit">
          <button
            onClick={() => setLayoutName("cose")}
            className={`px-3 py-1 text-xs rounded flex items-center gap-1.5 transition-colors ${layoutName === "cose" ? "bg-primary/10 text-primary" : "text-slate-400 hover:text-parchment"}`}
          >
            <Network size={14} /> <span className="hidden sm:inline">{t.kgExplorer.forceLayout}</span>
          </button>
          <button
            onClick={() => setLayoutName("dagre")}
            className={`px-3 py-1 text-xs rounded flex items-center gap-1.5 transition-colors ${layoutName === "dagre" ? "bg-primary/10 text-primary" : "text-slate-400 hover:text-parchment"}`}
          >
            <GitBranch size={14} /> <span className="hidden sm:inline">{t.kgExplorer.hierarchicalLayout}</span>
          </button>
        </div>
      </div>

      {/* Mobile Search Bar */}
      <div className="md:hidden px-4 py-2 bg-sidebar-dark border-b border-[#1a3130] z-10 shrink-0">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
          <input 
            type="text"
            placeholder={t.kgExplorer.searchPlaceholder}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onFocus={() => { if (searchResults.length > 0) setShowDropdown(true); }}
            onBlur={() => setTimeout(() => setShowDropdown(false), 200)}
            className="w-full bg-background-dark border border-[#1a3130] rounded-md py-1.5 pl-9 pr-4 text-sm text-parchment placeholder-slate-400 focus:outline-none focus:border-primary"
          />
          {isSearching && (
            <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 animate-spin text-primary" size={16} />
          )}
          {showDropdown && searchResults.length > 0 && (
            <div className="absolute top-full left-0 right-0 mt-1 bg-sidebar-dark border border-[#1a3130] rounded-md shadow-lg overflow-hidden max-h-60 overflow-y-auto z-50">
              {searchResults.map((res) => (
                <button
                  key={res.id}
                  onClick={() => {
                    setSearchQuery("");
                    setShowDropdown(false);
                    router.push(`/kg/${encodeURIComponent(res.id)}`);
                  }}
                  className="w-full text-left px-4 py-2 text-sm hover:bg-background-dark flex items-center justify-between group"
                >
                  <span className="font-serif group-hover:text-primary transition-colors">{res.label}</span>
                  <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ backgroundColor: `${TYPE_COLORS[res.type] || TYPE_COLORS.Default}20`, color: TYPE_COLORS[res.type] || TYPE_COLORS.Default }}>
                    {getTypeLabel(res.type)}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main Graph Area */}
      <div className="flex-1 relative w-full h-full bg-background-dark">
        {loading ? (
          <div className="absolute inset-0 flex items-center justify-center bg-background-dark/80 z-20 backdrop-blur-sm">
              <div className="flex flex-col items-center gap-3">
                <Loader2 className="animate-spin text-primary" size={32} />
                <p className="text-sm text-slate-400">{t.kgExplorer.mapping}</p>
              </div>
            </div>
        ) : (
          <CytoscapeComponent
            elements={elements}
            style={{ width: "100%", height: "100%" }}
            stylesheet={cytoscapeStylesheet}
            layout={getLayoutOptions(layoutName)}
            cy={(cy) => setupCyListeners(cy)}
            wheelSensitivity={0.2}
            className="w-full h-full"
          />
        )}

        {/* Info Panel */}
        {selectedNode && (
          <div className="absolute bottom-6 left-6 w-64 bg-sidebar-dark/95 backdrop-blur-md border border-[#1a3130] rounded-lg shadow-xl p-4 z-10 flex flex-col gap-3">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-slate-400 mb-1 flex items-center justify-between">
                <span className="font-medium px-2 py-0.5 rounded" style={{ backgroundColor: `${TYPE_COLORS[selectedNode.type] || TYPE_COLORS.Default}15`, color: TYPE_COLORS[selectedNode.type] || TYPE_COLORS.Default }}>
                  {getTypeLabel(selectedNode.type)}
                </span>
                <span className="bg-background-dark px-1.5 py-0.5 rounded text-[10px] border border-[#1a3130]">{selectedNode.degree} {t.kgExplorer.links}</span>
              </div>
              <h3 className="font-serif text-lg text-parchment leading-tight mt-2">{selectedNode.label}</h3>
            </div>
            
            <div className="flex flex-col gap-2 mt-1">
              <button
                onClick={handleExpand}
                disabled={expanding}
                className="w-full py-1.5 bg-primary/10 hover:bg-primary/20 text-primary text-sm rounded flex items-center justify-center gap-2 transition-colors disabled:opacity-50 font-medium"
              >
                {expanding ? <Loader2 size={14} className="animate-spin" /> : <Expand size={14} />}
                {expanding ? t.kgExplorer.fetching : t.kgExplorer.expandNetwork}
              </button>
              
              <button 
                onClick={() => router.push(`/kg/${encodeURIComponent(selectedNode.id)}`)}
                className="w-full py-1.5 bg-background-dark hover:bg-[#1a3130] text-slate-400 hover:text-parchment text-sm rounded flex items-center justify-center gap-2 transition-colors"
              >
                <ExternalLink size={14} />
                {t.kgExplorer.focusAsRoot}
              </button>
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="absolute bottom-6 right-6 bg-sidebar-dark/90 backdrop-blur-sm border border-[#1a3130] rounded-lg shadow-xl p-3 z-10 hidden sm:block">
          <h4 className="text-[10px] font-medium text-slate-400 mb-2 uppercase tracking-wider">{t.kgExplorer.entityTypes}</h4>
          <div className="grid grid-cols-2 gap-x-4 gap-y-2">
            {Object.entries(TYPE_COLORS).filter(([t]) => t !== "Default").map(([type, color]) => (
              <div key={type} className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-sm shadow-sm" style={{ backgroundColor: color }} />
                <span className="text-[11px] text-parchment/80">{getTypeLabel(type)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
