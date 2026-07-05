import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';

// ==========================================
// 1. CONSTANTS & GRAPH CONFIGURATIONS
// ==========================================

const NODE_POSITIONS = {
  0: { x: 100, y: 200, label: 'A' },
  1: { x: 260, y: 100, label: 'B' },
  2: { x: 260, y: 300, label: 'C' },
  3: { x: 420, y: 100, label: 'D' },
  4: { x: 420, y: 300, label: 'E' },
};

const PRESETS = {
  STANDARD: {
    name: 'Standard (Negative Edge, No Cycle)',
    nodes: [0, 1, 2, 3, 4],
    edges: [
      { id: 'e1', from: 0, to: 1, weight: 6 },
      { id: 'e2', from: 0, to: 2, weight: 7 },
      { id: 'e3', from: 2, to: 1, weight: -2 },
      { id: 'e4', from: 1, to: 3, weight: 5 },
      { id: 'e5', from: 2, to: 4, weight: 9 },
      { id: 'e6', from: 3, to: 2, weight: -3 },
      { id: 'e7', from: 3, to: 4, weight: 2 },
    ],
  },
  NEGATIVE_CYCLE: {
    name: 'Negative Cycle Detectable',
    nodes: [0, 1, 2, 3, 4],
    edges: [
      { id: 'e1', from: 0, to: 1, weight: 6 },
      { id: 'e2', from: 0, to: 2, weight: 7 },
      { id: 'e3', from: 2, to: 1, weight: -2 },
      { id: 'e4', from: 1, to: 3, weight: 5 },
      { id: 'e5', from: 3, to: 2, weight: -4 }, // This edge creates the negative cycle (C -> B -> D -> C: -2 + 5 - 4 = -1)
      { id: 'e6', from: 2, to: 4, weight: 3 },
    ],
  },
};

// ==========================================
// 2. CORE ALGORITHM LOGIC (MODULAR ENGINE)
// ==========================================

/**
 * Generates an array of visual steps executing the Bellman-Ford algorithm.
 * This function is isolated from the React rendering logic.
 * 
 * @param {Array} nodes List of node indices
 * @param {Array} edges List of directed edge objects
 * @param {number} source Node index to start from
 * @returns {Array} List of step objects containing system snapshots
 */
function generateBellmanFordSteps(nodes, edges, source) {
  const steps = [];
  const numVertices = nodes.length;

  // Initialize data structures
  const dist = {};
  const pred = {};
  nodes.forEach((node) => {
    dist[node] = Infinity;
    pred[node] = null;
  });
  dist[source] = 0;

  // Formatting helper for infinite values
  const formatDist = (val) => (val === Infinity ? '∞' : val);

  // Record initialization step
  steps.push({
    type: 'init',
    iteration: 0,
    activeEdgeId: null,
    activeNodeId: null,
    distances: { ...dist },
    predecessors: { ...pred },
    message: `Initialized graph. Set source node ${NODE_POSITIONS[source].label} distance to 0, and all other nodes to Infinity.`,
  });

  let changed = false;

  // Relax edges |V| - 1 times
  for (let i = 1; i < numVertices; i++) {
    changed = false;
    for (let j = 0; j < edges.length; j++) {
      const edge = edges[j];
      const u = edge.from;
      const v = edge.to;
      const w = edge.weight;

      const labelU = NODE_POSITIONS[u].label;
      const labelV = NODE_POSITIONS[v].label;

      const currDistU = dist[u];
      const currDistV = dist[v];
      const newCalculatedDist = currDistU === Infinity ? Infinity : currDistU + w;

      // Handle unreachable node scenarios gracefully with mathematical notations
      if (currDistU === Infinity) {
        steps.push({
          type: 'relax-unreachable',
          iteration: i,
          activeEdgeId: edge.id,
          activeNodeId: v,
          distances: { ...dist },
          predecessors: { ...pred },
          message: `Evaluating edge ${labelU}→${labelV} (weight ${w}): dist[${labelU}] is ∞. Since source is unreachable, skipping relaxation.`,
        });
        continue;
      }

      // Success vs. Failure mathematical evaluation & path update log output
      if (newCalculatedDist < currDistV) {
        dist[v] = newCalculatedDist;
        pred[v] = u;
        changed = true;

        steps.push({
          type: 'relax-success',
          iteration: i,
          activeEdgeId: edge.id,
          activeNodeId: v,
          distances: { ...dist },
          predecessors: { ...pred },
          message: `Relaxing edge ${labelU}→${labelV} (weight ${w}): dist[${labelU}] (${currDistU}) + ${w} = ${newCalculatedDist}. Since ${newCalculatedDist} < dist[${labelV}] (${formatDist(currDistV)}), updated dist[${labelV}] to ${newCalculatedDist}.`,
        });
      } else {
        steps.push({
          type: 'relax-fail',
          iteration: i,
          activeEdgeId: edge.id,
          activeNodeId: v,
          distances: { ...dist },
          predecessors: { ...pred },
          message: `Relaxing ${labelU}→${labelV} (weight ${w}): ${currDistU} + ${w} = ${newCalculatedDist}. Since ${newCalculatedDist} is not < dist[${labelV}] (${formatDist(currDistV)}), no update.`,
        });
      }
    }

    // Optimization: If no distances changed during an entire pass, early convergence is achieved.
    if (!changed) {
      steps.push({
        type: 'early-stop',
        iteration: i,
        activeEdgeId: null,
        activeNodeId: null,
        distances: { ...dist },
        predecessors: { ...pred },
        message: `Iteration ${i} completed with zero distance modifications. Early Convergence triggered. Shortest paths have converged; skipping remaining iterations.`,
      });
      break;
    }
  }

  // Check for negative weight cycles (V-th iteration check)
  if (changed || numVertices > 1) {
    for (let j = 0; j < edges.length; j++) {
      const edge = edges[j];
      const u = edge.from;
      const v = edge.to;
      const w = edge.weight;

      const labelU = NODE_POSITIONS[u].label;
      const labelV = NODE_POSITIONS[v].label;

      if (dist[u] !== Infinity && dist[u] + w < dist[v]) {
        // Negative cycle discovered
        steps.push({
          type: 'negative-cycle-found',
          iteration: numVertices,
          activeEdgeId: edge.id,
          activeNodeId: v,
          distances: { ...dist },
          predecessors: { ...pred },
          message: `CRITICAL: Negative-weight cycle detected along edge ${labelU} → ${labelV}! Path distance continues to decrease infinitely.`,
        });
        return steps;
      }
    }
  }

  // If we reach here, no negative cycle exists
  steps.push({
    type: 'completed',
    iteration: numVertices,
    activeEdgeId: null,
    activeNodeId: null,
    distances: { ...dist },
    predecessors: { ...pred },
    message: 'Algorithm finished. All shortest paths confirmed with no negative cycles.',
  });

  return steps;
}

// ==========================================
// 3. MAIN COMPONENT IMPLEMENTATION
// ==========================================

export default function App() {
  const [selectedPresetKey, setSelectedPresetKey] = useState('STANDARD');
  const [sourceNode, setSourceNode] = useState(0);
  const [currentStepIdx, setCurrentStepIdx] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1000); // ms per step
  const [isPlaying, setIsPlaying] = useState(false);

  const activePreset = PRESETS[selectedPresetKey];
  const timerRef = useRef(null);

  // Generate steps when data dependency shifts
  const steps = useMemo(() => {
    return generateBellmanFordSteps(activePreset.nodes, activePreset.edges, sourceNode);
  }, [selectedPresetKey, sourceNode]);

  // Constrain index within generated steps boundary
  useEffect(() => {
    setCurrentStepIdx(0);
    setIsPlaying(false);
  }, [steps]);

  const currentStep = steps[currentStepIdx] || {
    type: 'init',
    iteration: 0,
    activeEdgeId: null,
    activeNodeId: null,
    distances: {},
    predecessors: {},
    message: '',
  };

  // Playback control functions
  const handleNext = useCallback(() => {
    setCurrentStepIdx((prev) => Math.min(prev + 1, steps.length - 1));
  }, [steps.length]);

  const handlePrev = useCallback(() => {
    setCurrentStepIdx((prev) => Math.max(prev - 1, 0));
  }, []);

  const handleReset = useCallback(() => {
    setCurrentStepIdx(0);
    setIsPlaying(false);
  }, []);

  // Playback Loop
  useEffect(() => {
    if (isPlaying) {
      if (currentStepIdx >= steps.length - 1) {
        setIsPlaying(false);
        return;
      }
      timerRef.current = setTimeout(() => {
        handleNext();
      }, playbackSpeed);
    } else {
      if (timerRef.current) clearTimeout(timerRef.current);
    }

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isPlaying, currentStepIdx, playbackSpeed, steps.length, handleNext]);

  return (
    <div className="dev-app-container">
      {/* Visual Diagnostic Header */}
      <header className="dev-header">
        <div className="dev-header-left">
          <div className="dev-logo-badge">BF</div>
          <div>
            <h1>Bellman-Ford Diagnostic Engine</h1>
            <p className="dev-system-status">
              System State: <span className="status-live">ONLINE</span> | Module: <span className="status-module">PATH_FINDER_BF</span>
            </p>
          </div>
        </div>
        <div className="dev-header-right">
          <span className="dev-tag">v1.2.0-stable</span>
        </div>
      </header>

      {/* Main Structural Grid */}
      <main className="dev-grid">
        {/* Left Area: Canvas Visualization Card */}
        <div className="dev-card main-viz-card">
          <div className="dev-card-header">
            <div className="header-meta">
              <span className="indicator-dot blinking"></span>
              <h2>Graph Viewport</h2>
            </div>
            <div className="config-selectors">
              <div className="select-wrapper">
                <select
                  id="preset-select"
                  value={selectedPresetKey}
                  onChange={(e) => setSelectedPresetKey(e.target.value)}
                >
                  {Object.keys(PRESETS).map((key) => (
                    <option key={key} value={key}>
                      {PRESETS[key].name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="select-wrapper">
                <select
                  id="source-select"
                  value={sourceNode}
                  onChange={(e) => setSourceNode(Number(e.target.value))}
                >
                  {activePreset.nodes.map((node) => (
                    <option key={node} value={node}>
                      Src: Node {NODE_POSITIONS[node].label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          <div className="dev-canvas-viewport">
            <svg viewBox="0 0 540 400" className="dev-graph-svg">
              <defs>
                {/* Blueprint visual grid */}
                <pattern id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
                  <path d="M 30 0 L 0 0 0 30" fill="none" stroke="rgba(255, 255, 255, 0.03)" strokeWidth="1" />
                </pattern>

                {/* Arrow markers utilizing CSS custom variables */}
                <marker
                  id="arrow"
                  viewBox="0 0 10 10"
                  refX="6"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="var(--state-initial)" />
                </marker>
                <marker
                  id="arrow-active"
                  viewBox="0 0 10 10"
                  refX="6"
                  refY="5"
                  markerWidth="7"
                  markerHeight="7"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="var(--state-scanning)" />
                </marker>
                <marker
                  id="arrow-success"
                  viewBox="0 0 10 10"
                  refX="6"
                  refY="5"
                  markerWidth="7"
                  markerHeight="7"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="var(--state-relaxed)" />
                </marker>
                <marker
                  id="arrow-error"
                  viewBox="0 0 10 10"
                  refX="6"
                  refY="5"
                  markerWidth="7"
                  markerHeight="7"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="var(--state-error)" />
                </marker>
              </defs>

              <rect width="100%" height="100%" fill="url(#grid)" />

              {/* Render Edges */}
              {activePreset.edges.map((edge) => {
                const start = NODE_POSITIONS[edge.from];
                const end = NODE_POSITIONS[edge.to];
                const isActive = currentStep.activeEdgeId === edge.id;

                const dx = end.x - start.x;
                const dy = end.y - start.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const r = 24; 
                const x1 = start.x + (dx * r) / dist;
                const y1 = start.y + (dy * r) / dist;
                const x2 = end.x - (dx * r) / dist;
                const y2 = end.y - (dy * r) / dist;

                const mx = (start.x + end.x) / 2;
                const my = (start.y + end.y) / 2 - 10;

                // Color mappings using CSS Variables
                let edgeColor = 'var(--state-initial)';
                let markerId = 'arrow';
                let strokeWidth = 2;

                if (isActive) {
                  strokeWidth = 3.5;
                  if (currentStep.type === 'relax-success') {
                    edgeColor = 'var(--state-relaxed)';
                    markerId = 'arrow-success';
                  } else if (currentStep.type === 'negative-cycle-found') {
                    edgeColor = 'var(--state-error)';
                    markerId = 'arrow-error';
                  } else {
                    edgeColor = 'var(--state-scanning)';
                    markerId = 'arrow-active';
                  }
                }

                return (
                  <g key={edge.id} className="dev-edge-group">
                    <line
                      x1={x1}
                      y1={y1}
                      x2={x2}
                      y2={y2}
                      stroke={edgeColor}
                      strokeWidth={strokeWidth}
                      markerEnd={`url(#${markerId})`}
                      className="dev-edge-line"
                    />
                    <rect
                      x={mx - 15}
                      y={my - 12}
                      width={30}
                      height={20}
                      rx={4}
                      fill="var(--bg-main)"
                      stroke="var(--border-color)"
                      className="dev-edge-weight-bg"
                    />
                    <text
                      x={mx}
                      y={my}
                      textAnchor="middle"
                      alignmentBaseline="middle"
                      className={`dev-edge-weight ${edge.weight < 0 ? 'weight-negative' : ''}`}
                    >
                      {edge.weight}
                    </text>
                  </g>
                );
              })}

              {/* Render Nodes */}
              {activePreset.nodes.map((nodeIdx) => {
                const node = NODE_POSITIONS[nodeIdx];
                const distanceVal = currentStep.distances[nodeIdx];
                const isCurrentActive = currentStep.activeNodeId === nodeIdx;
                const isSource = nodeIdx === sourceNode;

                let borderClass = 'node-standard';
                if (isCurrentActive) {
                  if (currentStep.type === 'relax-success') borderClass = 'node-success';
                  else if (currentStep.type === 'negative-cycle-found') borderClass = 'node-error';
                  else borderClass = 'node-active';
                } else if (isSource) {
                  borderClass = 'node-source';
                }

                return (
                  <g key={nodeIdx} transform={`translate(${node.x}, ${node.y})`} className="dev-node-group">
                    <circle r="22" className={`dev-node-circle ${borderClass}`} />
                    <text textAnchor="middle" y="-4" className="dev-node-label">
                      {node.label}
                    </text>
                    <text textAnchor="middle" y="14" className="dev-node-distance">
                      {distanceVal === Infinity ? '∞' : distanceVal}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Dedicated Playback and Controls Panel */}
          <div className="dev-controls-container">
            <div className="playback-controls-row">
              <button onClick={handleReset} className="dev-btn btn-secondary" title="Reset Step-by-Step">
                Reset
              </button>
              <button onClick={handlePrev} disabled={currentStepIdx === 0} className="dev-btn btn-secondary">
                ◀ Back
              </button>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`dev-btn ${isPlaying ? 'btn-warn' : 'btn-primary'}`}
              >
                {isPlaying ? 'Pause' : 'Play Simulation'}
              </button>
              <button
                onClick={handleNext}
                disabled={currentStepIdx === steps.length - 1}
                className="dev-btn btn-secondary"
              >
                Forward ▶
              </button>
            </div>

            <div className="speed-slider-row">
              <div className="slider-label">
                <span>SIMULATION_DELAY</span>
                <span className="slider-value">{playbackSpeed}ms</span>
              </div>
              <input
                id="speed-slider"
                type="range"
                min="200"
                max="2500"
                step="100"
                value={playbackSpeed}
                onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                className="dev-slider"
              />
            </div>
          </div>
        </div>

        {/* Right Area: Sidebar Panels (Terminal + Arrays + Legends) */}
        <div className="dev-sidebar-grid">
          {/* Terminal / Log Container */}
          <div className="dev-card terminal-card">
            <div className="dev-card-header">
              <h2>Diagnostic Log</h2>
              <div className="terminal-header-meta">
                <span>ITERATION: {currentStep.iteration}</span>
                <span>STEP: {currentStepIdx + 1}/{steps.length}</span>
              </div>
            </div>
            <div className="dev-terminal-body">
              <div className="terminal-prompt">
                <span className="prompt-symbol">&gt;_</span>
                <span className="prompt-path">engine@bellman-ford:</span>
              </div>
              <div className="terminal-output">
                <p className={`log-message log-type-${currentStep.type}`}>
                  {currentStep.message}
                </p>
              </div>
            </div>
          </div>

          {/* Real-time State Table Container */}
          <div className="dev-card state-table-card">
            <div className="dev-card-header">
              <h2>Register State Array</h2>
            </div>
            <div className="dev-table-container">
              <table className="dev-state-table">
                <thead>
                  <tr>
                    <th>Register</th>
                    <th>Distance Value</th>
                    <th>Predecessor Reference</th>
                  </tr>
                </thead>
                <tbody>
                  {activePreset.nodes.map((nodeIdx) => {
                    const label = NODE_POSITIONS[nodeIdx].label;
                    const dist = currentStep.distances[nodeIdx];
                    const predNode = currentStep.predecessors[nodeIdx];
                    const predLabel = predNode !== null && predNode !== undefined ? NODE_POSITIONS[predNode].label : '0x0';

                    const isNodeActive = currentStep.activeNodeId === nodeIdx;
                    const isSuccessUpdate = isNodeActive && currentStep.type === 'relax-success';

                    return (
                      <tr key={nodeIdx} className={isSuccessUpdate ? 'dev-row-update' : ''}>
                        <td className="node-reg-cell">
                          <span className="reg-icon"></span>
                          REG_{label}
                        </td>
                        <td>
                          <span className={`reg-badge-dist ${dist === Infinity ? 'state-inf' : 'state-val'}`}>
                            {dist === Infinity ? '0xINF (∞)' : dist}
                          </span>
                        </td>
                        <td className="predecessor-cell">{predLabel !== '0x0' ? `REG_${predLabel}` : 'NULL'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Color Mapping System Legend */}
          <div className="dev-card legend-card">
            <div className="dev-card-header">
              <h2>Viewport Key</h2>
            </div>
            <div className="dev-legend-grid">
              <div className="legend-cell">
                <span className="legend-indicator" style={{ borderColor: 'var(--color-primary)', backgroundColor: 'rgba(59, 130, 246, 0.15)' }}></span>
                <span>Source Register</span>
              </div>
              <div className="legend-cell">
                <span className="legend-indicator" style={{ borderColor: 'var(--state-scanning)', backgroundColor: 'rgba(245, 158, 11, 0.15)' }}></span>
                <span>Scanning (Relaxing)</span>
              </div>
              <div className="legend-cell">
                <span className="legend-indicator" style={{ borderColor: 'var(--state-relaxed)', backgroundColor: 'rgba(16, 185, 129, 0.15)' }}></span>
                <span>Relaxed State</span>
              </div>
              <div className="legend-cell">
                <span className="legend-indicator" style={{ borderColor: 'var(--state-error)', backgroundColor: 'rgba(239, 68, 68, 0.15)' }}></span>
                <span>Cycle Detected</span>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}