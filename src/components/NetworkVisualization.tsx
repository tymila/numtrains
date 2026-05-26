import React from 'react';
import { NetworkState, ForwardResult } from '../lib/nn';

interface Props {
  nn: NetworkState;
  fwd: ForwardResult;
  animStage: string;
}

const SmoothLine = ({ x1, y1, x2, y2, stroke, strokeWidth, opacity }: any) => {
  const midX = (x1 + x2) / 2;
  const d = `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`;
  return (
    <path 
      d={d} 
      stroke={stroke} 
      strokeWidth={strokeWidth} 
      strokeOpacity={opacity} 
      fill="none" 
      className="transition-all duration-300"
    />
  );
};

const getLineColor = (w: number, isBackward = false) => {
  if (isBackward) return '#a855f7'; // Purple for error propagation
  const alpha = Math.min(1, Math.abs(w) * 0.8 + 0.1);
  // Red for positive, Blue for negative
  return w > 0 ? `rgba(239, 68, 68, ${alpha})` : `rgba(59, 130, 246, ${alpha})`;
};

export default function NetworkVisualization({ nn, fwd, animStage }: Props) {
  // Coordinate calculations mapped to a SVG viewBox of 1000x800
  const inputNodes = Array.from({ length: 25 }).map((_, i) => ({
    x: 100 + (i % 5) * 35,
    y: 280 + Math.floor(i / 5) * 35,
    act: fwd.input[i],
    isHighlight: animStage === 'inputs' || animStage === 'forward_1'
  }));

  const numHidden = nn.w1.length;
  const hiddenSpacing = 75;
  const hiddenStartY = 400 - ((numHidden - 1) * hiddenSpacing) / 2;
  
  const hiddenNodes = nn.w1.map((_, i) => ({
    x: 500,
    y: hiddenStartY + i * hiddenSpacing,
    act: fwd.a1[i],
    weights: nn.w1[i],
    bias: nn.b1[i],
    isHighlight: animStage === 'hidden' || animStage === 'forward_2'
  }));

  const numOutput = nn.w2.length;
  const outputSpacing = 65;
  const outputStartY = 400 - ((numOutput - 1) * outputSpacing) / 2;

  const outputNodes = nn.w2.map((_, i) => ({
    x: 820,
    y: outputStartY + i * outputSpacing,
    act: fwd.a2[i],
    bias: nn.b2[i],
    label: i,
    isHighlight: animStage === 'output' || animStage === 'backward_2'
  }));

  return (
    <div className="w-full h-full relative overflow-hidden bg-zinc-950 font-mono">
      <svg viewBox="0 0 1000 800" className="w-full h-full" preserveAspectRatio="xMidYMid meet">
        {/* Connection Lines Input -> Hidden */}
        {inputNodes.map((inNode, i) =>
          hiddenNodes.map((hidNode, j) => {
            const w = nn.w1[j][i];
            const isFwdPulse = animStage === 'forward_1';
            const isBckPulse = animStage === 'backward_1';
            const pulse = isFwdPulse || isBckPulse;
            
            return (
              <SmoothLine
                key={`l1-${i}-${j}`}
                x1={inNode.x}
                y1={inNode.y}
                x2={hidNode.x - 15}
                y2={hidNode.y}
                stroke={isFwdPulse ? '#fff' : getLineColor(w, isBckPulse)}
                strokeWidth={pulse ? 1.5 : Math.max(0.1, Math.abs(w) * 1.5)}
                opacity={pulse ? 0.8 : 0.2}
              />
            );
          })
        )}

        {/* Connection Lines Hidden -> Output */}
        {hiddenNodes.map((hidNode, j) =>
          outputNodes.map((outNode, k) => {
            const w = nn.w2[k][j];
            const isFwdPulse = animStage === 'forward_2';
            const isBckPulse = animStage === 'backward_2';
            const pulse = isFwdPulse || isBckPulse;
            
            return (
              <SmoothLine
                key={`l2-${j}-${k}`}
                x1={hidNode.x + 65} 
                y1={hidNode.y}
                x2={outNode.x - 20}
                y2={outNode.y}
                stroke={isFwdPulse ? '#fff' : getLineColor(w, isBckPulse)}
                strokeWidth={pulse ? 1.5 : Math.max(0.2, Math.abs(w) * 1.5)}
                opacity={pulse ? 0.8 : 0.2}
              />
            );
          })
        )}

        {/* Input Layer Grid Backdrop */}
        <text x="170" y="220" fontSize="14" fill="#a1a1aa" textAnchor="middle" className="font-sans font-medium">输入层</text>
        <rect x="85" y="265" width="170" height="170" rx="8" fill="none" stroke="#27272a" strokeWidth="2" strokeDasharray="4 4" />
        
        {/* Input Nodes */}
        {inputNodes.map((node, i) => (
          <g key={`in-${i}`}>
            <circle
              cx={node.x}
              cy={node.y}
              r={12}
              fill={`rgba(255,255,255,${node.act})`}
              stroke={node.isHighlight ? '#a855f7' : '#3f3f46'}
              strokeWidth={node.isHighlight ? 2 : 1}
              className="transition-all duration-300"
            />
          </g>
        ))}

        {/* Hidden Layer Header */}
        <text x="500" y="100" fontSize="14" fill="#a1a1aa" textAnchor="middle" className="font-sans font-medium">隐藏层</text>

        {/* Hidden Nodes and Heatmaps */}
        {hiddenNodes.map((node, i) => (
          <g key={`hid-${i}`}>
            <circle
              cx={node.x}
              cy={node.y}
              r={20}
              fill={`rgba(255,255,255,${node.act})`}
              stroke={node.isHighlight ? '#a855f7' : '#52525b'}
              strokeWidth={node.isHighlight ? 3 : 2}
              className="transition-all duration-300"
            />
            <text x={node.x} y={node.y} dominantBaseline="middle" textAnchor="middle" fontSize="12" fill={node.act > 0.5 ? '#000' : '#fff'}>
              {node.act.toFixed(2)}
            </text>
            
            {/* 5x5 Weight Heatmap for this neuron */}
            <g transform={`translate(${node.x + 30}, ${node.y - 35})`}>
              <rect x="-3" y="-3" width="76" height="76" rx="4" fill="#18181b" stroke="#27272a" />
              {node.weights.map((w, idx) => (
                <g key={`hm-${i}-${idx}`} transform={`translate(${(idx % 5) * 14}, ${Math.floor(idx / 5) * 14})`}>
                  <rect
                    width={14}
                    height={14}
                    fill={w > 0 ? `rgba(239, 68, 68, ${Math.min(1, Math.abs(w))})` : `rgba(59, 130, 246, ${Math.min(1, Math.abs(w))})`}
                    className="transition-colors duration-200"
                  />
                  <text x="7" y="7" fontSize="5" fill="#fff" textAnchor="middle" dominantBaseline="middle" className="font-sans font-medium">
                    {w.toFixed(1)}
                  </text>
                </g>
              ))}
            </g>
          </g>
        ))}

        {/* Output Layer Header */}
        <text x="820" y="50" fontSize="14" fill="#a1a1aa" textAnchor="middle" className="font-sans font-medium">输出层 (数字0-9)</text>

        {/* Output Nodes */}
        {outputNodes.map((node, i) => (
          <g key={`out-${i}`}>
            <circle
              cx={node.x}
              cy={node.y}
              r={20}
              fill={`rgba(255,255,255,${node.act})`}
              stroke={node.isHighlight ? '#a855f7' : '#52525b'}
              strokeWidth={node.isHighlight ? 4 : 2}
              className="transition-all duration-300"
            />
            <text x={node.x} y={node.y} dominantBaseline="middle" textAnchor="middle" fontSize="14" fill={node.act > 0.5 ? '#000' : '#fff'} className="font-bold">
              {node.act.toFixed(2)}
            </text>
            
            <text x={node.x + 30} y={node.y} dominantBaseline="middle" textAnchor="start" fontSize="16" fill="#fff" className="font-bold font-sans">
              {node.label}
            </text>
            <rect x={node.x + 55} y={node.y - 4} width="70" height="8" rx="4" fill="#27272a" />
            <rect x={node.x + 55} y={node.y - 4} width={Math.max(0, node.act * 70)} height="8" rx="4" fill={node.act > 0.5 ? '#10b981' : '#a855f7'} className="transition-all duration-300" />
            
            <text x={node.x + 135} y={node.y} dominantBaseline="middle" textAnchor="start" fontSize="12" fill={node.act > 0.5 ? '#10b981' : '#a1a1aa'}>
              {(node.act * 100).toFixed(1)}%
            </text>

            {/* 8 Hidden -> Output Weights */}
            <g transform={`translate(${node.x + 55}, ${node.y + 10})`}>
              <rect x="-2" y="-2" width="116" height="16" rx="2" fill="#18181b" stroke="#27272a" />
              {nn.w2[i].map((w, idx) => (
                <g key={`out-w-${i}-${idx}`} transform={`translate(${idx * 14}, 0)`}>
                   <rect width={14} height={12} fill={w > 0 ? `rgba(239, 68, 68, ${Math.min(1, Math.abs(w))})` : `rgba(59, 130, 246, ${Math.min(1, Math.abs(w))})`} />
                   <text x="7" y="6" fontSize="4.5" fill="#fff" textAnchor="middle" dominantBaseline="middle">
                     {w.toFixed(1)}
                   </text>
                </g>
              ))}
            </g>
          </g>
        ))}
      </svg>
    </div>
  );
}
