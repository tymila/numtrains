import React, { useState, useEffect, useRef } from 'react';
import { initNetwork, forward, backward, update, NetworkState, ForwardResult } from './lib/nn';
import { DATASET } from './lib/dataset';
import NetworkVisualization from './components/NetworkVisualization';
import { Play, Square, FastForward, Eraser, PenTool, LayoutGrid, RotateCcw, BoxSelect, SkipForward } from 'lucide-react';

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export default function App() {
  const [nn, setNN] = useState<NetworkState>(initNetwork());
  const [currentInput, setCurrentInput] = useState<number[]>(DATASET[0].x);
  const [fwd, setFwd] = useState<ForwardResult>(forward(nn, DATASET[0].x));
  
  const [isAutoTraining, setIsAutoTraining] = useState(false);
  const [animSpeed, setAnimSpeed] = useState<'slow' | 'fast'>('fast');
  const [animStage, setAnimStage] = useState('idle');
  
  const [epoch, setEpoch] = useState(0);
  const [losses, setLosses] = useState<number[]>([]);
  const [lr, setLr] = useState(0.1);

  // Stable refs for async training loop to access latest state
  const nnRef = useRef(nn);
  const fwdRef = useRef(fwd);
  const inputRef = useRef(currentInput);
  const lrRef = useRef(lr);
  const autoTrainRef = useRef(isAutoTraining);
  const animSpeedRef = useRef(animSpeed);

  useEffect(() => { nnRef.current = nn; }, [nn]);
  useEffect(() => { fwdRef.current = fwd; }, [fwd]);
  useEffect(() => { inputRef.current = currentInput; }, [currentInput]);
  useEffect(() => { lrRef.current = lr; }, [lr]);
  useEffect(() => { autoTrainRef.current = isAutoTraining; }, [isAutoTraining]);
  useEffect(() => { animSpeedRef.current = animSpeed; }, [animSpeed]);

  // Real-time inference visual update when the user draws or network changes (and is idle)
  useEffect(() => {
    if (animStage === 'idle') {
      const liveFwd = forward(nn, currentInput);
      setFwd(liveFwd);
      fwdRef.current = liveFwd;
    }
  }, [currentInput, nn, animStage]);

  const runTrainingStep = async (stepFast: boolean = false) => {
    const isSlow = !stepFast;
    if (isSlow) {
      // 1. Pick exactly one sample
      const sample = DATASET[Math.floor(Math.random() * DATASET.length)];
      
      // Animate Forward Propagation
      const nextFwd = forward(nnRef.current, sample.x);
      setFwd(nextFwd);
      
      setAnimStage('inputs');
      await sleep(150);
      setAnimStage('forward_1');
      await sleep(150);
      setAnimStage('hidden');
      await sleep(150);
      setAnimStage('forward_2');
      await sleep(150);
      setAnimStage('output');
      await sleep(300); // User interprets output
      
      if (!autoTrainRef.current && isAutoTraining) return; // Prevent race conditions
      
      // Animate Backpropagation and Learning
      const grads = backward(nnRef.current, fwdRef.current, sample.y);
      setAnimStage('backward_2'); 
      await sleep(150);
      setAnimStage('backward_1'); 
      await sleep(200);
      
      // Update Weights
      const newNN = update(nnRef.current, grads, lrRef.current);
      nnRef.current = newNN;
      setNN(newNN);
      setLosses(p => [...p.slice(-99), grads.loss]);
      setEpoch(e => e + 1);
      setAnimStage('idle');
      await sleep(150); // Small rest before next loop

    } else {
      // Fast mode! Process a batch to show loss dropping quickly without UI blocking
      let currentNN = nnRef.current;
      let lastLoss = 0;
      for (let i = 0; i < 20; i++) {
        const s = DATASET[Math.floor(Math.random() * DATASET.length)];
        const f = forward(currentNN, s.x);
        const g = backward(currentNN, f, s.y);
        currentNN = update(currentNN, g, lrRef.current);
        lastLoss = g.loss;
      }
      nnRef.current = currentNN;
      setNN(currentNN);
      setLosses(p => [...p.slice(-80), lastLoss]);
      setEpoch(e => e + 20);
      setFwd(forward(currentNN, inputRef.current));
      await sleep(60); // Yield to render frame
    }
  };

  // The training loop effect
  useEffect(() => {
    let active = true;
    const loop = async () => {
      while (active && autoTrainRef.current) {
        await runTrainingStep(animSpeedRef.current === 'fast');
      }
    };
    if (isAutoTraining) {
      setAnimStage('idle');
      loop();
    }
    return () => { active = false; };
  }, [isAutoTraining]);

  const togglePixel = (i: number) => {
    const newV = [...currentInput];
    newV[i] = newV[i] === 1 ? 0 : 1;
    setCurrentInput(newV);
  };

  const addNoise = () => {
    setCurrentInput(prev => prev.map(v => Math.random() > 0.85 ? (v === 1 ? 0 : 1) : v));
  };

  // Compute Loss Path
  const maxLoss = Math.max(...losses, 0.5);
  const lossPoints = losses.map((l, i) => {
    const x = losses.length > 1 ? (i / (losses.length - 1)) * 300 : 0;
    return `${x},${80 - (l / maxLoss) * 80}`; // padding bottom
  }).join(" ");

  const predictedDigit = fwd.a2.indexOf(Math.max(...fwd.a2));
  const maxProb = fwd.a2[predictedDigit];

  return (
    <div className="min-h-screen bg-zinc-950 text-white font-sans flex flex-col selection:bg-purple-500/30">
      <header className="p-4 border-b border-zinc-800 bg-zinc-950 flex items-center justify-between z-10 shrink-0 shadow-sm">
        <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-3">
          <div className="bg-purple-500/20 text-purple-400 p-2 rounded-lg">
            <LayoutGrid className="w-5 h-5" />
          </div>
          神经网络可视化学习系统
        </h1>
        <div className="text-sm font-medium bg-zinc-900 px-4 py-2 rounded-full border border-zinc-800 text-zinc-300">
          目标：识别数字 <span className="font-bold text-white">0 - 9</span>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel: Drawing Input */}
        <div className="w-80 border-r border-zinc-800 p-6 flex flex-col gap-8 bg-zinc-950/80 overflow-y-auto shrink-0 shadow-[4px_0_24px_-10px_rgba(0,0,0,0.5)] z-0 text-sm">
          
          <div>
            <div className="text-zinc-500 font-medium lowercase tracking-widest uppercase mb-4 flex items-center gap-2 text-xs">
              <PenTool className="w-4 h-4" /> 画布
            </div>
            
            <div className="flex justify-center mb-6">
              <div className="grid grid-cols-5 gap-1.5 p-3 rounded-xl bg-zinc-900 border border-zinc-800 shadow-inner">
                {currentInput.map((val, i) => (
                  <div 
                    key={i} 
                    onClick={() => togglePixel(i)}
                    className={`w-10 h-10 rounded shadow-sm hover:scale-105 active:scale-95 transition-all cursor-pointer ${
                      val === 1 
                        ? 'bg-white shadow-[0_0_10px_rgba(255,255,255,0.2)]' 
                        : 'bg-zinc-950 border border-zinc-800'
                    }`}
                  />
                ))}
              </div>
            </div>

            <div className="flex flex-wrap gap-2 mb-6">
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(num => (
                <button key={num} onClick={() => setCurrentInput(DATASET[num].x)} className="flex-1 min-w-[30%] py-1.5 px-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded font-medium text-xs">
                  {num}
                </button>
              ))}
            </div>

            <div className="grid grid-cols-2 gap-3 mb-6">
              <button onClick={addNoise} className="flex items-center justify-center gap-2 py-2 px-3 bg-zinc-800 hover:bg-zinc-700 text-zinc-100 rounded-lg transition-colors font-medium">
                <BoxSelect className="w-4 h-4 text-zinc-400" /> 添加噪声
              </button>
              <button onClick={() => setCurrentInput(Array(25).fill(0))} className="flex items-center justify-center gap-2 py-2 px-3 bg-zinc-800 hover:bg-zinc-700 text-zinc-100 rounded-lg transition-colors font-medium">
                <Eraser className="w-4 h-4 text-zinc-400" /> 清空
              </button>
            </div>
          </div>

          <div className="p-4 bg-purple-500/10 border border-purple-500/20 rounded-xl">
             <h3 className="font-semibold text-purple-300 mb-2">工作原理</h3>
             <p className="text-zinc-400 text-xs leading-relaxed">
               像素点亮<strong className="text-zinc-200">输入层</strong>。
               连接（权重）缩小或放大信号。
               <strong className="text-zinc-200">隐藏层</strong>逐渐形成空间特征偏好（观察热力图！）。
             </p>
          </div>
        </div>

        {/* Center Panel: Visualization */}
        <div className="flex-1 relative bg-zinc-950 flex flex-col items-center justify-center">
           {animStage !== 'idle' && (
             <div className="absolute top-6 left-1/2 -translate-x-1/2 bg-purple-500 text-white px-4 py-1.5 rounded-full text-xs font-bold tracking-widest uppercase animate-pulse shadow-[0_0_15px_rgba(168,85,247,0.5)]">
                {animStage.includes('forward') || animStage === 'hidden' || animStage === 'inputs' || animStage === 'output' 
                  ? '前向传播中...' 
                  : '反向传播误差中...'}
             </div>
           )}
           <NetworkVisualization nn={nn} fwd={fwd} animStage={animStage} />
        </div>

        {/* Right Panel: Output & Controls */}
        <div className="w-80 border-l border-zinc-800 p-6 flex flex-col gap-6 bg-zinc-950/80 overflow-y-auto shrink-0 z-0">
          
          <div className="bg-zinc-900 rounded-2xl p-6 shadow-inner border border-zinc-800 text-center relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-purple-500 to-transparent opacity-50" />
            <div className="text-zinc-500 text-xs font-semibold tracking-widest uppercase">预测结果</div>
            <div className="text-6xl font-sans font-bold mt-4 mb-2 tracking-tighter text-white">
              {predictedDigit}
            </div>
            <div className={`text-sm font-medium transition-colors ${maxProb > 0.5 ? 'text-emerald-400' : 'text-zinc-500'}`}>
              置信度: {(maxProb * 100).toFixed(1)}%
            </div>
          </div>

          <div className="flex flex-col gap-3">
            <div className="text-zinc-500 font-medium tracking-widest uppercase text-xs mb-2">训练引擎</div>
            
            <div className="flex gap-2">
              <button 
                onClick={() => setIsAutoTraining(!isAutoTraining)}
                className={`flex-1 py-3 px-4 rounded-xl font-bold flex justify-center items-center gap-2 transition-all shadow-sm
                  ${isAutoTraining 
                    ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30 hover:bg-rose-500/30' 
                    : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/30'}`
                }
              >
                {isAutoTraining ? <><Square className="w-4 h-4 fill-current"/> 停止训练</> : <><Play className="w-4 h-4 fill-current"/> 自动训练</>}
              </button>
              
              <button 
                disabled={isAutoTraining}
                onClick={() => { setAnimSpeed('slow'); runTrainingStep(false); }}
                className="w-12 flex justify-center items-center bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-xl transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                title="单步慢速"
              >
                <SkipForward className="w-4 h-4" />
              </button>
            </div>

            <button 
              onClick={() => { setNN(initNetwork()); setEpoch(0); setLosses([]); }}
              className="py-2.5 px-4 bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 border border-zinc-800 rounded-xl transition-colors flex items-center justify-center gap-2 text-sm font-medium"
            >
              <RotateCcw className="w-4 h-4" /> 重置权重
            </button>
          </div>
          
          <div className="bg-zinc-900 border border-zinc-800 p-4 rounded-xl flex items-center justify-between">
            <div className="text-zinc-400 text-sm font-medium flex items-center gap-2">
              模式: <span className={animSpeed === 'slow' ? 'text-purple-400' : 'text-blue-400'}>{animSpeed === 'slow' ? '慢速详解' : '快速刷轮'}</span>
            </div>
            <button 
              onClick={() => setAnimSpeed(s => s === 'slow' ? 'fast' : 'slow')}
              className="text-xs bg-zinc-800 hover:bg-zinc-700 px-3 py-1.5 rounded border border-zinc-700 text-zinc-300 transition-colors flex items-center gap-1"
            >
              <FastForward className="w-3 h-3" /> 切换
            </button>
          </div>

          <div className="pt-2 border-t border-zinc-800 mt-2 flex flex-col gap-4">
             <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-zinc-400 text-xs font-semibold uppercase tracking-widest">学习率 (LR)</label>
                  <span className="text-zinc-300 font-mono text-xs">{lr.toFixed(3)}</span>
                </div>
                <input 
                  type="range" min="0.01" max="1" step="0.01" 
                  value={lr} onChange={(e) => setLr(parseFloat(e.target.value))} 
                  className="w-full h-1 bg-zinc-800 rounded-full appearance-none cursor-pointer accent-purple-500"
                />
             </div>

             <div>
               <div className="flex justify-between items-end mb-2">
                 <div className="text-zinc-400 text-xs font-semibold uppercase tracking-widest">Loss 曲线</div>
                 <div className="text-purple-400 font-mono text-sm font-bold bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/20">第 {epoch} 轮</div>
               </div>
               
               <div className="h-28 w-full bg-[#0a0a0c] border border-zinc-800/50 rounded-lg overflow-hidden flex items-end pt-4 opacity-90 relative">
                  <div className="absolute inset-0 bg-[linear-gradient(to_bottom,transparent_0%,rgba(168,85,247,0.05)_100%)] pointer-events-none" />
                  <svg viewBox="0 0 300 80" preserveAspectRatio="none" className="w-full h-full">
                     <polyline points={lossPoints} fill="none" stroke="#a855f7" strokeWidth="2" vectorEffect="non-scaling-stroke" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
               </div>
             </div>
          </div>
          
        </div>
      </div>
    </div>
  );
}
