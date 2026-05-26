export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export const softmax = (arr: number[]) => {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
};

export type NetworkState = {
  w1: number[][]; // [8 hidden][25 inputs]
  b1: number[];   // [8 hidden]
  w2: number[][]; // [10 outputs][8 hidden]
  b2: number[];   // [10 outputs]
};

export function initNetwork(): NetworkState {
  const numHidden = 8;
  const numOutput = 10;
  // Initialize with small random weights
  const rand = () => (Math.random() - 0.5) * 1.5;
  return {
    w1: Array.from({ length: numHidden }, () => Array.from({ length: 25 }, rand)),
    b1: Array.from({ length: numHidden }, rand),
    w2: Array.from({ length: numOutput }, () => Array.from({ length: numHidden }, rand)),
    b2: Array.from({ length: numOutput }, rand),
  };
}

export type ForwardResult = {
  input: number[];
  z1: number[];
  a1: number[];
  z2: number[];
  a2: number[];
};

export function forward(nn: NetworkState, input: number[]): ForwardResult {
  const z1 = nn.w1.map((row, i) => row.reduce((sum, w, j) => sum + w * input[j], 0) + nn.b1[i]);
  const a1 = z1.map(sigmoid);
  
  const z2 = nn.w2.map((row, i) => row.reduce((sum, w, j) => sum + w * a1[j], 0) + nn.b2[i]);
  const a2 = softmax(z2);
  
  return { input, z1, a1, z2, a2 };
}

export function backward(nn: NetworkState, fwd: ForwardResult, target: number[]) {
  // Cross entropy loss
  const loss = -target.reduce((sum, t, i) => sum + t * Math.log(fwd.a2[i] + 1e-8), 0);
  
  // Output layer error signal for Softmax + Cross Entropy is simply (a2 - y)
  const dz2 = fwd.a2.map((a, i) => a - target[i]);
  
  // Gradients for w2 and b2
  const dw2 = dz2.map(d => fwd.a1.map(a => d * a));
  const db2 = dz2;
  
  // Hidden layer error signal
  const da1 = fwd.a1.map((_, j) => dz2.reduce((sum, d, i) => sum + nn.w2[i][j] * d, 0));
  const dz1 = da1.map((d, i) => d * (fwd.a1[i] * (1 - fwd.a1[i])));
  
  // Gradients for w1 and b1
  const dw1 = nn.w1.map((row, i) => row.map((w, j) => dz1[i] * fwd.input[j]));
  const db1 = dz1;
  
  return { dw1, db1, dw2, db2, loss };
}

export function update(nn: NetworkState, grads: any, lr: number): NetworkState {
  return {
    w1: nn.w1.map((row, i) => row.map((w, j) => w - lr * grads.dw1[i][j])),
    b1: nn.b1.map((b, i) => b - lr * grads.db1[i]),
    w2: nn.w2.map((row, i) => row.map((w, j) => w - lr * grads.dw2[i][j])),
    b2: nn.b2.map((b, i) => b - lr * grads.db2[i]),
  };
}
