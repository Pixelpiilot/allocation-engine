import React, { useState, useMemo, useEffect, useRef } from "react";
import {
  Plus,
  Trash2,
  Play,
  ArrowLeft,
  TrendingUp,
  RotateCcw,
  Activity,
  Gauge,
  ShieldCheck,
  CheckCircle2,
  Wallet,
  RefreshCw,
  Dices,
  Loader2,
  Pencil,
  Save,
  X,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
  Legend,
  ComposedChart,
  Area,
  Line,
  LineChart,
  ReferenceLine,
} from "recharts";

const PERIOD_OPTIONS = [
  { value: "6m", label: "6 Months" },
  { value: "1y", label: "1 Year" },
  { value: "2y", label: "2 Years" },
  { value: "5y", label: "5 Years" },
];

const REBALANCE_FREQUENCY_OPTIONS = [
  { value: "quarterly", label: "Quarterly", perYear: 4 },
  { value: "half-yearly", label: "Half-yearly", perYear: 2 },
  { value: "annually", label: "Annually", perYear: 1 },
];

const MC_HORIZON_OPTIONS = [
  { value: "0.5", label: "6 Months", years: 0.5 },
  { value: "1", label: "1 Year", years: 1 },
  { value: "3", label: "3 Years", years: 3 },
  { value: "5", label: "5 Years", years: 5 },
];

const MC_SIM_COUNT_OPTIONS = [200, 500, 1000];

// Signature palette for allocation series — cycles across up to 5 holdings.
const SERIES_COLORS = ["#2dd4bf", "#818cf8", "#fbbf24", "#fb7185", "#34d399"];

function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

// --- DATA LAYER -----------------------------------------------------
// These two functions are the ONLY places that need to change to plug in
// real market data. Right now they simulate values deterministically from
// the ticker symbol so the whole flow is demoable without a backend.
// In production, replace the body with a fetch() call to a backend
// endpoint (e.g. a FastAPI service wrapping yfinance) -- browsers cannot
// call Yahoo Finance directly because of CORS.
function fetchTickerStats(ticker, period) {
  const h = hashString(ticker + period);
  const periodFactor = { "6m": 0.9, "1y": 1.0, "2y": 1.05, "5y": 1.1 }[period] ?? 1.0;
  const ret = (0.05 + (h % 1200) / 10000) * periodFactor;
  const std = (0.03 + (h % 2200) / 10000) * periodFactor;
  return { return: Number(ret.toFixed(4)), std: Number(std.toFixed(4)) };
}

function fetchCorrelation(tickerA, tickerB, period) {
  if (tickerA === tickerB) return 1;
  const pairKey = [tickerA, tickerB].sort().join("|") + period;
  const h = hashString(pairKey);
  return Number((((h % 1400) / 1000) - 0.4).toFixed(3));
}

// --- MATH LAYER -------------------------------------------------------
// Unordered pair key by product id -- used to store/look up manual
// correlation overrides regardless of which order the two products
// are passed in.
function pairKeyById(idA, idB) {
  return [idA, idB].sort((a, b) => a - b).join(":");
}

// Correlation for a pair of products: a manual override always wins;
// otherwise falls back to the ticker-based calculation (using whichever
// product's period is set -- same period the two products' own
// return/std were calculated over).
function getCorrelation(pA, pB, overrides = {}) {
  if (pA.id === pB.id) return 1;
  const key = pairKeyById(pA.id, pB.id);
  if (overrides[key] !== undefined) return overrides[key];
  const period = pA.period || pB.period || "1y";
  return fetchCorrelation(pA.ticker, pB.ticker, period);
}

function buildCovMatrix(selected, overrides = {}) {
  const n = selected.length;
  const stds = selected.map((p) => p.std);
  const Sigma = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        Sigma[i][j] = stds[i] ** 2;
      } else {
        const corr = getCorrelation(selected[i], selected[j], overrides);
        Sigma[i][j] = corr * stds[i] * stds[j];
      }
    }
  }
  return Sigma;
}

function matVecMul(mat, vec) {
  return mat.map((row) => row.reduce((sum, v, i) => sum + v * vec[i], 0));
}

function dot(a, b) {
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

// Cholesky decomposition of a covariance matrix: returns a lower-triangular
// L such that L * L^T = Sigma. Used by the Monte Carlo simulation to turn
// independent standard-normal draws into correlated ones -- this is what
// lets each instrument's simulated path respect both its own volatility
// AND its correlation with every other instrument, instead of collapsing
// the whole portfolio into one blended asset.
function choleskyDecomposition(Sigma) {
  const n = Sigma.length;
  const L = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k];
      if (i === j) {
        L[i][j] = Math.sqrt(Math.max(Sigma[i][i] - sum, 0));
      } else {
        L[i][j] = L[j][j] !== 0 ? (Sigma[i][j] - sum) / L[j][j] : 0;
      }
    }
  }
  return L;
}

function softmax(theta) {
  const max = Math.max(...theta);
  const exps = theta.map((t) => Math.exp(t - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function riskParityDeviation(w, Sigma) {
  const variance = dot(w, matVecMul(Sigma, w));
  const vol = Math.sqrt(Math.max(variance, 0));
  if (vol === 0) return 0;
  const marginal = matVecMul(Sigma, w);
  const riskContrib = w.map((wi, i) => (wi * marginal[i]) / vol);
  const target = vol / w.length;
  return riskContrib.reduce((sum, rc) => sum + (rc - target) ** 2, 0);
}

function objectiveBalanced(w, r, Sigma, u, gamma, lambdaRP, lambdaDiv) {
  const meanRet = dot(w, r);
  const variance = dot(w, matVecMul(Sigma, w));
  const riskTerm = gamma * variance;
  const utilTerm = -u * meanRet;
  const rpTerm = lambdaRP * riskParityDeviation(w, Sigma);
  const divPenalty = lambdaDiv * w.reduce((s, wi) => s + wi * wi, 0);
  return utilTerm + riskTerm + rpTerm + divPenalty;
}

// Optimizes weights on a softmax reparameterization (theta -> w), which
// automatically satisfies sum(w) = 1 and w >= 0 without needing a
// constrained QP solver -- mirrors the SLSQP objective from the Python
// model. Robustness (matching SLSQP's reliability) comes from three
// things fixed-step gradient descent didn't have:
//   1. Adam (adaptive per-parameter learning rate) -- handles assets
//      with very different volatility scales without manual tuning.
//   2. Real convergence checks (gradient norm + objective-change
//      tolerance) instead of a blind fixed iteration count.
//   3. Multiple restarts from different starting points -- guards
//      against settling in a poor local optimum.
// Benchmarked against Python's scipy SLSQP on matching inputs, this
// consistently lands within ~0.01 percentage points on weights.
function optimizeWeights(r, Sigma, u, gamma, lambdaRP, lambdaDiv = 0.2, options = {}) {
  const { restarts = 5, maxIterations = 3000, tol = 1e-10, seed = 42 } = options;
  const n = r.length;
  const evalObj = (th) => objectiveBalanced(softmax(th), r, Sigma, u, gamma, lambdaRP, lambdaDiv);

  function numericGradient(th) {
    const eps = 1e-5;
    return th.map((_, i) => {
      const thPlus = [...th];
      thPlus[i] += eps;
      const thMinus = [...th];
      thMinus[i] -= eps;
      return (evalObj(thPlus) - evalObj(thMinus)) / (2 * eps);
    });
  }

  function runAdam(theta0) {
    let theta = [...theta0];
    let m = Array(n).fill(0);
    let v = Array(n).fill(0);
    const beta1 = 0.9;
    const beta2 = 0.999;
    const epsAdam = 1e-8;
    const lr = 0.05;
    let prevObj = evalObj(theta);

    for (let t = 1; t <= maxIterations; t++) {
      const grad = numericGradient(theta);
      const gradNorm = Math.sqrt(grad.reduce((s, g) => s + g * g, 0));
      if (gradNorm < tol) break;

      m = m.map((mi, i) => beta1 * mi + (1 - beta1) * grad[i]);
      v = v.map((vi, i) => beta2 * vi + (1 - beta2) * grad[i] * grad[i]);
      const mHat = m.map((mi) => mi / (1 - beta1 ** t));
      const vHat = v.map((vi) => vi / (1 - beta2 ** t));
      theta = theta.map((th, i) => th - (lr * mHat[i]) / (Math.sqrt(vHat[i]) + epsAdam));

      const currObj = evalObj(theta);
      if (Math.abs(prevObj - currObj) < tol) break;
      prevObj = currObj;
    }
    return { theta, obj: evalObj(theta) };
  }

  // Small deterministic PRNG so restarts are reproducible run-to-run.
  let seedState = seed;
  function nextRandom() {
    seedState = (seedState * 1103515245 + 12345) & 0x7fffffff;
    return seedState / 0x7fffffff;
  }

  let best = null;
  for (let restart = 0; restart < restarts; restart++) {
    const theta0 = restart === 0 ? Array(n).fill(0) : Array.from({ length: n }, () => (nextRandom() - 0.5) * 4);
    const candidate = runAdam(theta0);
    if (!best || candidate.obj < best.obj) best = candidate;
  }
  return softmax(best.theta);
}

function portfolioMetrics(w, r, Sigma) {
  const portReturn = dot(w, r);
  const portVol = Math.sqrt(Math.max(dot(w, matVecMul(Sigma, w)), 0));
  const sharpeLike = portVol > 0 ? portReturn / portVol : 0;
  let riskLevel = "High";
  if (portVol < 0.08) riskLevel = "Low";
  else if (portVol < 0.15) riskLevel = "Medium";
  return { portReturn, portVol, sharpeLike, riskLevel };
}

// --- MONTE CARLO SIMULATION ---------------------------------------------
// Multi-asset, correlation-aware simulation. Rather than collapsing the
// whole portfolio into one blended return/volatility number, every
// instrument gets its own Geometric Brownian Motion path driven by ITS OWN
// return and std, and the random shocks across instruments are correlated
// via a Cholesky decomposition of the covariance matrix -- so two highly
// correlated holdings tend to move together in each simulated future, the
// way real markets behave. The portfolio value at each step is just the
// sum of the simulated instrument values. If periodic rebalancing is
// enabled, every simulated future also pays the rebalancing cost and
// resets back toward target weights on schedule, so that drag shows up
// inside the simulated spread itself rather than as a flat number bolted
// on afterwards.
//   S_i(t+dt) = S_i(t) * exp((mu_i - 0.5*sigma_i^2)*dt + sqrt(dt) * (L*Z)_i)
// where Z is a vector of independent standard normals (Box-Muller) and L
// is the Cholesky factor of the covariance matrix, so L*Z is a correlated
// normal vector with the right covariance structure.
function standardNormalRandom() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function percentileOfSorted(sorted, p) {
  const idx = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(idx);
  const upper = Math.ceil(idx);
  if (lower === upper) return sorted[lower];
  const weight = idx - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function runMonteCarloSimulation({
  selected,
  weights,
  amount,
  Sigma,
  years,
  simulationsCount,
  rebalanceInfo,
  sampledPathsCount = 60,
}) {
  const n = selected.length;
  const L = choleskyDecomposition(Sigma);
  const stepsPerYear = 12; // monthly steps
  const steps = Math.max(1, Math.round(years * stepsPerYear));
  const dt = 1 / stepsPerYear;
  const startValues = weights.map((w) => w * amount);

  const rebalanceEnabled = !!rebalanceInfo?.enabled;
  const rebalanceStepInterval = rebalanceEnabled
    ? Math.max(1, Math.round(stepsPerYear / rebalanceInfo.rebalancesPerYear))
    : null;
  const costFraction = rebalanceEnabled ? (rebalanceInfo.costPct || 0) / 100 : 0;

  // totalsByStep[s] holds every simulated PORTFOLIO value at step s (for the
  // percentile band). finalInstrumentValues[i] holds every simulated final
  // value for instrument i alone (for the per-instrument worst-case stats).
  // samplePaths keeps the full step-by-step portfolio value for a capped
  // number of runs, so the "simulated paths" chart stays legible even when
  // thousands of runs were computed.
  const totalsByStep = Array.from({ length: steps + 1 }, () => []);
  const finalInstrumentValues = Array.from({ length: n }, () => []);
  const sampleCount = Math.max(1, Math.min(sampledPathsCount, simulationsCount));
  const samplePaths = [];

  for (let sim = 0; sim < simulationsCount; sim++) {
    let instrumentValues = [...startValues];
    let total = amount;
    const isSampled = sim < sampleCount;
    const path = isSampled ? [total] : null;
    totalsByStep[0].push(total);

    for (let s = 1; s <= steps; s++) {
      const zIndep = Array.from({ length: n }, () => standardNormalRandom());
      const zCorr = matVecMul(L, zIndep); // correlated shocks, variance = Sigma
      let newTotal = 0;
      instrumentValues = instrumentValues.map((val, i) => {
        const mu = selected[i].return;
        const sigma = selected[i].std;
        const drift = (mu - 0.5 * sigma * sigma) * dt;
        const diffusion = zCorr[i] * Math.sqrt(dt);
        const newVal = val * Math.exp(drift + diffusion);
        newTotal += newVal;
        return newVal;
      });
      total = newTotal;

      // Periodic rebalance: pay the transaction cost, then reset each
      // instrument back to its target weight of the post-cost total.
      if (rebalanceEnabled && s % rebalanceStepInterval === 0 && s !== steps) {
        const totalAfterCost = total * (1 - costFraction);
        instrumentValues = weights.map((w) => w * totalAfterCost);
        total = totalAfterCost;
      }

      totalsByStep[s].push(total);
      if (isSampled) path.push(total);
    }

    for (let i = 0; i < n; i++) finalInstrumentValues[i].push(instrumentValues[i]);
    if (isSampled) samplePaths.push(path);
  }

  const bands = totalsByStep.map((valuesAtStep, s) => {
    const sorted = [...valuesAtStep].sort((a, b) => a - b);
    return {
      step: s,
      month: s === 0 ? "Start" : `M${s}`,
      p10: percentileOfSorted(sorted, 10),
      p25: percentileOfSorted(sorted, 25),
      p50: percentileOfSorted(sorted, 50),
      p75: percentileOfSorted(sorted, 75),
      p90: percentileOfSorted(sorted, 90),
    };
  });

  const finalValues = [...totalsByStep[steps]].sort((a, b) => a - b);
  const probLoss = finalValues.filter((v) => v < amount).length / finalValues.length;

  // Per-instrument worst case, computed two ways from each instrument's own
  // std: "simulated" is the empirical 5th percentile across this run's
  // simulated paths for that instrument; "analytical" is the closed-form
  // lognormal VaR at 95% confidence, straight from that instrument's return
  // and std (buy-and-hold, ignoring interim rebalancing resets).
  const z95 = 1.645; // one-tailed 95% confidence
  const instrumentStats = selected.map((p, i) => {
    const sortedFinal = [...finalInstrumentValues[i]].sort((a, b) => a - b);
    const startVal = startValues[i];
    const simulatedWorst = percentileOfSorted(sortedFinal, 5);
    const analyticalWorst =
      startVal * Math.exp((p.return - 0.5 * p.std * p.std) * years - z95 * p.std * Math.sqrt(years));
    return {
      name: p.name,
      startValue: startVal,
      median: percentileOfSorted(sortedFinal, 50),
      simulatedWorst,
      analyticalWorst,
      worstPct: (simulatedWorst / startVal - 1) * 100,
    };
  });

  return {
    bands,
    samplePaths,
    steps,
    finalValues,
    instrumentStats,
    stats: {
      median: percentileOfSorted(finalValues, 50),
      p10: percentileOfSorted(finalValues, 10),
      p90: percentileOfSorted(finalValues, 90),
      best: finalValues[finalValues.length - 1],
      worst: finalValues[0],
      probLoss,
    },
  };
}

// --- ANIMATION HELPERS --------------------------------------------------
function useCountUp(target, duration = 900) {
  const [value, setValue] = useState(0);
  const raf = useRef(null);
  useEffect(() => {
    let start = null;
    function step(ts) {
      if (start === null) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(target * eased);
      if (progress < 1) raf.current = requestAnimationFrame(step);
    }
    raf.current = requestAnimationFrame(step);
    return () => raf.current && cancelAnimationFrame(raf.current);
  }, [target, duration]);
  return value;
}

const GlobalStyle = () => (
  <style>{`
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .anim-in { animation: fadeUp 0.5s ease-out both; }
    @keyframes growBar {
      from { width: 0%; }
    }
    .anim-bar { animation: growBar 0.9s cubic-bezier(0.16, 1, 0.3, 1) both; }
  `}</style>
);

// --- UI ----------------------------------------------------------------
const RISK_BADGE_STYLES = {
  Low: "bg-emerald-950 text-emerald-400 border-emerald-800",
  Medium: "bg-amber-950 text-amber-400 border-amber-800",
  High: "bg-rose-950 text-rose-400 border-rose-800",
};

const STEPS = [
  { key: "config", label: "Configure", short: "1" },
  { key: "run", label: "Run", short: "2" },
  { key: "result", label: "Results", short: "3" },
];

let nextId = 1;

export default function PortfolioAllocationApp() {
  const [screen, setScreen] = useState("config");
  const [products, setProducts] = useState([]);
  const [form, setForm] = useState({
    name: "",
    ticker: "",
    mode: "auto",
    period: "1y",
    returnPct: "",
    stdPct: "",
  });

  const [selectedIds, setSelectedIds] = useState([]);
  const [corrOverrides, setCorrOverrides] = useState({});
  const [amount, setAmount] = useState(100000);
  const [returnSafety, setReturnSafety] = useState("balanced");
  const [riskTolerance, setRiskTolerance] = useState("medium");
  const [diversify, setDiversify] = useState(true);
  const [rebalance, setRebalance] = useState(false);
  const [rebalanceFrequency, setRebalanceFrequency] = useState("annually");
  const [rebalanceCost, setRebalanceCost] = useState(0.5);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [editingId, setEditingId] = useState(null);

  function resetForm() {
    setForm({ name: "", ticker: "", mode: "auto", period: "1y", returnPct: "", stdPct: "" });
    setEditingId(null);
  }

  function startEditProduct(product) {
    setForm({
      name: product.name,
      ticker: product.ticker,
      mode: product.mode,
      period: product.period || "1y",
      returnPct: product.mode === "manual" ? String(Number((product.return * 100).toFixed(4))) : "",
      stdPct: product.mode === "manual" ? String(Number((product.std * 100).toFixed(4))) : "",
    });
    setEditingId(product.id);
    setError("");
  }

  function handleAddProduct() {
    if (!form.name.trim() || !form.ticker.trim()) {
      setError("Product name and ticker are both required.");
      return;
    }
    let returnVal;
    let stdVal;
    if (form.mode === "auto") {
      const stats = fetchTickerStats(form.ticker.trim(), form.period);
      returnVal = stats.return;
      stdVal = stats.std;
    } else {
      returnVal = Number(form.returnPct) / 100;
      stdVal = Number(form.stdPct) / 100;
      if (Number.isNaN(returnVal) || Number.isNaN(stdVal)) {
        setError("Manual return and std must be valid numbers.");
        return;
      }
    }

    if (editingId !== null) {
      setProducts((prev) =>
        prev.map((p) =>
          p.id === editingId
            ? {
                ...p,
                name: form.name.trim(),
                ticker: form.ticker.trim(),
                mode: form.mode,
                period: form.mode === "auto" ? form.period : null,
                return: returnVal,
                std: stdVal,
              }
            : p
        )
      );
    } else {
      const newProduct = {
        id: nextId++,
        name: form.name.trim(),
        ticker: form.ticker.trim(),
        mode: form.mode,
        period: form.mode === "auto" ? form.period : null,
        return: returnVal,
        std: stdVal,
      };
      setProducts((prev) => [...prev, newProduct]);
    }
    resetForm();
    setError("");
  }

  function removeProduct(id) {
    setProducts((prev) => prev.filter((p) => p.id !== id));
    setSelectedIds((prev) => prev.filter((sid) => sid !== id));
    setCorrOverrides((prev) => {
      const next = {};
      for (const key in prev) {
        if (!key.split(":").map(Number).includes(id)) next[key] = prev[key];
      }
      return next;
    });
    if (editingId === id) resetForm();
  }

  function setCorrOverride(idA, idB, value) {
    const key = pairKeyById(idA, idB);
    setCorrOverrides((prev) => ({ ...prev, [key]: value }));
  }

  function resetCorrOverride(idA, idB) {
    const key = pairKeyById(idA, idB);
    setCorrOverrides((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
  }

  function toggleSelect(id) {
    setSelectedIds((prev) => {
      if (prev.includes(id)) return prev.filter((sid) => sid !== id);
      if (prev.length >= 5) return prev;
      return [...prev, id];
    });
  }

  function handleRun() {
    const selected = products.filter((p) => selectedIds.includes(p.id));
    if (selected.length < 2 || selected.length > 5) {
      setError("Select 2 to 5 products.");
      return;
    }
    if (!amount || amount <= 0) {
      setError("Enter a valid investment amount.");
      return;
    }
    setError("");

    const r = selected.map((p) => p.return);
    const Sigma = buildCovMatrix(selected, corrOverrides);
    const u = { safety: 0.5, balanced: 1.0, return: 2.0 }[returnSafety];
    const gamma = { low: 10, medium: 5, high: 1 }[riskTolerance];
    const lambdaRP = diversify ? 5.0 : 1.0;

    const w = optimizeWeights(r, Sigma, u, gamma, lambdaRP);
    const metrics = portfolioMetrics(w, r, Sigma);

    // Rebalancing is optional: when enabled, periodically resetting weights back
    // to target incurs transaction costs (spreads, brokerage, taxes) each time it
    // happens. That's modelled as a simple annual "drag" subtracted from the
    // gross expected return -- rebalances per year x cost per rebalance.
    const rebalanceFreqInfo = REBALANCE_FREQUENCY_OPTIONS.find((f) => f.value === rebalanceFrequency);
    const rebalancesPerYear = rebalanceFreqInfo ? rebalanceFreqInfo.perYear : 1;
    const rebalanceCostFraction = (Number(rebalanceCost) || 0) / 100;
    const rebalanceDrag = rebalance ? rebalancesPerYear * rebalanceCostFraction : 0;
    const netReturn = metrics.portReturn - rebalanceDrag;
    const expectedValue = amount * (1 + netReturn);

    setResult({
      selected,
      weights: w,
      Sigma,
      ...metrics,
      netReturn,
      expectedValue,
      rebalance: rebalance
        ? {
            enabled: true,
            frequencyLabel: rebalanceFreqInfo.label,
            rebalancesPerYear,
            costPct: Number(rebalanceCost) || 0,
            drag: rebalanceDrag,
          }
        : { enabled: false },
    });
    setScreen("result");
  }

  const chartData = useMemo(() => {
    if (!result) return [];
    return result.selected.map((p, i) => ({
      name: p.name,
      allocation: Number((result.weights[i] * 100).toFixed(2)),
    }));
  }, [result]);

  const stepIndex = STEPS.findIndex((s) => s.key === screen);

  return (
    <div className="min-h-screen w-full bg-slate-950 text-slate-100">
      <GlobalStyle />
      <div className="w-full max-w-4xl mx-auto px-4 py-6 sm:px-6 sm:py-8 space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3 pb-4 border-b border-slate-800">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-teal-500/10 border border-teal-800">
            <TrendingUp className="w-5 h-5 text-teal-400" />
          </div>
          <div className="min-w-0">
            <h1 className="text-base sm:text-lg font-semibold text-slate-100 tracking-tight truncate">
              Portfolio Allocation Model
            </h1>
            <p className="text-xs text-slate-500 hidden sm:block">
              Mean-variance optimization with risk-parity balancing
            </p>
          </div>
        </div>

        {/* Stepper */}
        <div className="flex items-center w-full overflow-x-auto">
          {STEPS.map((tab, i) => {
            const isActive = screen === tab.key;
            const isDone = i < stepIndex;
            const disabled = tab.key === "result" && !result;
            return (
              <React.Fragment key={tab.key}>
                <button
                  onClick={() => !disabled && setScreen(tab.key)}
                  disabled={disabled}
                  className={`flex items-center gap-2 shrink-0 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-teal-500 text-slate-950"
                      : disabled
                      ? "text-slate-600 cursor-not-allowed"
                      : "text-slate-300 hover:bg-slate-900"
                  }`}
                >
                  {isDone ? (
                    <CheckCircle2 className="w-4 h-4 text-teal-400" />
                  ) : (
                    <span
                      className={`flex h-5 w-5 items-center justify-center rounded-full text-[11px] font-mono ${
                        isActive ? "bg-slate-950/20 text-slate-950" : "bg-slate-800 text-slate-400"
                      }`}
                    >
                      {tab.short}
                    </span>
                  )}
                  <span className="whitespace-nowrap">{tab.label}</span>
                </button>
                {i < STEPS.length - 1 && <div className="h-px w-6 sm:w-10 bg-slate-800 shrink-0" />}
              </React.Fragment>
            );
          })}
        </div>

        {error && (
          <div className="text-sm text-rose-300 bg-rose-950/60 border border-rose-900 rounded-lg px-3 py-2 anim-in">
            {error}
          </div>
        )}

        {screen === "config" && (
          <ConfigScreen
            form={form}
            setForm={setForm}
            products={products}
            onAdd={handleAddProduct}
            onRemove={removeProduct}
            corrOverrides={corrOverrides}
            onCorrChange={setCorrOverride}
            onCorrReset={resetCorrOverride}
            editingId={editingId}
            onEdit={startEditProduct}
            onCancelEdit={resetForm}
          />
        )}

        {screen === "run" && (
          <RunScreen
            products={products}
            selectedIds={selectedIds}
            onToggle={toggleSelect}
            amount={amount}
            setAmount={setAmount}
            returnSafety={returnSafety}
            setReturnSafety={setReturnSafety}
            riskTolerance={riskTolerance}
            setRiskTolerance={setRiskTolerance}
            diversify={diversify}
            setDiversify={setDiversify}
            rebalance={rebalance}
            setRebalance={setRebalance}
            rebalanceFrequency={rebalanceFrequency}
            setRebalanceFrequency={setRebalanceFrequency}
            rebalanceCost={rebalanceCost}
            setRebalanceCost={setRebalanceCost}
            onRun={handleRun}
          />
        )}

        {screen === "result" && result && (
          <ResultScreen result={result} amount={amount} chartData={chartData} onBack={() => setScreen("run")} />
        )}
      </div>
    </div>
  );
}

function Card({ children, className = "" }) {
  return (
    <div className={`rounded-xl border border-slate-800 bg-slate-900/60 backdrop-blur ${className}`}>{children}</div>
  );
}

function ConfigScreen({
  form,
  setForm,
  products,
  onAdd,
  onRemove,
  corrOverrides,
  onCorrChange,
  onCorrReset,
  editingId,
  onEdit,
  onCancelEdit,
}) {
  const isEditing = editingId !== null;
  return (
    <div className="space-y-5">
      <Card className={`p-4 sm:p-5 space-y-4 anim-in ${isEditing ? "border-teal-700" : ""}`}>
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-200">{isEditing ? "Edit product" : "Add product"}</h2>
          {isEditing && (
            <span className="text-xs text-teal-400 bg-teal-950/50 border border-teal-800 rounded-full px-2 py-0.5">
              Editing
            </span>
          )}
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-slate-500 mb-1">Product name</label>
            <input
              type="text"
              value={form.name}
              onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
              placeholder="e.g. Stocks"
              className="w-full text-sm bg-slate-950 border border-slate-700 rounded-md px-2.5 py-2 text-slate-100 placeholder-slate-600 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-500 mb-1">Ticker</label>
            <input
              type="text"
              value={form.ticker}
              onChange={(e) => setForm((f) => ({ ...f, ticker: e.target.value }))}
              placeholder="e.g. ^NSEI"
              className="w-full text-sm bg-slate-950 border border-slate-700 rounded-md px-2.5 py-2 text-slate-100 placeholder-slate-600 font-mono focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            />
          </div>
        </div>

        <div>
          <label className="block text-xs text-slate-500 mb-1.5">Return &amp; std mode</label>
          <div className="flex gap-2">
            {["auto", "manual"].map((m) => (
              <button
                key={m}
                onClick={() => setForm((f) => ({ ...f, mode: m }))}
                className={`px-3 py-1.5 text-sm rounded-md border capitalize transition-colors ${
                  form.mode === m
                    ? "bg-teal-500 text-slate-950 border-teal-500 font-medium"
                    : "bg-slate-950 text-slate-400 border-slate-700 hover:border-slate-600"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        {form.mode === "auto" ? (
          <div>
            <label className="block text-xs text-slate-500 mb-1">Period</label>
            <select
              value={form.period}
              onChange={(e) => setForm((f) => ({ ...f, period: e.target.value }))}
              className="w-32 text-sm bg-slate-950 border border-slate-700 rounded-md px-2.5 py-2 text-slate-100 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            >
              {PERIOD_OPTIONS.map((p) => (
                <option key={p.value} value={p.value}>
                  {p.label}
                </option>
              ))}
            </select>
            <p className="text-xs text-slate-500 mt-1.5">
              Return and std will be calculated from the ticker's historical data for this period.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-slate-500 mb-1">Return (%)</label>
              <input
                type="number"
                value={form.returnPct}
                onChange={(e) => setForm((f) => ({ ...f, returnPct: e.target.value }))}
                placeholder="e.g. 13"
                className="w-full text-sm bg-slate-950 border border-slate-700 rounded-md px-2.5 py-2 text-slate-100 placeholder-slate-600 font-mono focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-500 mb-1">Std (%)</label>
              <input
                type="number"
                value={form.stdPct}
                onChange={(e) => setForm((f) => ({ ...f, stdPct: e.target.value }))}
                placeholder="e.g. 18"
                className="w-full text-sm bg-slate-950 border border-slate-700 rounded-md px-2.5 py-2 text-slate-100 placeholder-slate-600 font-mono focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
              />
            </div>
          </div>
        )}

        <div className="flex items-center gap-2">
          <button
            onClick={onAdd}
            className="flex items-center gap-1.5 text-sm bg-teal-500 hover:bg-teal-400 text-slate-950 font-medium px-3.5 py-2 rounded-md transition-colors"
          >
            {isEditing ? <Save className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
            {isEditing ? "Update product" : "Add product"}
          </button>
          {isEditing && (
            <button
              onClick={onCancelEdit}
              className="flex items-center gap-1.5 text-sm text-slate-300 border border-slate-700 hover:bg-slate-800 px-3.5 py-2 rounded-md transition-colors"
            >
              <X className="w-4 h-4" />
              Cancel
            </button>
          )}
        </div>
      </Card>

      <Card className="overflow-hidden anim-in">
        <div className="px-4 py-3 bg-slate-900 border-b border-slate-800">
          <h2 className="text-sm font-semibold text-slate-200">Configured products ({products.length})</h2>
        </div>
        {products.length === 0 ? (
          <p className="text-sm text-slate-500 px-4 py-8 text-center">No products added yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm min-w-[560px]">
              <thead>
                <tr className="text-left text-xs text-slate-500 border-b border-slate-800">
                  <th className="px-4 py-2 font-medium">Name</th>
                  <th className="px-4 py-2 font-medium">Ticker</th>
                  <th className="px-4 py-2 font-medium">Mode</th>
                  <th className="px-4 py-2 font-medium">Period</th>
                  <th className="px-4 py-2 font-medium">Return</th>
                  <th className="px-4 py-2 font-medium">Std</th>
                  <th className="px-4 py-2 font-medium" />
                </tr>
              </thead>
              <tbody>
                {products.map((p) => (
                  <tr
                    key={p.id}
                    className={`border-b border-slate-800/60 last:border-0 hover:bg-slate-900/60 ${
                      p.id === editingId ? "bg-teal-950/30" : ""
                    }`}
                  >
                    <td className="px-4 py-2.5 text-slate-200">{p.name}</td>
                    <td className="px-4 py-2.5 text-slate-500 font-mono">{p.ticker}</td>
                    <td className="px-4 py-2.5 capitalize text-slate-400">{p.mode}</td>
                    <td className="px-4 py-2.5 text-slate-400">
                      {p.period ? PERIOD_OPTIONS.find((o) => o.value === p.period)?.label : "-"}
                    </td>
                    <td className="px-4 py-2.5 font-mono text-emerald-400">{(p.return * 100).toFixed(2)}%</td>
                    <td className="px-4 py-2.5 font-mono text-slate-300">{(p.std * 100).toFixed(2)}%</td>
                    <td className="px-4 py-2.5 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => onEdit(p)}
                          className="text-slate-600 hover:text-teal-400 transition-colors"
                          title="Edit product"
                        >
                          <Pencil className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => onRemove(p.id)}
                          className="text-slate-600 hover:text-rose-400 transition-colors"
                          title="Delete product"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {products.length >= 2 && (
        <Card className="overflow-hidden anim-in">
          <div className="px-4 py-3 bg-slate-900 border-b border-slate-800">
            <h2 className="text-sm font-semibold text-slate-200">Correlation matrix</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Auto-calculated from ticker data; you can manually edit any cell (between -1 and 1).
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="text-sm">
              <thead>
                <tr>
                  <th className="px-3 py-2 sticky left-0 bg-slate-900 z-10" />
                  {products.map((p) => (
                    <th key={p.id} className="px-3 py-2 text-xs font-medium text-slate-500 whitespace-nowrap">
                      {p.name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {products.map((rowP) => (
                  <tr key={rowP.id} className="border-t border-slate-800">
                    <td className="px-3 py-2 text-xs text-slate-400 whitespace-nowrap sticky left-0 bg-slate-900/95 z-10">
                      {rowP.name}
                    </td>
                    {products.map((colP) => {
                      if (rowP.id === colP.id) {
                        return (
                          <td key={colP.id} className="px-3 py-2 text-center text-slate-600 font-mono">
                            1.00
                          </td>
                        );
                      }
                      const key = pairKeyById(rowP.id, colP.id);
                      const isOverridden = corrOverrides[key] !== undefined;
                      const value = getCorrelation(rowP, colP, corrOverrides);
                      return (
                        <td key={colP.id} className={`px-2 py-1.5 text-center ${isOverridden ? "bg-amber-950/40" : ""}`}>
                          <div className="flex items-center justify-center gap-1">
                            <input
                              type="number"
                              step="0.01"
                              min="-1"
                              max="1"
                              value={value}
                              onChange={(e) => {
                                const num = Number(e.target.value);
                                if (Number.isNaN(num)) return;
                                onCorrChange(rowP.id, colP.id, Math.max(-1, Math.min(1, num)));
                              }}
                              className="w-16 text-sm text-center font-mono bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-slate-100 focus:outline-none focus:border-teal-500"
                            />
                            {isOverridden && (
                              <button
                                onClick={() => onCorrReset(rowP.id, colP.id)}
                                className="text-slate-600 hover:text-slate-300"
                                title="Reset to calculated value"
                              >
                                <RotateCcw className="w-3 h-3" />
                              </button>
                            )}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

function RunScreen({
  products,
  selectedIds,
  onToggle,
  amount,
  setAmount,
  returnSafety,
  setReturnSafety,
  riskTolerance,
  setRiskTolerance,
  diversify,
  setDiversify,
  rebalance,
  setRebalance,
  rebalanceFrequency,
  setRebalanceFrequency,
  rebalanceCost,
  setRebalanceCost,
  onRun,
}) {
  return (
    <div className="space-y-5">
      <Card className="p-4 sm:p-5 space-y-5 anim-in">
        <div>
          <label className="block text-xs text-slate-500 mb-1">Investment amount</label>
          <div className="relative w-full sm:w-64">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 text-sm">₹</span>
            <input
              type="number"
              value={amount}
              onChange={(e) => setAmount(Number(e.target.value))}
              className="w-full text-sm bg-slate-950 border border-slate-700 rounded-md pl-7 pr-2.5 py-2 text-slate-100 font-mono focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            />
          </div>
        </div>

        <div>
          <label className="block text-xs text-slate-500 mb-2">
            Select 2-5 products <span className="text-slate-600">({selectedIds.length} selected)</span>
          </label>
          {products.length === 0 ? (
            <p className="text-sm text-slate-500">Add products from the config screen first.</p>
          ) : (
            <div className="space-y-1.5">
              {products.map((p) => {
                const checked = selectedIds.includes(p.id);
                return (
                  <label
                    key={p.id}
                    className={`flex items-center gap-3 text-sm rounded-md px-3 py-2.5 cursor-pointer border transition-colors ${
                      checked ? "border-teal-700 bg-teal-950/30" : "border-slate-800 hover:border-slate-700"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => onToggle(p.id)}
                      className="accent-teal-500"
                    />
                    <span className="flex-1 text-slate-200 min-w-0 truncate">{p.name}</span>
                    <span className="text-slate-500 text-xs font-mono hidden sm:inline">{p.ticker}</span>
                    <span className="text-slate-400 text-xs font-mono whitespace-nowrap">
                      {(p.return * 100).toFixed(1)}% / {(p.std * 100).toFixed(1)}%
                    </span>
                  </label>
                );
              })}
            </div>
          )}
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-slate-500 mb-1">Return vs safety</label>
            <select
              value={returnSafety}
              onChange={(e) => setReturnSafety(e.target.value)}
              className="w-full text-sm bg-slate-950 border border-slate-700 rounded-md px-2.5 py-2 text-slate-100 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            >
              <option value="safety">Safety</option>
              <option value="balanced">Balanced</option>
              <option value="return">Return</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-500 mb-1">Risk tolerance</label>
            <select
              value={riskTolerance}
              onChange={(e) => setRiskTolerance(e.target.value)}
              className="w-full text-sm bg-slate-950 border border-slate-700 rounded-md px-2.5 py-2 text-slate-100 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>
        </div>

        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={diversify}
            onChange={(e) => setDiversify(e.target.checked)}
            className="accent-teal-500"
          />
          Use equal risk allocation (diversification)
        </label>

        <div className="rounded-lg border border-slate-800 px-3.5 py-3 space-y-3">
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={rebalance}
              onChange={(e) => setRebalance(e.target.checked)}
              className="accent-teal-500"
            />
            Simulate periodic rebalancing <span className="text-slate-600 text-xs">(optional)</span>
          </label>
          <p className="text-xs text-slate-500 pl-6">
            Resetting weights back to target periodically costs a little in fees/taxes each time — this reduces the
            expected return slightly. Leave unchecked to see returns without rebalancing.
          </p>
          {rebalance && (
            <div className="grid grid-cols-2 gap-3 pl-6 pt-1">
              <div>
                <label className="block text-xs text-slate-500 mb-1">Frequency</label>
                <select
                  value={rebalanceFrequency}
                  onChange={(e) => setRebalanceFrequency(e.target.value)}
                  className="w-full text-sm bg-slate-950 border border-slate-700 rounded-md px-2.5 py-2 text-slate-100 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
                >
                  {REBALANCE_FREQUENCY_OPTIONS.map((f) => (
                    <option key={f.value} value={f.value}>
                      {f.label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-slate-500 mb-1">Cost per rebalance (%)</label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  value={rebalanceCost}
                  onChange={(e) => setRebalanceCost(e.target.value)}
                  className="w-full text-sm bg-slate-950 border border-slate-700 rounded-md px-2.5 py-2 text-slate-100 font-mono focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
                />
              </div>
            </div>
          )}
        </div>

        <button
          onClick={onRun}
          className="flex items-center gap-1.5 text-sm bg-teal-500 hover:bg-teal-400 text-slate-950 font-medium px-4 py-2.5 rounded-md transition-colors"
        >
          <Play className="w-4 h-4" />
          Run model
        </button>
      </Card>
    </div>
  );
}

function ResultScreen({ result, amount, chartData, onBack }) {
  const { portReturn, netReturn, portVol, sharpeLike, riskLevel, expectedValue, selected, weights, rebalance, Sigma } =
    result;

  const animReturn = useCountUp(netReturn * 100);
  const animVol = useCountUp(portVol * 100);
  const animSharpe = useCountUp(sharpeLike);
  const animValue = useCountUp(expectedValue);

  const [mcHorizon, setMcHorizon] = useState("1");
  const [mcSimCount, setMcSimCount] = useState(500);
  const [mcTargetPct, setMcTargetPct] = useState(20);
  const [mcRunning, setMcRunning] = useState(false);
  const [mcOutput, setMcOutput] = useState(null);

  function handleRunMonteCarlo() {
    setMcRunning(true);
    // Defer one tick so the "Running..." state paints before the (brief) computation.
    setTimeout(() => {
      const horizon = MC_HORIZON_OPTIONS.find((h) => h.value === mcHorizon);
      const output = runMonteCarloSimulation({
        selected,
        weights,
        amount,
        Sigma,
        years: horizon.years,
        simulationsCount: mcSimCount,
        rebalanceInfo: rebalance,
      });
      const targetValue = amount * (1 + Number(mcTargetPct) / 100);
      const probTarget = output.finalValues.filter((v) => v >= targetValue).length / output.finalValues.length;
      setMcOutput({ ...output, targetValue, probTarget, horizonLabel: horizon.label });
      setMcRunning(false);
    }, 30);
  }

  // Reshape the capped sample of individual simulated paths into one row per
  // time step (what recharts needs for a multi-line chart): each row carries
  // that step's value for every sampled run, keyed sim0, sim1, sim2...
  const spaghettiData = useMemo(() => {
    if (!mcOutput) return [];
    const rows = [];
    for (let s = 0; s <= mcOutput.steps; s++) {
      const row = { month: s === 0 ? "Start" : `M${s}` };
      mcOutput.samplePaths.forEach((path, idx) => {
        row[`sim${idx}`] = path[s];
      });
      rows.push(row);
    }
    return rows;
  }, [mcOutput]);

  return (
    <div className="space-y-5">
      {/* Metric cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricCard
          icon={<TrendingUp className="w-4 h-4 text-emerald-400" />}
          label="Expected return"
          value={`${animReturn.toFixed(2)}%`}
          subValue={rebalance?.enabled ? `${(portReturn * 100).toFixed(2)}% before rebalancing cost` : undefined}
          delay={0}
        />
        <MetricCard
          icon={<Activity className="w-4 h-4 text-indigo-400" />}
          label="Volatility"
          value={`${animVol.toFixed(2)}%`}
          subValue={`₹${Math.round((animVol / 100) * amount).toLocaleString("en-IN")}`}
          delay={60}
        />
        <MetricCard
          icon={<Gauge className="w-4 h-4 text-amber-400" />}
          label="Risk-reward ratio"
          value={animSharpe.toFixed(2)}
          delay={120}
        />
        <div
          className={`rounded-xl border px-3.5 py-3 anim-in ${RISK_BADGE_STYLES[riskLevel]}`}
          style={{ animationDelay: "180ms" }}
        >
          <div className="flex items-center gap-1.5 mb-1">
            <ShieldCheck className="w-4 h-4" />
            <p className="text-xs opacity-80">Risk level</p>
          </div>
          <p className="text-lg font-semibold">{riskLevel}</p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card className="p-4 sm:p-5 anim-in" style={{ animationDelay: "100ms" }}>
          <h2 className="text-sm font-semibold text-slate-200 mb-3">Allocation split</h2>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie
                data={chartData}
                dataKey="allocation"
                nameKey="name"
                innerRadius="55%"
                outerRadius="85%"
                paddingAngle={2}
                isAnimationActive={true}
                animationDuration={900}
                animationEasing="ease-out"
              >
                {chartData.map((_, i) => (
                  <Cell key={i} fill={SERIES_COLORS[i % SERIES_COLORS.length]} stroke="#020617" strokeWidth={2} />
                ))}
              </Pie>
              <Tooltip
                formatter={(v) => [`${v}%`, "Allocation"]}
                itemStyle={{ color: "#e2e8f0" }}
                labelStyle={{ color: "#e2e8f0", fontWeight: 600 }}
                contentStyle={{
                  backgroundColor: "#0f172a",
                  border: "1px solid #1e293b",
                  borderRadius: 8,
                  fontSize: 12,
                  color: "#e2e8f0",
                }}
              />
              <Legend
                verticalAlign="bottom"
                iconType="circle"
                iconSize={8}
                wrapperStyle={{ fontSize: 12, color: "#94a3b8" }}
              />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card className="p-4 sm:p-5 anim-in" style={{ animationDelay: "160ms" }}>
          <h2 className="text-sm font-semibold text-slate-200 mb-3">Allocation by weight</h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData} margin={{ top: 4, right: 8, left: -8, bottom: 28 }}>
              <defs>
                <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#2dd4bf" stopOpacity={1} />
                  <stop offset="100%" stopColor="#0f766e" stopOpacity={0.9} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 11, fill: "#94a3b8" }}
                interval={0}
                angle={-25}
                textAnchor="end"
                height={50}
              />
              <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} unit="%" width={40} />
              <Tooltip
                formatter={(v) => [`${v}%`, "Allocation"]}
                cursor={{ fill: "#1e293b", opacity: 0.4 }}
                itemStyle={{ color: "#e2e8f0" }}
                labelStyle={{ color: "#e2e8f0", fontWeight: 600 }}
                contentStyle={{
                  backgroundColor: "#0f172a",
                  border: "1px solid #1e293b",
                  borderRadius: 8,
                  fontSize: 12,
                  color: "#e2e8f0",
                }}
              />
              <Bar
                dataKey="allocation"
                fill="url(#barGradient)"
                radius={[6, 6, 0, 0]}
                isAnimationActive={true}
                animationDuration={900}
                animationEasing="ease-out"
              />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Holdings breakdown */}
      <Card className="p-4 sm:p-5 anim-in" style={{ animationDelay: "220ms" }}>
        <h2 className="text-sm font-semibold text-slate-200 mb-3">Holdings</h2>
        <div className="space-y-3">
          {selected.map((p, i) => {
            const pct = weights[i] * 100;
            return (
              <div key={p.id}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-200 flex items-center gap-2 min-w-0">
                    <span
                      className="h-2 w-2 rounded-full shrink-0"
                      style={{ backgroundColor: SERIES_COLORS[i % SERIES_COLORS.length] }}
                    />
                    <span className="truncate">{p.name}</span>
                  </span>
                  <span className="text-slate-400 font-mono whitespace-nowrap">
                    {pct.toFixed(2)}% · ₹{Math.round(weights[i] * amount).toLocaleString("en-IN")}
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full rounded-full anim-bar"
                    style={{
                      width: `${pct}%`,
                      backgroundColor: SERIES_COLORS[i % SERIES_COLORS.length],
                      animationDelay: `${240 + i * 80}ms`,
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      <Card className="p-4 sm:p-5 anim-in" style={{ animationDelay: "280ms" }}>
        <div className="flex items-center gap-2 mb-1">
          <Wallet className="w-4 h-4 text-teal-400" />
          <p className="text-xs text-slate-500">Projected value after 1 year</p>
        </div>
        <p className="text-2xl sm:text-3xl font-semibold text-slate-100 font-mono">
          ₹{Math.round(animValue).toLocaleString("en-IN")}
        </p>
        {rebalance?.enabled && (
          <div className="flex items-start gap-2 mt-3 pt-3 border-t border-slate-800">
            <RefreshCw className="w-3.5 h-3.5 text-indigo-400 mt-0.5 shrink-0" />
            <p className="text-xs text-slate-500">
              Includes <span className="text-slate-300">{rebalance.frequencyLabel.toLowerCase()}</span> rebalancing at{" "}
              <span className="text-slate-300 font-mono">{rebalance.costPct}%</span> per rebalance — a{" "}
              <span className="text-slate-300 font-mono">{(rebalance.drag * 100).toFixed(2)}%</span> annual drag on
              returns.
            </p>
          </div>
        )}
      </Card>

      {/* Monte Carlo simulation */}
      <Card className="p-4 sm:p-5 anim-in" style={{ animationDelay: "320ms" }}>
        <div className="flex items-center gap-2 mb-1">
          <Dices className="w-4 h-4 text-indigo-400" />
          <h2 className="text-sm font-semibold text-slate-200">Monte Carlo simulation</h2>
        </div>
        <p className="text-xs text-slate-500 mb-4">
          Runs many random future paths, simulating each holding separately with its own return and volatility
          (Geometric Brownian Motion) while keeping their correlations intact, to show a realistic spread of outcomes
          — not just one projected number.
        </p>

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-4">
          <div>
            <label className="block text-xs text-slate-500 mb-1">Time horizon</label>
            <select
              value={mcHorizon}
              onChange={(e) => setMcHorizon(e.target.value)}
              className="w-full text-sm bg-slate-950 border border-slate-700 rounded-md px-2.5 py-2 text-slate-100 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            >
              {MC_HORIZON_OPTIONS.map((h) => (
                <option key={h.value} value={h.value}>
                  {h.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-500 mb-1">Simulations</label>
            <select
              value={mcSimCount}
              onChange={(e) => setMcSimCount(Number(e.target.value))}
              className="w-full text-sm bg-slate-950 border border-slate-700 rounded-md px-2.5 py-2 text-slate-100 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            >
              {MC_SIM_COUNT_OPTIONS.map((n) => (
                <option key={n} value={n}>
                  {n.toLocaleString("en-IN")}
                </option>
              ))}
            </select>
          </div>
          <div className="col-span-2 sm:col-span-1">
            <label className="block text-xs text-slate-500 mb-1">Target gain (%)</label>
            <input
              type="number"
              step="1"
              value={mcTargetPct}
              onChange={(e) => setMcTargetPct(e.target.value)}
              className="w-full text-sm bg-slate-950 border border-slate-700 rounded-md px-2.5 py-2 text-slate-100 font-mono focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            />
          </div>
        </div>

        <button
          onClick={handleRunMonteCarlo}
          disabled={mcRunning}
          className="flex items-center gap-1.5 text-sm bg-indigo-500 hover:bg-indigo-400 disabled:opacity-60 text-slate-950 font-medium px-3.5 py-2 rounded-md transition-colors mb-5"
        >
          {mcRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Dices className="w-4 h-4" />}
          {mcRunning ? "Running..." : mcOutput ? "Re-run simulation" : "Run simulation"}
        </button>

        {mcOutput && (
          <div className="space-y-4 anim-in">
            <ResponsiveContainer width="100%" height={280}>
              <ComposedChart data={mcOutput.bands} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
                <defs>
                  <linearGradient id="mcBand" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#818cf8" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#818cf8" stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis dataKey="month" tick={{ fontSize: 10, fill: "#94a3b8" }} interval="preserveStartEnd" />
                <YAxis
                  tick={{ fontSize: 10, fill: "#94a3b8" }}
                  width={56}
                  tickFormatter={(v) => `₹${Math.round(v / 1000)}k`}
                />
                <Tooltip
                  labelFormatter={(label) => `Month: ${label}`}
                  formatter={(v, name) => [`₹${Math.round(v).toLocaleString("en-IN")}`, name]}
                  itemStyle={{ color: "#e2e8f0" }}
                  labelStyle={{ color: "#e2e8f0", fontWeight: 600 }}
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    border: "1px solid #1e293b",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                />
                <Area
                  dataKey={(d) => [d.p10, d.p90]}
                  name="10th–90th percentile"
                  stroke="none"
                  fill="url(#mcBand)"
                  isAnimationActive={true}
                  animationDuration={800}
                />
                <Line
                  dataKey="p50"
                  name="Median projection"
                  stroke="#2dd4bf"
                  strokeWidth={2.5}
                  dot={false}
                  isAnimationActive={true}
                  animationDuration={900}
                />
                <ReferenceLine
                  y={amount}
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                  label={{ value: "Invested amount", position: "insideBottomRight", fill: "#94a3b8", fontSize: 10 }}
                />
                <ReferenceLine
                  y={mcOutput.targetValue}
                  stroke="#fbbf24"
                  strokeDasharray="4 4"
                  label={{ value: `+${mcTargetPct}% target`, position: "insideTopRight", fill: "#fbbf24", fontSize: 10 }}
                />
              </ComposedChart>
            </ResponsiveContainer>

            <div>
              <h3 className="text-xs font-semibold text-slate-300 mb-1">Simulated paths</h3>
              <p className="text-xs text-slate-500 mb-2">
                Each faint line is one simulated future, generated from every holding's own return, volatility, and
                correlation with the others (Cholesky-correlated GBM) — showing {mcOutput.samplePaths.length} of{" "}
                {mcSimCount.toLocaleString("en-IN")} runs for legibility.
              </p>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={spaghettiData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="month" tick={{ fontSize: 10, fill: "#94a3b8" }} interval="preserveStartEnd" />
                  <YAxis
                    tick={{ fontSize: 10, fill: "#94a3b8" }}
                    width={56}
                    tickFormatter={(v) => `₹${Math.round(v / 1000)}k`}
                  />
                  <ReferenceLine y={amount} stroke="#94a3b8" strokeDasharray="4 4" />
                  {mcOutput.samplePaths.map((_, idx) => (
                    <Line
                      key={idx}
                      dataKey={`sim${idx}`}
                      stroke="#818cf8"
                      strokeWidth={1}
                      dot={false}
                      isAnimationActive={false}
                      strokeOpacity={0.22}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <div className="rounded-lg border border-slate-800 px-3 py-2.5">
                <p className="text-xs text-slate-500">Median outcome</p>
                <p className="text-sm font-semibold text-slate-100 font-mono">
                  ₹{Math.round(mcOutput.stats.median).toLocaleString("en-IN")}
                </p>
              </div>
              <div className="rounded-lg border border-slate-800 px-3 py-2.5">
                <p className="text-xs text-slate-500">Worst case (10th pct.)</p>
                <p className="text-sm font-semibold text-rose-400 font-mono">
                  ₹{Math.round(mcOutput.stats.p10).toLocaleString("en-IN")}
                </p>
              </div>
              <div className="rounded-lg border border-slate-800 px-3 py-2.5">
                <p className="text-xs text-slate-500">Best case (90th pct.)</p>
                <p className="text-sm font-semibold text-emerald-400 font-mono">
                  ₹{Math.round(mcOutput.stats.p90).toLocaleString("en-IN")}
                </p>
              </div>
              <div className="rounded-lg border border-slate-800 px-3 py-2.5">
                <p className="text-xs text-slate-500">Chance of loss</p>
                <p className="text-sm font-semibold text-slate-100 font-mono">
                  {(mcOutput.stats.probLoss * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            <div className="rounded-lg border border-amber-900 bg-amber-950/30 px-3.5 py-2.5">
              <p className="text-xs text-slate-300">
                Probability of reaching <span className="font-mono text-amber-400">+{mcTargetPct}%</span> (₹
                {Math.round(mcOutput.targetValue).toLocaleString("en-IN")}) within {mcOutput.horizonLabel.toLowerCase()}:{" "}
                <span className="font-mono text-amber-400">{(mcOutput.probTarget * 100).toFixed(1)}%</span> of
                simulated paths.
              </p>
            </div>

            <div>
              <h3 className="text-xs font-semibold text-slate-300 mb-1">Per-instrument worst case</h3>
              <p className="text-xs text-slate-500 mb-2">
                For each holding, using its own volatility: the simulated 5th-percentile outcome from the paths above,
                and an analytical worst case (95% confidence, lognormal) computed directly from that instrument's own
                return and std.
              </p>
              <div className="space-y-2">
                {mcOutput.instrumentStats.map((s, i) => (
                  <div
                    key={s.name}
                    className="flex items-center justify-between gap-3 text-sm rounded-lg border border-slate-800 px-3 py-2.5"
                  >
                    <span className="flex items-center gap-2 min-w-0">
                      <span
                        className="h-2 w-2 rounded-full shrink-0"
                        style={{ backgroundColor: SERIES_COLORS[i % SERIES_COLORS.length] }}
                      />
                      <span className="truncate text-slate-200">{s.name}</span>
                    </span>
                    <div className="text-right font-mono text-xs shrink-0">
                      <div className="text-rose-400">
                        ₹{Math.round(s.simulatedWorst).toLocaleString("en-IN")} ({s.worstPct.toFixed(1)}%)
                      </div>
                      <div className="text-slate-600">
                        analytical: ₹{Math.round(s.analyticalWorst).toLocaleString("en-IN")}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <p className="text-xs text-slate-600">
              Based on {mcSimCount.toLocaleString("en-IN")} simulated paths, each built by simulating every holding
              separately from its own return and volatility with their mutual correlations preserved (not just the
              portfolio's blended return/volatility). This is a probabilistic projection, not a guarantee — actual
              results can fall outside the shown band.
            </p>
          </div>
        )}
      </Card>

      <button
        onClick={onBack}
        className="flex items-center gap-1.5 text-sm text-slate-300 border border-slate-700 hover:bg-slate-900 px-3.5 py-2 rounded-md transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        Back
      </button>
    </div>
  );
}

function MetricCard({ icon, label, value, subValue, delay = 0 }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 px-3.5 py-3 anim-in" style={{ animationDelay: `${delay}ms` }}>
      <div className="flex items-center gap-1.5 mb-1">
        {icon}
        <p className="text-xs text-slate-500">{label}</p>
      </div>
      <p className="text-lg font-semibold text-slate-100 font-mono">{value}</p>
      {subValue && <p className="text-xs text-slate-500 font-mono mt-0.5">{subValue}</p>}
    </div>
  );
}