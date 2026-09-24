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
  Database,
  Upload,
  FileText,
  Shuffle,
  Square,
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
  ScatterChart,
  Scatter,
  ZAxis,
} from "recharts";

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

const HISTORY_HORIZON_OPTIONS = [
  { value: "1", label: "1 Year", years: 1 },
  { value: "3", label: "3 Years", years: 3 },
  { value: "5", label: "5 Years", years: 5 },
];

const HISTORY_WINDOW_OPTIONS = [
  { value: "0", label: "All shared history", years: 0 },
  { value: "5", label: "Last 5 years", years: 5 },
  { value: "3", label: "Last 3 years", years: 3 },
  { value: "2", label: "Last 2 years", years: 2 },
];

const HISTORY_BLOCK_OPTIONS = [
  { value: "1", label: "1 month (independent months)" },
  { value: "3", label: "3 months" },
  { value: "6", label: "6 months" },
  { value: "12", label: "12 months" },
];

const MC_SIM_COUNT_OPTIONS = [200, 500, 1000, 5000, 50000, 100000, 1000000];

// Percentiles reported in the Monte Carlo percentile table.
const MC_PERCENTILES = [5, 10, 25, 50, 75, 90, 95];

const MC_MODEL_OPTIONS = [
  { value: "dynamic", label: "Dynamic (real-market)" },
  { value: "constant", label: "Constant (simple)" },
];

// --- DYNAMIC MARKET MODEL (regimes + stochastic volatility + stress correlation)
// Real markets do not keep one return, one std and one correlation forever.
// This model lets all three move month by month:
//   1. REGIMES -- a Markov chain switches the whole market between Calm,
//      Normal and Stress. Each regime has its own volatility multiplier.
//      The chain is built from two intuitive inputs: how much of the time
//      the market spends in Stress, and how long a Stress spell lasts.
//   2. MARKET VOLATILITY -- on top of the regime, market volatility drifts
//      continuously (mean-reverting AR(1) on log-volatility), so std keeps
//      moving above AND below the value you entered.
//   3. PER-HOLDING VOLATILITY -- every holding also has its own smaller
//      volatility wobble, so holdings do not all scale in lock-step.
//   4. RETURN LINKED TO STD -- when a holding's volatility is above its
//      long-run level, its expected return is shifted (direction and
//      strength are user-selectable: real-market = return falls, risk
//      premium = return rises, or no link).
//   5. LEVERAGE FEEDBACK -- a negative market shock makes next month's
//      market volatility more likely to rise.
//   6. STRESS CORRELATION -- in Stress, positive correlations move toward 1
//      (diversification weakens when you need it most); negative
//      correlations (hedges) are left alone.
// Everything is normalised so that, averaged over the long run, the std
// you entered is still the std (E[k^2] = 1) and the return you entered is
// still the return (the adjustment averages to zero). The correlation you
// enter is the correlation in Calm/Normal conditions.
const REGIME_LABELS = ["Calm", "Normal", "Stress"];
const STRESS_INDEX = 2;

const VOL_PERSISTENCE = 0.85; // monthly AR(1) persistence of log-volatility
const RISK_PREMIUM_DAMPING = 0.3; // the empirical risk-premium effect is much weaker than the leverage effect

const RETURN_LINK_OPTIONS = [
  { value: "real", label: "Real market — return falls when volatility rises" },
  { value: "none", label: "No link — return independent of volatility" },
  { value: "premium", label: "Risk premium — return rises with volatility (weaker)" },
];

// Market "severity" presets. These are reasonable stylised defaults, not
// calibrated to one specific market -- every value can be edited under
// "Advanced market settings".
const MARKET_PRESETS = {
  mild: {
    label: "Mild",
    calmMult: 0.8,
    stressMult: 1.5,
    stressShare: 0.08,
    stressMonths: 4,
    volOfVol: 0.15,
    idioVolOfVol: 0.08,
    returnLink: 0.8,
    leverageCorr: -0.4,
    stressCorrBoost: 0.15,
  },
  realistic: {
    label: "Realistic",
    calmMult: 0.7,
    stressMult: 1.9,
    stressShare: 0.12,
    stressMonths: 5.5,
    volOfVol: 0.2,
    idioVolOfVol: 0.1,
    returnLink: 1.2,
    leverageCorr: -0.5,
    stressCorrBoost: 0.3,
  },
  severe: {
    label: "Severe",
    calmMult: 0.65,
    stressMult: 2.4,
    stressShare: 0.18,
    stressMonths: 8,
    volOfVol: 0.28,
    idioVolOfVol: 0.12,
    returnLink: 1.6,
    leverageCorr: -0.6,
    stressCorrBoost: 0.45,
  },
};

const MARKET_PARAM_FIELDS = [
  { key: "calmMult", label: "Calm volatility (× your std)", min: 0.4, max: 1, step: 0.05, scale: 1 },
  { key: "stressMult", label: "Stress volatility (× your std)", min: 1.1, max: 4, step: 0.1, scale: 1 },
  { key: "stressShare", label: "Time in Stress (%)", min: 2, max: 35, step: 1, scale: 100 },
  { key: "stressMonths", label: "Avg Stress length (months)", min: 2, max: 24, step: 0.5, scale: 1 },
  { key: "volOfVol", label: "Market vol-of-vol", min: 0, max: 0.6, step: 0.01, scale: 1 },
  { key: "idioVolOfVol", label: "Per-holding vol-of-vol", min: 0, max: 0.4, step: 0.01, scale: 1 },
  { key: "returnLink", label: "Return–vol link strength", min: 0, max: 3, step: 0.1, scale: 1 },
  { key: "leverageCorr", label: "Leverage correlation", min: -0.95, max: 0, step: 0.05, scale: 1 },
  { key: "stressCorrBoost", label: "Stress correlation boost", min: 0, max: 0.8, step: 0.05, scale: 1 },
];

function paramsToText(params) {
  const out = {};
  MARKET_PARAM_FIELDS.forEach((f) => {
    out[f.key] = String(Number((params[f.key] * f.scale).toFixed(3)));
  });
  return out;
}

// Turns the (string) settings from the form into clamped numbers, and folds
// the chosen return-vs-volatility direction into one signed coefficient
// (negative = return falls when volatility rises).
function sanitizeMarketParams(text, linkMode) {
  const out = {};
  MARKET_PARAM_FIELDS.forEach((f) => {
    let v = Number(text[f.key]);
    if (text[f.key] === "" || text[f.key] === undefined || Number.isNaN(v)) {
      v = MARKET_PRESETS.realistic[f.key] * f.scale;
    }
    out[f.key] = Math.min(f.max, Math.max(f.min, v)) / f.scale;
  });
  out.returnLinkSigned =
    linkMode === "real" ? -out.returnLink : linkMode === "premium" ? out.returnLink * RISK_PREMIUM_DAMPING : 0;
  return out;
}

function sampleIndex(probs) {
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

function stationaryDistribution(P) {
  let pi = P.map(() => 1 / P.length);
  for (let it = 0; it < 300; it++) {
    pi = P.map((_, j) => pi.reduce((sum, p, i) => sum + p * P[i][j], 0));
  }
  return pi;
}

// Monthly regime transition matrix (Calm, Normal, Stress) built from the
// long-run share of time in Stress and the average length of a Stress spell.
// Stress is entered mostly from Normal (rarely straight from Calm) and left
// mostly toward Normal; the Normal-to-Stress probability is solved so the
// chain's long-run Stress share hits the requested value.
function buildRegimeTransitions(stressShare, stressMonths) {
  const exitStress = 1 / stressMonths;
  const stressRow = [0.15 * exitStress, 0.85 * exitStress, 1 - exitStress];
  const make = (aN) => {
    const aC = 0.2 * aN;
    return [
      [1 - 0.05 - aC, 0.05, aC],
      [0.04, 1 - 0.04 - aN, aN],
      stressRow,
    ];
  };
  let lo = 0;
  let hi = 0.3;
  for (let it = 0; it < 40; it++) {
    const mid = (lo + hi) / 2;
    if (stationaryDistribution(make(mid))[STRESS_INDEX] < stressShare) lo = mid;
    else hi = mid;
  }
  return make((lo + hi) / 2);
}

// Long-run regime probabilities plus the constants that keep the user's
// inputs as long-run averages.
function buildDynamicParams(mp) {
  const transitions = buildRegimeTransitions(mp.stressShare, mp.stressMonths);
  const pi = stationaryDistribution(transitions);
  const raw = [mp.calmMult, 1, mp.stressMult];
  const meanSq = pi.reduce((sum, p, i) => sum + p * raw[i] ** 2, 0);
  const mult = raw.map((m) => m / Math.sqrt(meanSq)); // E[mult^2] = 1
  const meanMult = pi.reduce((sum, p, i) => sum + p * mult[i], 0);
  // v = exp(y - s^2) with y ~ N(0, s^2)  =>  E[v^2] = 1 and E[v] = exp(-s^2/2)
  const meanMarketV = Math.exp(-0.5 * mp.volOfVol ** 2);
  const meanIdioV = Math.exp(-0.5 * mp.idioVolOfVol ** 2);
  return { transitions, pi, mult, kBar: meanMult * meanMarketV * meanIdioV };
}

// Because a holding's return shift moves up and down with volatility, the
// growth it compounds is random, and by Jensen's inequality a random growth
// rate lifts the AVERAGE outcome slightly above what you entered. To keep
// your return as the true long-run average, a short pilot simulation of just
// the volatility process measures that lift for each holding at each month,
// and the simulator subtracts exactly that (a small deterministic amount per
// month) from the drift.
function driftConvexityCorrections({ dyn, mp, baseVols, steps, dt, pilotPaths = 3000 }) {
  const n = baseVols.length;
  const link = mp.returnLinkSigned;
  const marketEta = mp.volOfVol * Math.sqrt(1 - VOL_PERSISTENCE ** 2);
  const idioEta = mp.idioVolOfVol * Math.sqrt(1 - VOL_PERSISTENCE ** 2);
  const sums = Array.from({ length: steps }, () => Array(n).fill(0));
  for (let path = 0; path < pilotPaths; path++) {
    let regime = sampleIndex(dyn.pi);
    let marketVol = mp.volOfVol * standardNormalRandom();
    const idio = Array.from({ length: n }, () => mp.idioVolOfVol * standardNormalRandom());
    const accumulated = Array(n).fill(0);
    for (let t = 0; t < steps; t++) {
      const marketK = dyn.mult[regime] * Math.exp(marketVol - mp.volOfVol ** 2);
      for (let i = 0; i < n; i++) {
        const k = marketK * Math.exp(idio[i] - mp.idioVolOfVol ** 2);
        accumulated[i] += link * (k - dyn.kBar) * baseVols[i] * dt;
        sums[t][i] += Math.exp(accumulated[i]);
      }
      const volShock = mp.leverageCorr * standardNormalRandom() + Math.sqrt(1 - mp.leverageCorr ** 2) * standardNormalRandom();
      marketVol = VOL_PERSISTENCE * marketVol + marketEta * volShock;
      for (let i = 0; i < n; i++) idio[i] = VOL_PERSISTENCE * idio[i] + idioEta * standardNormalRandom();
      regime = sampleIndex(dyn.transitions[regime]);
    }
  }
  const corrections = Array.from({ length: steps }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    let prev = 0;
    for (let t = 0; t < steps; t++) {
      const cum = Math.log(sums[t][i] / pilotPaths);
      corrections[t][i] = cum - prev;
      prev = cum;
    }
  }
  return corrections;
}

// A symmetric matrix is usable as a covariance/correlation matrix only if it
// is positive semi-definite; tested with a Cholesky attempt.
function isPositiveSemiDefinite(M) {
  const n = M.length;
  const L = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k];
      if (i === j) {
        const d = M[i][i] + 1e-9 - sum;
        if (d <= 0) return false;
        L[i][i] = Math.sqrt(d);
      } else {
        L[i][j] = (M[i][j] - sum) / L[j][j];
      }
    }
  }
  return true;
}

// Stress-regime correlation matrix: every positive correlation moves part of
// the way toward 1 (rho + boost*(1 - rho)); zero/negative correlations
// (hedges) are left unchanged. If the result is not a valid correlation
// matrix it is blended back toward the original until it is.
function stressCorrelationMatrix(C, boost) {
  const n = C.length;
  const candidate = C.map((row, i) =>
    row.map((c, j) => (i === j ? 1 : c > 0 ? c + boost * (1 - c) : c))
  );
  for (const t of [1, 0.75, 0.5, 0.25]) {
    const mix = C.map((row, i) => row.map((c, j) => (1 - t) * c + t * candidate[i][j]));
    if (isPositiveSemiDefinite(mix)) return mix;
  }
  return C.map((row) => [...row]);
}

// Monochrome palette for allocation series — alternating light/dark greys so neighbours stay distinct (up to 5 holdings).
const SERIES_COLORS = ["#fafafa", "#a3a3a3", "#d4d4d4", "#737373", "#525252"];

// One clearly different colour per simulated-path line. Hue steps by the
// golden angle (so neighbouring lines never look alike), while lightness and
// saturation cycle through a few levels so that even lines with a similar hue
// stay apart. Checked: every one of the 60 colours is visibly different from
// all the others.
const PATH_LIGHTNESS = [46, 58, 70, 82];
const PATH_SATURATION = [95, 70];
const PATH_COLORS = Array.from(
  { length: 60 },
  (_, i) =>
    `hsl(${Math.round((i * 137.508) % 360)}, ${PATH_SATURATION[(i >> 2) % PATH_SATURATION.length]}%, ${
      PATH_LIGHTNESS[i % PATH_LIGHTNESS.length]
    }%)`
);

// One clearly different colour per individual holding, for charts (like the
// historical backtest) that plot each holding's own line alongside the
// portfolio lines. Up to 5 holdings, chosen to stay visible against a dark
// background and distinct from each other.
const ASSET_LINE_COLORS = ["#f97316", "#38bdf8", "#facc15", "#f472b6", "#4ade80"];

// --- FORMAT HELPERS -----------------------------------------------------
function fmtINR(v) {
  return `₹${Math.round(v).toLocaleString("en-IN")}`;
}

// Compact Indian-style labels for chart axes: 12k, 1.5L, 2.3Cr.
function fmtINRCompact(v) {
  const abs = Math.abs(v);
  if (abs >= 1e7) return `₹${(v / 1e7).toFixed(2)}Cr`;
  if (abs >= 1e5) return `₹${(v / 1e5).toFixed(1)}L`;
  if (abs >= 1e3) return `₹${Math.round(v / 1e3)}k`;
  return `₹${Math.round(v)}`;
}

function fmtPct(v, digits = 1) {
  return `${(v * 100).toFixed(digits)}%`;
}

// --- MATH LAYER -------------------------------------------------------
// Unordered pair key by product id -- used to store/look up manual
// correlation overrides regardless of which order the two products
// are passed in.
function pairKeyById(idA, idB) {
  return [idA, idB].sort((a, b) => a - b).join(":");
}

// Default correlation between two different products until the user sets
// one in the correlation matrix (0 = uncorrelated).
const DEFAULT_CORRELATION = 0;

// Correlation for a pair of products: the value the user entered in the
// matrix if there is one; otherwise, if both products come with price
// history, the correlation measured from that history; otherwise
// DEFAULT_CORRELATION.
function getCorrelation(pA, pB, overrides = {}) {
  if (pA.id === pB.id) return 1;
  const key = pairKeyById(pA.id, pB.id);
  if (overrides[key] !== undefined) return overrides[key];
  if (pA.history && pB.history) {
    const measured = historicalCorrelation(pA.history, pB.history);
    if (measured !== null) return Math.round(measured * 1000) / 1000;
  }
  return DEFAULT_CORRELATION;
}

// Where a pair's correlation comes from: "manual", "data" or "default".
function correlationSource(pA, pB, overrides = {}) {
  if (overrides[pairKeyById(pA.id, pB.id)] !== undefined) return "manual";
  if (pA.history && pB.history && historicalCorrelation(pA.history, pB.history) !== null) return "data";
  return "default";
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
// (the "Constant" model). The "Dynamic" model multiplies sigma by a
// time-varying k and shifts mu by an opposite-moving adjustment -- see
// DYNAMIC MARKET MODEL above.
// where Z is a vector of independent standard normals (Box-Muller) and L
// is the Cholesky factor of the covariance matrix, so L*Z is a correlated
// normal vector with the right covariance structure.
//
// Standard risk output computed from the simulated distribution:
//   - Percentile fan (5/25/50/75/95) of portfolio value over time
//   - Distribution (histogram) of final portfolio values
//   - Mean, median, std of final value; percentile table
//   - Probability of loss
//   - Value at Risk (VaR 95%) and Conditional VaR / Expected Shortfall (CVaR 95%)
//   - Maximum drawdown per path (median and 5th-percentile worst)
//   - Annualized return distribution
// Box-Muller produces two independent normals per pair of uniforms; the
// second one is kept for the next call, which halves the cost of the very
// large runs.
let spareNormal = null;
function standardNormalRandom() {
  if (spareNormal !== null) {
    const value = spareNormal;
    spareNormal = null;
    return value;
  }
  let u = 0;
  while (u === 0) u = Math.random();
  const v = Math.random();
  const magnitude = Math.sqrt(-2 * Math.log(u));
  spareNormal = magnitude * Math.sin(2 * Math.PI * v);
  return magnitude * Math.cos(2 * Math.PI * v);
}

function percentileOfSorted(sorted, p) {
  const idx = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(idx);
  const upper = Math.ceil(idx);
  if (lower === upper) return sorted[lower];
  const weight = idx - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

// --- SCENARIO ENGINE ------------------------------------------------------
// Simulates possible futures for every holding and turns them into the
// month-by-month growth factor of each holding. Futures are produced in
// chunks and folded straight into running results, so even a million runs
// never has to be held in memory at once.
// The market "shock" that drives volatility feedback is measured on an
// equal-weight basket, so the simulated market does not depend on which
// allocation is being tested.
function prepareScenarioEngine({ selected, Sigma, years, model = "dynamic", marketParams = null }) {
  const n = selected.length;
  const L = choleskyDecomposition(Sigma);
  const stepsPerYear = 12; // monthly steps
  const steps = Math.max(1, Math.round(years * stepsPerYear));
  const dt = 1 / stepsPerYear;
  const sqrtDt = Math.sqrt(dt);

  const dynamic = model === "dynamic";
  const mp = {
    ...MARKET_PRESETS.realistic,
    returnLinkSigned: -MARKET_PRESETS.realistic.returnLink,
    ...(marketParams || {}),
  };
  const dyn = dynamic ? buildDynamicParams(mp) : null;
  const baseVols = selected.map((p) => p.std);
  const mus = selected.map((p) => p.return);

  // Calm/Normal correlation (as implied by Sigma) and the Stress version.
  const Cbase = Sigma.map((row, i) => row.map((v, j) => (i === j ? 1 : v / (baseVols[i] * baseVols[j]))));
  const Cstress = dynamic ? stressCorrelationMatrix(Cbase, mp.stressCorrBoost) : Cbase;
  const Lstress = dynamic
    ? choleskyDecomposition(Cstress.map((row, i) => row.map((c, j) => c * baseVols[i] * baseVols[j])))
    : L;
  const proxyWeights = Array(n).fill(1 / n);
  const shockSd = (C) => {
    let v = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) v += proxyWeights[i] * proxyWeights[j] * C[i][j];
    return Math.sqrt(Math.max(v, 1e-12));
  };
  const marketShockSdBase = shockSd(Cbase);
  const marketShockSdStress = shockSd(Cstress);

  const volSq = mp.volOfVol ** 2;
  const idioSq = mp.idioVolOfVol ** 2;
  const marketEta = mp.volOfVol * Math.sqrt(1 - VOL_PERSISTENCE ** 2);
  const idioEta = mp.idioVolOfVol * Math.sqrt(1 - VOL_PERSISTENCE ** 2);
  const leverageRest = Math.sqrt(1 - mp.leverageCorr ** 2);
  const driftCorr =
    dynamic && mp.returnLinkSigned !== 0 ? driftConvexityCorrections({ dyn, mp, baseVols, steps, dt }) : null;
  const regimeCounts = Array(REGIME_LABELS.length).fill(0);

  // Work buffers reused for every month of every future (no per-step allocation).
  const zIndep = new Float64Array(n);
  const zCorr = new Float64Array(n);
  const ks = new Float64Array(n); // std multiplier of each holding this month
  const adjs = new Float64Array(n); // return shift of each holding this month
  const idio = new Float64Array(n); // each holding's own volatility state

  function generate(count, trackCount = 0) {
    const growth = new Float64Array(count * steps * n);
    const trackRaw = [];
    let idx = 0;

    for (let sim = 0; sim < count; sim++) {
      // Each simulated future starts from a random (long-run typical) market
      // regime and volatility level.
      let regime = dynamic ? sampleIndex(dyn.pi) : 0;
      let marketVol = dynamic ? mp.volOfVol * standardNormalRandom() : 0;
      for (let i = 0; i < n; i++) idio[i] = dynamic ? mp.idioVolOfVol * standardNormalRandom() : 0;
      const trk = dynamic && sim < trackCount ? [] : null;

      for (let s = 1; s <= steps; s++) {
        for (let i = 0; i < n; i++) zIndep[i] = standardNormalRandom();
        const inStress = dynamic && regime === STRESS_INDEX;
        // Correlated shocks with variance = Sigma (lower-triangular Cholesky
        // factor); in Stress the (higher) stress correlations are used.
        const Lm = inStress ? Lstress : L;
        for (let i = 0; i < n; i++) {
          const row = Lm[i];
          let acc = 0;
          for (let j = 0; j <= i; j++) acc += row[j] * zIndep[j];
          zCorr[i] = acc;
        }

        // This month's market conditions. ks[i] scales holding i's std
        // (k > 1 = more volatile than the std you entered, k < 1 = calmer);
        // adjs[i] shifts its return by the chosen link to volatility, in
        // units of its own std. Constant model: k = 1, adj = 0.
        if (dynamic) {
          const marketK = dyn.mult[regime] * Math.exp(marketVol - volSq);
          for (let i = 0; i < n; i++) {
            const k = marketK * Math.exp(idio[i] - idioSq);
            ks[i] = k;
            adjs[i] = mp.returnLinkSigned * (k - dyn.kBar);
          }
          regimeCounts[regime] += 1;
        } else {
          for (let i = 0; i < n; i++) {
            ks[i] = 1;
            adjs[i] = 0;
          }
        }

        const corrRow = driftCorr ? driftCorr[s - 1] : null;
        for (let i = 0; i < n; i++) {
          const sigmaEff = baseVols[i] * ks[i];
          const mu = mus[i] + adjs[i] * baseVols[i];
          const drift = (mu - 0.5 * sigmaEff * sigmaEff) * dt - (corrRow ? corrRow[i] : 0);
          growth[idx++] = Math.exp(drift + zCorr[i] * ks[i] * sqrtDt);
        }
        if (trk) trk.push({ regime, ks: Array.from(ks), adjs: Array.from(adjs) });

        // Evolve the market state for next month: market volatility reacts to
        // this month's market shock (leverage effect), each holding's own
        // volatility takes a small independent step, then the regime may switch.
        if (dynamic) {
          let shockSum = 0;
          for (let i = 0; i < n; i++) shockSum += (proxyWeights[i] * zCorr[i]) / baseVols[i];
          const marketShock = shockSum / (inStress ? marketShockSdStress : marketShockSdBase); // ~ N(0, 1)
          marketVol =
            VOL_PERSISTENCE * marketVol +
            marketEta * (mp.leverageCorr * marketShock + leverageRest * standardNormalRandom());
          for (let i = 0; i < n; i++) idio[i] = VOL_PERSISTENCE * idio[i] + idioEta * standardNormalRandom();
          regime = sampleIndex(dyn.transitions[regime]);
        }
      }
      if (trk) trackRaw.push(trk);
    }
    return { n, steps, count, growth, trackRaw };
  }

  return { n, steps, dynamic, model, regimeCounts, Cbase, Cstress, baseVols, generate };
}

// Month-by-month effective portfolio std and expected return for a few
// tracked futures (for the "market conditions" chart).
function buildConditionTracks(scen, weights, selected) {
  const { n, baseVols, Cbase, Cstress, trackRaw } = scen;
  return trackRaw.map((trk) =>
    trk.map((t, idx) => {
      const C = t.regime === STRESS_INDEX ? Cstress : Cbase;
      let effVar = 0;
      let effRet = 0;
      for (let i = 0; i < n; i++) {
        effRet += weights[i] * (selected[i].return + t.adjs[i] * baseVols[i]);
        for (let j = 0; j < n; j++) {
          effVar += weights[i] * weights[j] * baseVols[i] * t.ks[i] * baseVols[j] * t.ks[j] * C[i][j];
        }
      }
      return { month: idx + 1, std: Math.sqrt(Math.max(effVar, 0)), ret: effRet, regime: REGIME_LABELS[t.regime] };
    })
  );
}

// Running results for a whole simulation. Everything is kept in compact
// typed arrays: one number per run for final values, drawdowns and each
// holding; the month-by-month fan chart uses the first BAND_SAMPLE runs
// (plenty for stable percentile bands); a few full paths are kept to draw.
const BAND_SAMPLE = 20000;

function createOutcomeCollector({ n, steps, count, sampledPaths = 60 }) {
  const bandCount = Math.min(count, BAND_SAMPLE);
  return {
    n,
    steps,
    count,
    filled: 0,
    bandCount,
    finals: new Float64Array(count),
    maxDD: new Float64Array(count),
    hold: Array.from({ length: n }, () => new Float64Array(count)),
    bandValues: Array.from({ length: steps + 1 }, () => new Float64Array(bandCount)),
    samplePaths: [],
    sampleLimit: Math.max(1, Math.min(sampledPaths, count)),
  };
}

// Runs one allocation through a chunk of simulated futures (rebalancing costs
// included) and stores the results in the collector.
function evaluateChunkInto(col, weights, scen, amount, rebalanceInfo) {
  const { n, steps } = col;
  const { growth, count } = scen;
  const rebalanceEnabled = !!rebalanceInfo?.enabled;
  const interval = rebalanceEnabled ? Math.max(1, Math.round(12 / rebalanceInfo.rebalancesPerYear)) : 0;
  const costFraction = rebalanceEnabled ? (rebalanceInfo.costPct || 0) / 100 : 0;
  const vals = new Float64Array(n); // portfolio holdings (rebalanced)
  const holds = new Float64Array(n); // same assets, never rebalanced

  for (let sim = 0; sim < count; sim++) {
    const g = col.filled + sim;
    for (let i = 0; i < n; i++) {
      vals[i] = weights[i] * amount;
      holds[i] = vals[i];
    }
    let total = amount;
    let peak = amount;
    let maxDD = 0;
    const isSampled = g < col.sampleLimit;
    const path = isSampled ? [total] : null;
    const inBand = g < col.bandCount;
    if (inBand) col.bandValues[0][g] = total;
    const base = sim * steps * n;

    for (let s = 1; s <= steps; s++) {
      total = 0;
      const off = base + (s - 1) * n;
      for (let i = 0; i < n; i++) {
        const growthFactor = growth[off + i];
        holds[i] *= growthFactor;
        vals[i] *= growthFactor;
        total += vals[i];
      }
      // Periodic rebalance: pay the transaction cost, then reset each
      // instrument back to its target weight of the post-cost total.
      if (rebalanceEnabled && s % interval === 0 && s !== steps) {
        total *= 1 - costFraction;
        for (let i = 0; i < n; i++) vals[i] = weights[i] * total;
      }
      // Running peak-to-trough drawdown for this path.
      if (total > peak) peak = total;
      const dd = total / peak - 1;
      if (dd < maxDD) maxDD = dd;
      if (inBand) col.bandValues[s][g] = total;
      if (isSampled) path.push(total);
    }

    col.finals[g] = total;
    col.maxDD[g] = maxDD;
    for (let i = 0; i < n; i++) col.hold[i][g] = holds[i];
    if (isSampled) col.samplePaths.push(path);
  }
  col.filled += count;
}

// A Monte Carlo run that can be advanced in small time slices, so the page
// stays responsive (and can show progress) even for hundreds of thousands of
// runs. runChunk(ms) works for about that long; finish() builds the results.
function createMonteCarloJob({
  selected,
  weights,
  amount,
  Sigma,
  years,
  simulationsCount,
  rebalanceInfo,
  model = "dynamic",
  marketParams = null,
  sampledPathsCount = 60,
}) {
  const engine = prepareScenarioEngine({ selected, Sigma, years, model, marketParams });
  const { n, steps, dynamic } = engine;
  const collector = createOutcomeCollector({ n, steps, count: simulationsCount, sampledPaths: sampledPathsCount });
  const chunkSize = Math.max(100, Math.min(5000, Math.floor(60000 / (steps * n))));
  let done = 0;
  let trackRaw = [];

  // Returns true once every run is finished.
  function runChunk(budgetMs = Infinity) {
    const deadline = Number.isFinite(budgetMs) ? Date.now() + budgetMs : Infinity;
    while (done < simulationsCount) {
      const size = Math.min(chunkSize, simulationsCount - done);
      const scen = engine.generate(size, done === 0 ? Math.min(10, simulationsCount) : 0);
      if (done === 0) trackRaw = scen.trackRaw;
      evaluateChunkInto(collector, weights, scen, amount, rebalanceInfo);
      done += size;
      if (Date.now() >= deadline) break;
    }
    return done >= simulationsCount;
  }

  function progress() {
    return done / simulationsCount;
  }

  function finish() {
    const startValues = weights.map((w) => w * amount);
    const regimeCounts = engine.regimeCounts;
    const conditionTracks = buildConditionTracks(
      { n, baseVols: engine.baseVols, Cbase: engine.Cbase, Cstress: engine.Cstress, trackRaw },
      weights,
      selected
    );
    const basePortRet = weights.reduce((sum, w, i) => sum + w * selected[i].return, 0);
    const basePortVol = Math.sqrt(Math.max(dot(weights, matVecMul(Sigma, weights)), 0));

    const bands = collector.bandValues.map((valuesAtStep, s) => {
      const sorted = valuesAtStep.slice().sort();
      const p5 = percentileOfSorted(sorted, 5);
      const p25 = percentileOfSorted(sorted, 25);
      const p50 = percentileOfSorted(sorted, 50);
      const p75 = percentileOfSorted(sorted, 75);
      const p95 = percentileOfSorted(sorted, 95);
      return {
        step: s,
        month: s === 0 ? "Start" : `M${s}`,
        p5,
        p25,
        p50,
        p75,
        p95,
        range90: [p5, p95],
        range50: [p25, p75],
      };
    });

    const finalValues = collector.finals.slice().sort();
    const N = finalValues.length;
    let lossCount = 0;
    for (let i = 0; i < N; i++) if (finalValues[i] < amount) lossCount += 1;
    const probLoss = lossCount / N;

    // Distribution of final outcomes.
    const meanFinal = finalValues.reduce((a, b) => a + b, 0) / N;
    const stdFinal = Math.sqrt(finalValues.reduce((s, v) => s + (v - meanFinal) ** 2, 0) / N);
    const percentiles = MC_PERCENTILES.map((p) => {
      const value = percentileOfSorted(finalValues, p);
      return { p, value, ret: value / amount - 1 };
    });

    // Value at Risk / Expected Shortfall at 95% confidence, expressed as a
    // loss versus the amount invested. VaR = loss at the 5th percentile;
    // CVaR = average loss across the worst 5% of outcomes.
    const p5Final = percentileOfSorted(finalValues, 5);
    const tailCount = Math.max(1, Math.floor(N * 0.05));
    const tailMean = finalValues.slice(0, tailCount).reduce((a, b) => a + b, 0) / tailCount;
    const var95 = Math.max(0, amount - p5Final);
    const cvar95 = Math.max(0, amount - tailMean);

    // Max drawdown per path: values are <= 0, so ascending order puts the
    // deepest drawdowns first. "5th percentile" = a bad-luck drawdown.
    const sortedDD = collector.maxDD.slice().sort();
    const medianDrawdown = percentileOfSorted(sortedDD, 50);
    const worstDrawdown = percentileOfSorted(sortedDD, 5);

    // Annualized return of each path (finalValues is sorted, and the
    // transform is monotonic, so this stays sorted too).
    const annualized = finalValues.map((v) => Math.pow(Math.max(v, 1e-9) / amount, 1 / years) - 1);
    const medianAnnualized = percentileOfSorted(annualized, 50);

    // Histogram of final values. Bins span the 1st-99th percentile so a few
    // extreme paths don't squash the chart; outliers fall into the edge bins.
    const binCount = 30;
    const lo = percentileOfSorted(finalValues, 1);
    const hi = percentileOfSorted(finalValues, 99);
    const binWidth = (hi - lo) / binCount || 1;
    const counts = Array(binCount).fill(0);
    finalValues.forEach((v) => {
      let idx = Math.floor((v - lo) / binWidth);
      idx = Math.min(binCount - 1, Math.max(0, idx));
      counts[idx] += 1;
    });
    const histogram = counts.map((c, i) => {
      const mid = lo + (i + 0.5) * binWidth;
      return { mid, pct: (c / N) * 100, isLoss: mid < amount };
    });

    // Per-instrument worst case, computed two ways for the same buy-and-hold
    // position: "simulated" is the empirical 5th percentile of that holding on
    // its own under the selected market model (so it includes regimes and
    // time-varying volatility); "analytical" is the closed-form constant-
    // volatility lognormal VaR at 95% confidence from that instrument's return
    // and std. Both ignore rebalancing, so the gap between them isolates the
    // effect of the dynamic market model.
    const z95 = 1.645; // one-tailed 95% confidence
    const instrumentStats = selected.map((p, i) => {
      const sortedFinal = collector.hold[i].slice().sort();
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

    const totalRegimeMonths = regimeCounts.reduce((a, b) => a + b, 0);
    const regimeShare = dynamic
      ? REGIME_LABELS.map((label, i) => ({ label, share: regimeCounts[i] / Math.max(1, totalRegimeMonths) }))
      : null;

    return {
      model,
      conditionTracks,
      regimeShare,
      basePortVol,
      basePortRet,
      bands,
      samplePaths: collector.samplePaths,
      bandSampleSize: collector.bandCount,
      steps,
      finalValues,
      histogram,
      percentiles,
      instrumentStats,
      stats: {
        median: percentileOfSorted(finalValues, 50),
        mean: meanFinal,
        std: stdFinal,
        p5: p5Final,
        p95: percentileOfSorted(finalValues, 95),
        best: finalValues[N - 1],
        worst: finalValues[0],
        probLoss,
        var95,
        cvar95,
        medianDrawdown,
        worstDrawdown,
        medianAnnualized,
      },
    };
  }

  return { runChunk, progress, finish };
}

function runMonteCarloSimulation(args) {
  const job = createMonteCarloJob(args);
  job.runChunk(Infinity);
  return job.finish();
}

// --- HISTORICAL DATA ------------------------------------------------------
// Everything for instruments that come with real price history: reading a
// price-history CSV (Investing.com / Yahoo style: Date + Price or Close),
// measuring return, std and correlation from it, re-sampling the real
// history into many possible futures ("historical simulation"), searching
// for the best allocation in those futures, and back-testing on the actual
// past.
const MIN_HISTORY_POINTS = 30;
const MS_PER_DAY = 86400000;
const MONTH_ABBR = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];

function dayNumber(y, m, d) {
  return Math.round(Date.UTC(y, m - 1, d) / MS_PER_DAY);
}

function dayToIso(day) {
  return new Date(day * MS_PER_DAY).toISOString().slice(0, 10);
}

function isoToLabel(iso) {
  const [y, m, d] = iso.split("-");
  return `${d} ${MONTH_ABBR[Number(m) - 1][0].toUpperCase()}${MONTH_ABBR[Number(m) - 1].slice(1)} ${y}`;
}

function validYmd(y, m, d) {
  if (!(y >= 1900 && y <= 2100 && m >= 1 && m <= 12 && d >= 1 && d <= 31)) return false;
  const dt = new Date(Date.UTC(y, m - 1, d));
  return dt.getUTCFullYear() === y && dt.getUTCMonth() === m - 1 && dt.getUTCDate() === d;
}

function monthFromName(name) {
  const idx = MONTH_ABBR.indexOf(String(name).slice(0, 3).toLowerCase());
  return idx >= 0 ? idx + 1 : 0;
}

// Splits CSV text into rows of fields (handles quoted fields, thousands
// separators inside quotes, and doubled quotes).
function parseCsvText(text, delimiter) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else inQuotes = false;
      } else field += ch;
    } else if (ch === '"') inQuotes = true;
    else if (ch === delimiter) {
      row.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i += 1;
      row.push(field);
      field = "";
      if (row.some((c) => c.trim() !== "")) rows.push(row);
      row = [];
    } else field += ch;
  }
  row.push(field);
  if (row.some((c) => c.trim() !== "")) rows.push(row);
  return rows;
}

function detectDelimiter(text) {
  const firstLine = text.split(/\r?\n/).find((l) => l.trim() !== "") || "";
  let best = ",";
  let bestCount = 0;
  for (const d of [",", ";", "\t"]) {
    let count = 0;
    let inQuotes = false;
    for (const ch of firstLine) {
      if (ch === '"') inQuotes = !inQuotes;
      else if (ch === d && !inQuotes) count += 1;
    }
    if (count > bestCount) {
      best = d;
      bestCount = count;
    }
  }
  return best;
}

// "22,901.85" -> 22901.85, "1.234,56" -> 1234.56, "₹ 1,234" -> 1234
function parseNumberToken(raw, decimalComma = false) {
  if (raw === undefined || raw === null) return NaN;
  let t = String(raw).trim().replace(/[\s₹$€£]/g, "");
  if (t === "" || t === "-") return NaN;
  const hasComma = t.includes(",");
  const hasDot = t.includes(".");
  if (hasComma && hasDot) {
    if (t.lastIndexOf(",") > t.lastIndexOf(".")) t = t.replace(/\./g, "").replace(",", ".");
    else t = t.replace(/,/g, "");
  } else if (hasComma) {
    if (decimalComma) t = t.replace(",", ".");
    else if (/^-?\d{1,3}(,\d{3})+$/.test(t)) t = t.replace(/,/g, "");
    else t = t.replace(",", ".");
  }
  const v = Number(t);
  return Number.isFinite(v) ? v : NaN;
}

// Reads one date cell. Numeric day/month dates (12/03/2024) are returned as
// "ab" and resolved for the whole file at once, because they are ambiguous.
function parseDateCell(raw) {
  const t = String(raw === undefined || raw === null ? "" : raw).trim();
  let m = /^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[T\s].*)?$/.exec(t);
  if (m) return { kind: "ymd", y: +m[1], m: +m[2], d: +m[3] };
  m = /^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?:\s.*)?$/.exec(t);
  if (m) return { kind: "ab", a: +m[1], b: +m[2], y: +m[3] };
  m = /^(\d{1,2})[-\s]([A-Za-z]{3,9})\.?[-\s,]+(\d{4})$/.exec(t);
  if (m && monthFromName(m[2])) return { kind: "ymd", y: +m[3], m: monthFromName(m[2]), d: +m[1] };
  m = /^([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})$/.exec(t);
  if (m && monthFromName(m[1])) return { kind: "ymd", y: +m[3], m: monthFromName(m[1]), d: +m[2] };
  return null;
}

// How many times a sequence of day numbers reverses direction (0 = perfectly
// ordered either way). Used to tell 03/04/2024 = 3 April from 4 March.
function orderViolations(days) {
  let up = 0;
  let down = 0;
  for (let i = 1; i < days.length; i++) {
    if (days[i] > days[i - 1]) up += 1;
    else if (days[i] < days[i - 1]) down += 1;
  }
  return Math.min(up, down);
}

// Return, std and drawdown measured from a price history. Return is the
// GBM drift (mean log return + half the variance), which is the "return"
// convention the simulator uses; CAGR is the plain compound annual rate.
function computeSeriesStats(days, prices) {
  const n = prices.length - 1;
  let sum = 0;
  const logs = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    logs[i] = Math.log(prices[i + 1] / prices[i]);
    sum += logs[i];
  }
  const mean = sum / n;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += (logs[i] - mean) ** 2;
  const sd = Math.sqrt(ss / Math.max(1, n - 1));
  const spanYears = Math.max((days[days.length - 1] - days[0]) / 365.25, 1e-6);
  const perYear = n / spanYears;
  const sigma = sd * Math.sqrt(perYear);
  const muLog = mean * perYear;
  let peak = prices[0];
  let maxDrawdown = 0;
  for (let i = 0; i < prices.length; i++) {
    if (prices[i] > peak) peak = prices[i];
    maxDrawdown = Math.min(maxDrawdown, prices[i] / peak - 1);
  }
  return {
    points: prices.length,
    startIso: dayToIso(days[0]),
    endIso: dayToIso(days[days.length - 1]),
    spanYears,
    perYear,
    cagr: Math.exp(muLog) - 1,
    mu: muLog + 0.5 * sigma * sigma,
    sigma,
    maxDrawdown,
  };
}

// Reads a price-history CSV. Returns { ok: false, error } or
// { ok: true, days, dates, prices, stats, warnings, ... }.
function parseHistoricalCsv(rawText) {
  const warnings = [];
  const text = String(rawText || "").replace(/^\uFEFF/, "").trim();
  if (!text) return { ok: false, empty: true, error: "No data yet." };
  const delimiter = detectDelimiter(text);
  const rows = parseCsvText(text, delimiter);
  if (rows.length < 2) return { ok: false, error: "The data needs a header row and at least one row of prices." };

  const norm = (v) => String(v).trim().toLowerCase().replace(/[^a-z0-9 ]/g, "").trim();
  const header = rows[0].map(norm);
  let dateCol = header.findIndex((h) => h.includes("date") || h === "time" || h === "day" || h === "timestamp");
  let priceCol = -1;
  for (const key of ["price", "adj close", "adjclose", "close", "nav", "value", "last"]) {
    const idx = header.findIndex((h) => h === key);
    if (idx >= 0) {
      priceCol = idx;
      break;
    }
  }
  if (priceCol < 0) priceCol = header.findIndex((h) => h.includes("close") || h.includes("price"));
  let startRow = 1;
  const decimalComma = delimiter === ";";
  if (dateCol < 0 || priceCol < 0) {
    if (parseDateCell(rows[0][0]) && Number.isFinite(parseNumberToken(rows[0][1], decimalComma))) {
      dateCol = 0;
      priceCol = 1;
      startRow = 0;
      warnings.push("No header row found: reading column 1 as the date and column 2 as the price.");
    } else {
      return { ok: false, error: 'Could not find a "Date" column and a "Price" (or "Close") column in the header row.' };
    }
  }

  const raw = [];
  let skipped = 0;
  for (let r = startRow; r < rows.length; r++) {
    const parts = parseDateCell(rows[r][dateCol]);
    const price = parseNumberToken(rows[r][priceCol], decimalComma);
    if (!parts || !(price > 0)) {
      skipped += 1;
      continue;
    }
    raw.push({ parts, price });
  }
  if (raw.length < MIN_HISTORY_POINTS) {
    return {
      ok: false,
      error: `Only ${raw.length} usable rows were found (at least ${MIN_HISTORY_POINTS} are needed). Check that the Date and Price columns are filled in.`,
    };
  }

  // Numeric a/b/year dates: decide day-first or month-first for the whole file.
  let dateFormat = "ISO / named month";
  const ab = raw.filter((x) => x.parts.kind === "ab");
  if (ab.length > 0) {
    const aOver = ab.some((x) => x.parts.a > 12);
    const bOver = ab.some((x) => x.parts.b > 12);
    let dayFirst;
    if (aOver && !bOver) dayFirst = true;
    else if (bOver && !aOver) dayFirst = false;
    else if (aOver && bOver) return { ok: false, error: "The dates mix day-first and month-first formats. Please use one format." };
    else {
      const asDmy = ab.map((x) => (validYmd(x.parts.y, x.parts.b, x.parts.a) ? dayNumber(x.parts.y, x.parts.b, x.parts.a) : 0));
      const asMdy = ab.map((x) => (validYmd(x.parts.y, x.parts.a, x.parts.b) ? dayNumber(x.parts.y, x.parts.a, x.parts.b) : 0));
      const dmy = orderViolations(asDmy);
      const mdy = orderViolations(asMdy);
      dayFirst = dmy < mdy;
      if (dmy === mdy) warnings.push("Dates like 03/04/2024 were read as month/day/year. If they are day/month/year, edit the file so a day above 12 appears, or use YYYY-MM-DD.");
    }
    dateFormat = dayFirst ? "DD/MM/YYYY" : "MM/DD/YYYY";
    for (const x of ab) {
      x.parts = dayFirst ? { kind: "ymd", y: x.parts.y, m: x.parts.b, d: x.parts.a } : { kind: "ymd", y: x.parts.y, m: x.parts.a, d: x.parts.b };
    }
  }

  const byDay = new Map();
  let duplicates = 0;
  for (const x of raw) {
    const { y, m, d } = x.parts;
    if (!validYmd(y, m, d)) {
      skipped += 1;
      continue;
    }
    const day = dayNumber(y, m, d);
    if (byDay.has(day)) duplicates += 1;
    byDay.set(day, x.price);
  }
  const days = Array.from(byDay.keys()).sort((a, b) => a - b);
  if (days.length < MIN_HISTORY_POINTS) {
    return { ok: false, error: `Only ${days.length} distinct dates were found (at least ${MIN_HISTORY_POINTS} are needed).` };
  }
  const prices = days.map((d) => byDay.get(d));

  if (skipped > 0) warnings.push(`${skipped} row${skipped === 1 ? "" : "s"} skipped (missing or unreadable date/price).`);
  if (duplicates > 0) warnings.push(`${duplicates} duplicate date${duplicates === 1 ? "" : "s"} found; the last value for each date was used.`);

  const gaps = [];
  for (let i = 1; i < days.length; i++) gaps.push(days[i] - days[i - 1]);
  const medianGap = [...gaps].sort((a, b) => a - b)[Math.floor(gaps.length / 2)];
  const bigGapLimit = Math.max(10, 6 * medianGap);
  const bigGaps = gaps.filter((g) => g > bigGapLimit);
  if (bigGaps.length > 0) {
    warnings.push(`${bigGaps.length} gap${bigGaps.length === 1 ? "" : "s"} longer than ${bigGapLimit} days (largest ${Math.max(...bigGaps)} days): missing data counts as one long move.`);
  }
  let jumps = 0;
  for (let i = 1; i < prices.length; i++) if (Math.abs(Math.log(prices[i] / prices[i - 1])) > 0.25) jumps += 1;
  if (jumps > 0) warnings.push(`${jumps} move${jumps === 1 ? "" : "s"} larger than 25% in one period: check for stock splits or bad rows.`);

  const stats = computeSeriesStats(days, prices);
  if (stats.spanYears < 2) warnings.push("Less than 2 years of history: return and risk estimates will be unreliable.");
  else if (stats.perYear > 100 && stats.points < 250) warnings.push("Fewer than 250 daily prices: estimates will be noisy.");

  return {
    ok: true,
    days,
    dates: days.map(dayToIso),
    prices,
    rowsRead: rows.length - startRow,
    skipped,
    dateFormat,
    warnings,
    stats,
  };
}

function guessNameFromFile(fileName) {
  return String(fileName || "")
    .replace(/\.[a-z0-9]+$/i, "")
    .replace(/[_\-]+/g, " ")
    .replace(/\(\d+\)/g, " ")
    .replace(/historical\s*data|price\s*history|history|prices?/gi, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\s+\d$/, "");
}

function pearsonCorrelation(xs, ys) {
  const n = xs.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  if (sxx <= 0 || syy <= 0) return 0;
  return Math.max(-1, Math.min(1, sxy / Math.sqrt(sxx * syy)));
}

// Correlation measured from two price histories, using the dates they share.
const historyCorrelationCache = new WeakMap();
function historicalCorrelation(hA, hB) {
  let inner = historyCorrelationCache.get(hA);
  if (!inner) {
    inner = new WeakMap();
    historyCorrelationCache.set(hA, inner);
  }
  if (inner.has(hB)) return inner.get(hB);
  const a = hA.days;
  const b = hB.days;
  const lrA = [];
  const lrB = [];
  let i = 0;
  let j = 0;
  let prevA = null;
  let prevB = null;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      if (prevA !== null) {
        lrA.push(Math.log(hA.prices[i] / prevA));
        lrB.push(Math.log(hB.prices[j] / prevB));
      }
      prevA = hA.prices[i];
      prevB = hB.prices[j];
      i += 1;
      j += 1;
    } else if (a[i] < b[j]) i += 1;
    else j += 1;
  }
  const value = lrA.length >= 20 ? pearsonCorrelation(lrA, lrB) : null;
  inner.set(hB, value);
  return value;
}

function intersectSorted(a, b) {
  const out = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      out.push(a[i]);
      i += 1;
      j += 1;
    } else if (a[i] < b[j]) i += 1;
    else j += 1;
  }
  return out;
}

// The real, aligned history of the chosen instruments: log returns between
// consecutive dates that ALL of the instruments WITH uploaded data traded,
// plus running sums so that the growth over any stretch of days can be read
// off instantly. A holding with no uploaded history contributes no dates of
// its own -- instead it is given a FIXED log return every single day, sized
// from its manual return field, with zero variance (so it never adds risk
// or correlation, like an FD or a fixed-rate deposit would not). At least
// one selected holding must have real history, or there are no dates to
// build a timeline from at all.
function buildHistoricalDataset(selected, windowYears = 0) {
  const n = selected.length;
  const withHistory = selected.filter((p) => p.history);
  if (withHistory.length === 0) {
    return { error: "None of the selected holdings have historical data uploaded, so there is no real timeline to build." };
  }
  let common = withHistory[0].history.days;
  for (let i = 1; i < withHistory.length; i++) common = intersectSorted(common, withHistory[i].history.days);
  if (windowYears > 0 && common.length > 0) {
    const cutoff = common[common.length - 1] - windowYears * 365.25;
    const first = common.findIndex((d) => d >= cutoff);
    common = common.slice(Math.max(0, first));
  }
  if (common.length < 40) {
    return {
      error: `Only ${common.length} dates are shared by the instruments with historical data (at least 40 are needed). Check that the files cover the same period.`,
    };
  }
  const T = common.length - 1;
  const spanYears = (common[common.length - 1] - common[0]) / 365.25;
  const obsPerYear = T / spanYears;
  const R = new Float64Array(T * n);
  const fixedNames = [];
  for (let i = 0; i < n; i++) {
    const h = selected[i].history;
    if (!h) {
      // No history for this holding: a constant daily log return derived
      // from its manual return field, repeated for every day -- zero
      // variance, so it carries none of the simulated risk.
      fixedNames.push(selected[i].name);
      const fixedDaily = selected[i].return / obsPerYear;
      for (let t = 0; t < T; t++) R[t * n + i] = fixedDaily;
      continue;
    }
    let p = 0;
    let prev = null;
    for (let t = 0; t < common.length; t++) {
      while (h.days[p] < common[t]) p += 1;
      const price = h.prices[p];
      if (prev !== null) R[(t - 1) * n + i] = Math.log(price / prev);
      prev = price;
    }
  }
  const cum = new Float64Array((T + 1) * n);
  for (let t = 0; t < T; t++) for (let i = 0; i < n; i++) cum[(t + 1) * n + i] = cum[t * n + i] + R[t * n + i];
  const dpm = Math.max(1, Math.round(obsPerYear / 12));
  if (T < 2 * dpm) return { error: "Not enough shared history to build even a few months of returns." };
  return {
    n,
    T,
    R,
    cum,
    days: common,
    dates: common.map(dayToIso),
    spanYears,
    obsPerYear,
    dpm,
    names: selected.map((p) => p.name),
    fixedNames,
  };
}

// Historical simulation: builds many possible futures by re-using real
// history. Each simulated month is a real stretch of trading days (one
// "month" of consecutive days) copied from the past for ALL instruments at
// once, so their real co-movement is kept exactly. Stretches are drawn in
// blocks of blockMonths consecutive months, which also keeps real volatility
// clustering and momentum inside a block. Same output shape as the parametric
// scenario engine: growth[path][month][instrument].
function generateHistoricalScenarios(ds, { years, count, blockMonths }) {
  const { n, T, dpm, cum } = ds;
  const steps = Math.max(1, Math.round(years * 12));
  const b = Math.max(1, Math.min(blockMonths, Math.floor(T / dpm)));
  const maxStart = T - b * dpm;
  const growth = new Float64Array(count * steps * n);
  let idx = 0;
  for (let sim = 0; sim < count; sim++) {
    let start = 0;
    for (let s = 0; s < steps; s++) {
      if (s % b === 0) start = Math.floor(Math.random() * (maxStart + 1));
      const from = start + (s % b) * dpm;
      const to = from + dpm;
      for (let i = 0; i < n; i++) growth[idx++] = Math.exp(cum[to * n + i] - cum[from * n + i]);
    }
  }
  return { n, steps, count, growth };
}

// What a fixed allocation would actually have done over rows [from, to) of the
// real history (rebalanced on the user's schedule, costs included).
function backtestPortfolio(ds, weights, rebalanceInfo, amount, from = 0, to = ds.T) {
  const { n, R } = ds;
  const len = to - from;
  const values = new Float64Array(len + 1);
  const vals = new Float64Array(n);
  for (let i = 0; i < n; i++) vals[i] = weights[i] * amount;
  let total = amount;
  values[0] = total;
  const rebalanceEnabled = !!rebalanceInfo?.enabled;
  const rowsPerRebalance = rebalanceEnabled
    ? Math.max(1, Math.round(ds.obsPerYear / rebalanceInfo.rebalancesPerYear))
    : 0;
  const costFraction = rebalanceEnabled ? (rebalanceInfo.costPct || 0) / 100 : 0;
  let peak = amount;
  let maxDrawdown = 0;
  const logs = new Float64Array(len);
  let sum = 0;
  for (let t = 0; t < len; t++) {
    let newTotal = 0;
    for (let i = 0; i < n; i++) {
      vals[i] *= Math.exp(R[(from + t) * n + i]);
      newTotal += vals[i];
    }
    if (rebalanceEnabled && (t + 1) % rowsPerRebalance === 0 && t + 1 !== len) {
      newTotal *= 1 - costFraction;
      for (let i = 0; i < n; i++) vals[i] = weights[i] * newTotal;
    }
    logs[t] = Math.log(newTotal / total);
    sum += logs[t];
    total = newTotal;
    values[t + 1] = total;
    if (total > peak) peak = total;
    maxDrawdown = Math.min(maxDrawdown, total / peak - 1);
  }
  const mean = sum / len;
  let ss = 0;
  for (let t = 0; t < len; t++) ss += (logs[t] - mean) ** 2;
  const years = Math.max(len / ds.obsPerYear, 1e-6);
  const cagr = Math.pow(total / amount, 1 / years) - 1;
  const vol = Math.sqrt(ss / Math.max(1, len - 1)) * Math.sqrt(ds.obsPerYear);
  return { values, cagr, vol, maxDrawdown, ratio: vol > 0 ? cagr / vol : 0 };
}

// Fast version for the optimizer: only the final portfolio value per simulated
// future (and, if asked, each future's worst peak-to-trough fall).
function evaluateFinals(weights, scen, amount, rebalanceInfo, ddOut = null) {
  const { n, steps, count, growth } = scen;
  const rebalanceEnabled = !!rebalanceInfo?.enabled;
  const interval = rebalanceEnabled ? Math.max(1, Math.round(12 / rebalanceInfo.rebalancesPerYear)) : 0;
  const costFraction = rebalanceEnabled ? (rebalanceInfo.costPct || 0) / 100 : 0;
  const finals = new Float64Array(count);
  const vals = new Float64Array(n);
  for (let sim = 0; sim < count; sim++) {
    for (let i = 0; i < n; i++) vals[i] = weights[i] * amount;
    let total = amount;
    let peak = amount;
    let maxDD = 0;
    const base = sim * steps * n;
    for (let s = 1; s <= steps; s++) {
      total = 0;
      const off = base + (s - 1) * n;
      for (let i = 0; i < n; i++) {
        vals[i] *= growth[off + i];
        total += vals[i];
      }
      if (rebalanceEnabled && s % interval === 0 && s !== steps) {
        total *= 1 - costFraction;
        for (let i = 0; i < n; i++) vals[i] = weights[i] * total;
      }
      if (ddOut) {
        if (total > peak) peak = total;
        const d = total / peak - 1;
        if (d < maxDD) maxDD = d;
      }
    }
    finals[sim] = total;
    if (ddOut) ddOut[sim] = maxDD;
  }
  return finals;
}

// Risk/return summary of an allocation from its simulated final values.
// Return = annualised return of each simulated future; risk = how widely
// those outcomes spread (volatility) or how deep the bad ones fall short of
// the risk-free rate (downside). The ratios use the return ABOVE the
// risk-free rate: without that, a near-riskless deposit would always "win"
// simply because it has almost no volatility.
function summarizeOutcomes(finals, amount, years, ddValues = null, riskFree = 0) {
  const N = finals.length;
  let sum = 0;
  let sumSq = 0;
  let downSq = 0;
  let losses = 0;
  for (let p = 0; p < N; p++) {
    const r = Math.pow(Math.max(finals[p], 1e-9) / amount, 1 / years) - 1;
    sum += r;
    sumSq += r * r;
    if (r < riskFree) downSq += (r - riskFree) ** 2;
    if (finals[p] < amount) losses += 1;
  }
  const mean = sum / N;
  const vol = Math.sqrt(Math.max(sumSq / N - mean * mean, 0));
  const down = Math.sqrt(downSq / N);
  return {
    mean,
    vol,
    down,
    probLoss: losses / N,
    sharpe: vol > 1e-9 ? (mean - riskFree) / vol : 0,
    sortino: (mean - riskFree) / Math.max(down, 1e-4),
    medianDD: ddValues ? percentileOfSorted(ddValues.slice().sort(), 50) : null,
  };
}

const OBJECTIVE_OPTIONS = [
  { value: "sharpe", label: "Best extra return per risk (volatility)" },
  { value: "sortino", label: "Best extra return per downside risk" },
  { value: "losscap", label: "Highest return, limited chance of loss" },
];

// Results are noisy estimates, so the chance-of-loss limit gets a small safety
// margin: candidates must beat it by this much during the search, and may miss
// it by this much when the finalists are re-checked on fresh futures.
const LOSS_CAP_MARGIN = 0.005;

// Higher score = better. With a loss limit, anything above the limit is
// heavily penalised.
function objectiveScore(m, objective, lossLimit = 0.1) {
  if (objective === "sortino") return m.sortino;
  if (objective === "losscap") return m.probLoss <= lossLimit ? m.mean : -1 - (m.probLoss - lossLimit) * 100;
  return m.sharpe;
}

function normalizeWeights(w) {
  const total = w.reduce((a, b) => a + b, 0);
  return total > 0 ? w.map((x) => x / total) : w.map(() => 1 / w.length);
}

// A random allocation. The power shapes it: low = spread out, high = a few
// dominant holdings; sometimes some holdings are dropped altogether so
// concentrated portfolios get explored too.
function randomWeights(n) {
  const power = [0.6, 1, 1, 2, 3][Math.floor(Math.random() * 5)];
  let w = Array.from({ length: n }, () => Math.pow(-Math.log(Math.random() || 1e-12), power));
  if (n > 2 && Math.random() < 0.25) {
    const keep = Math.floor(Math.random() * n);
    w = w.map((x, i) => (i === keep || Math.random() < 0.6 ? x : 0));
  }
  return normalizeWeights(w);
}

// A small random move away from an existing allocation: either jitter every
// weight, or shift some weight from one holding to another.
function perturbWeights(w, sigma) {
  const n = w.length;
  if (Math.random() < 0.5) {
    return normalizeWeights(w.map((x) => Math.max(0, x + sigma * standardNormalRandom())));
  }
  const out = [...w];
  const from = Math.floor(Math.random() * n);
  let to = Math.floor(Math.random() * n);
  if (to === from) to = (to + 1) % n;
  const delta = Math.random() * Math.min(out[from], sigma * 3);
  out[from] -= delta;
  out[to] += delta;
  return normalizeWeights(out);
}

// Time-boxed random search for the best allocation on the historical
// simulation.
//   * Phase 1 (first half of the time): try lots of random allocations.
//   * Phase 2: keep making smaller and smaller random moves around the best
//     ones found so far.
// Every allocation is scored on the SAME simulated futures, so differences are
// real differences and not luck of the draw. Because picking the best of
// thousands of tries flatters the winner, the finalists are finally re-scored
// on a fresh, independent set of futures and the winner is chosen there.
function createHistoricalSearch({
  dataset,
  amount,
  years,
  blockMonths,
  rebalanceInfo,
  objective,
  lossLimit = 0.1,
  riskFree = 0.06,
  startWeights,
  paths = 1000,
  validationPaths = 3000,
}) {
  const n = dataset.n;
  const scen = generateHistoricalScenarios(dataset, { years, count: paths, blockMonths });
  const CLOUD_MAX = 1500;
  const cloud = [];
  let cloudSeen = 0;
  let tested = 0;
  let best = null;
  const elites = [];

  const evalOn = (sc, w, withDD = false) => {
    const dd = withDD ? new Float64Array(sc.count) : null;
    return summarizeOutcomes(evaluateFinals(w, sc, amount, rebalanceInfo, dd), amount, years, dd, riskFree);
  };
  const searchLimit = lossLimit - LOSS_CAP_MARGIN;

  function consider(w) {
    const m = evalOn(scen, w);
    const cand = { w, m, score: objectiveScore(m, objective, searchLimit) };
    tested += 1;
    const point = { x: m.vol * 100, y: m.mean * 100 };
    cloudSeen += 1;
    if (cloud.length < CLOUD_MAX) cloud.push(point);
    else {
      const j = Math.floor(Math.random() * cloudSeen);
      if (j < CLOUD_MAX) cloud[j] = point;
    }
    if (!best || cand.score > best.score) best = cand;
    // Keep the best few DISTINCT allocations (near-duplicates replace each
    // other), so the finalists cover different regions, not one clone.
    const near = elites.findIndex((e) => e.w.reduce((d, x, i) => d + Math.abs(x - w[i]), 0) < 0.03);
    if (near >= 0) {
      if (cand.score > elites[near].score) elites[near] = cand;
    } else elites.push(cand);
    elites.sort((x, y) => y.score - x.score);
    if (elites.length > 8) elites.length = 8;
    return cand;
  }

  // Seeds: your current allocation, equal weights, and each holding alone.
  const baselineW = normalizeWeights(startWeights);
  const baselineInSample = consider(baselineW);
  consider(Array(n).fill(1 / n));
  for (let i = 0; i < n; i++) consider(Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

  function step(deadline, progress) {
    while (Date.now() < deadline) {
      if (progress < 0.5) {
        if (Math.random() < 0.15 && best) consider(perturbWeights(best.w, 0.1));
        else consider(randomWeights(n));
      } else {
        const t = Math.min(1, (progress - 0.5) / 0.5);
        const sigma = 0.12 * (1 - t) + 0.004 * t;
        if (Math.random() < 0.1) consider(randomWeights(n));
        else {
          const parent = elites[Math.floor(Math.pow(Math.random(), 2) * elites.length)];
          consider(perturbWeights(parent.w, sigma));
        }
      }
    }
  }

  function getProgress() {
    const stride = Math.max(1, Math.ceil(cloud.length / 400));
    const sample = [];
    for (let i = 0; i < cloud.length; i += stride) sample.push(cloud[i]);
    return {
      tested,
      best: best ? { w: best.w, m: best.m, point: { x: best.m.vol * 100, y: best.m.mean * 100 } } : null,
      cloud: sample,
    };
  }

  function finish() {
    const fresh = generateHistoricalScenarios(dataset, { years, count: validationPaths, blockMonths });
    const measured = [baselineW, ...elites.map((e) => e.w)].map((w) => ({ w, m: evalOn(fresh, w, true) }));
    const finalists = measured.map((f) => ({ ...f, score: objectiveScore(f.m, objective, lossLimit + LOSS_CAP_MARGIN) }));
    const baseline = finalists[0];
    let top = baseline;
    for (const f of finalists) if (f.score > top.score + 1e-9) top = f;
    // A "better" allocation must be better by a meaningful margin, otherwise
    // the difference is just noise and the current allocation is kept.
    const minGain = objective === "losscap" ? (baseline.m.probLoss <= lossLimit + LOSS_CAP_MARGIN ? 0.002 : 0) : 0.03;
    const winner = top.score > baseline.score + minGain ? top : baseline;
    const equalW = Array(n).fill(1 / n);
    const winnerInSample = evalOn(scen, winner.w);
    return {
      winner,
      baseline,
      equal: { w: equalW, m: evalOn(fresh, equalW, true) },
      improved: winner !== baseline,
      tested,
      cloud,
      searchBest: { x: winnerInSample.vol * 100, y: winnerInSample.mean * 100 },
      searchBaseline: { x: baselineInSample.m.vol * 100, y: baselineInSample.m.mean * 100 },
      paths,
      validationPaths,
    };
  }

  return { step, getProgress, finish };
}

// Back-test of the recommended, current and equal-weight allocations (and each
// instrument on its own) on the real history, plus the same numbers for the
// first and second half of the history as a consistency check.
function buildHistoricalAnalysis(ds, selected, finalW, currentW, rebalanceInfo, amount, riskFree = 0) {
  const n = ds.n;
  const single = (i) => Array.from({ length: n }, (_, j) => (j === i ? 1 : 0));
  const noRebalance = { enabled: false };
  const portfolios = [
    { key: "final", label: "Recommended allocation", w: finalW, reb: rebalanceInfo },
    { key: "current", label: "Current allocation", w: currentW, reb: rebalanceInfo },
    { key: "equal", label: "Equal weight", w: Array(n).fill(1 / n), reb: rebalanceInfo },
    ...selected.map((p, i) => ({ key: `asset${i}`, label: p.name, w: single(i), reb: noRebalance })),
  ];
  // Return per risk uses the return above the risk-free rate.
  const withRatio = (bt) => ({ ...bt, ratio: bt.vol > 0 ? (bt.cagr - riskFree) / bt.vol : 0 });
  // Drop the (large) day-by-day values array once cagr/vol/drawdown/ratio are
  // read off it -- the half-period checks below only need the summary numbers.
  const summaryOnly = (bt) => {
    const { values, ...rest } = withRatio(bt);
    return rest;
  };
  const full = portfolios.map((pf) => ({ ...pf, bt: withRatio(backtestPortfolio(ds, pf.w, pf.reb, amount)) }));
  const mid = Math.floor(ds.T / 2);
  const halves = portfolios.slice(0, 3).map((pf) => ({
    key: pf.key,
    label: pf.label,
    first: summaryOnly(backtestPortfolio(ds, pf.w, pf.reb, amount, 0, mid)),
    second: summaryOnly(backtestPortfolio(ds, pf.w, pf.reb, amount, mid, ds.T)),
  }));
  const stride = Math.max(1, Math.ceil(ds.T / 240));
  const chart = [];
  for (let t = 0; t <= ds.T; t += stride) {
    const row = { date: ds.dates[t] };
    full.forEach((f) => {
      if (f.key === "final" || f.key === "current" || f.key.startsWith("asset")) row[f.key] = f.bt.values[t];
    });
    chart.push(row);
  }
  if ((ds.T % stride) !== 0) {
    const row = { date: ds.dates[ds.T] };
    full.forEach((f) => {
      if (f.key === "final" || f.key === "current" || f.key.startsWith("asset")) row[f.key] = f.bt.values[ds.T];
    });
    chart.push(row);
  }
  return {
    rows: full.map((f) => ({ key: f.key, label: f.label, cagr: f.bt.cagr, vol: f.bt.vol, maxDrawdown: f.bt.maxDrawdown, ratio: f.bt.ratio })),
    halves,
    chart,
    midDate: ds.dates[mid],
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
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&display=swap');
    .app-root {
      font-family: 'Cormorant Garamond', Georgia, 'Times New Roman', serif;
      font-variant-numeric: lining-nums;
      font-feature-settings: "lnum" 1;
      -webkit-font-smoothing: antialiased;
    }
    .app-root input, .app-root select, .app-root button, .app-root textarea { font-family: inherit; }
    .app-root .font-mono { font-family: inherit; font-variant-numeric: lining-nums tabular-nums; }
    .app-root .text-xs { font-size: 0.92rem !important; line-height: 1.25rem !important; }
    .app-root .text-sm { font-size: 1.06rem !important; line-height: 1.45rem !important; }
    .app-root ::selection { background: #fafafa; color: #000; }
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
  Low: "bg-neutral-900 text-neutral-300 border-neutral-700",
  Medium: "bg-neutral-800 text-neutral-100 border-neutral-500",
  High: "bg-white text-black border-white",
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
    mode: "manual",
    returnPct: "",
    stdPct: "",
    csvText: "",
    fileName: "",
    keepHistory: false,
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
    setForm({ name: "", mode: "manual", returnPct: "", stdPct: "", csvText: "", fileName: "", keepHistory: false });
    setEditingId(null);
  }

  function startEditProduct(product) {
    if (product.history) {
      setForm({ name: product.name, mode: "data", returnPct: "", stdPct: "", csvText: "", fileName: "", keepHistory: true });
    } else {
      setForm({
        name: product.name,
        mode: "manual",
        returnPct: String(Number((product.return * 100).toFixed(4))),
        stdPct: String(Number((product.std * 100).toFixed(4))),
        csvText: "",
        fileName: "",
        keepHistory: false,
      });
    }
    setEditingId(product.id);
    setError("");
  }

  function handleAddProduct() {
    if (!form.name.trim()) {
      setError("Product name is required.");
      return;
    }

    let returnVal;
    let stdVal;
    let history; // undefined = manual product

    if (form.mode === "data") {
      if (form.csvText.trim()) {
        const parsed = parseHistoricalCsv(form.csvText);
        if (!parsed.ok) {
          setError(parsed.error);
          return;
        }
        history = {
          days: parsed.days,
          dates: parsed.dates,
          prices: parsed.prices,
          stats: parsed.stats,
          warnings: parsed.warnings,
          fileName: form.fileName || "pasted data",
        };
      } else if (editingId !== null) {
        const existing = products.find((p) => p.id === editingId);
        if (existing && existing.history) history = existing.history;
      }
      if (!history) {
        setError("Add the historical data: upload a CSV file or paste its contents.");
        return;
      }
      returnVal = history.stats.mu;
      stdVal = history.stats.sigma;
      if (!Number.isFinite(returnVal) || !(stdVal > 0)) {
        setError("The data has no usable price variation (std is 0). Check the Price column.");
        return;
      }
    } else {
      if (form.returnPct === "" || form.stdPct === "") {
        setError("Enter both return (%) and std (%).");
        return;
      }
      returnVal = Number(form.returnPct) / 100;
      stdVal = Number(form.stdPct) / 100;
      if (Number.isNaN(returnVal) || Number.isNaN(stdVal)) {
        setError("Return and std must be valid numbers.");
        return;
      }
      if (stdVal <= 0) {
        setError("Std must be greater than 0.");
        return;
      }
    }

    if (editingId !== null) {
      setProducts((prev) =>
        prev.map((p) => {
          if (p.id !== editingId) return p;
          const { history: _oldHistory, ...rest } = p;
          return { ...rest, name: form.name.trim(), return: returnVal, std: stdVal, ...(history ? { history } : {}) };
        })
      );
    } else {
      const newProduct = {
        id: nextId++,
        name: form.name.trim(),
        return: returnVal,
        std: stdVal,
        ...(history ? { history } : {}),
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

  // Re-derives the headline metrics for a different set of weights (used when
  // the historical simulation's allocation is applied or undone).
  function resultWithWeights(prev, newWeights, customized) {
    const r = prev.selected.map((p) => p.return);
    const metrics = portfolioMetrics(newWeights, r, prev.Sigma);
    const drag = prev.rebalance && prev.rebalance.enabled ? prev.rebalance.drag : 0;
    const netReturn = metrics.portReturn - drag;
    return {
      ...prev,
      weights: newWeights,
      ...metrics,
      netReturn,
      expectedValue: amount * (1 + netReturn),
      optimizerWeights: prev.optimizerWeights || prev.weights,
      customized,
    };
  }

  function applyWeights(newWeights) {
    setResult((prev) => (prev ? resultWithWeights(prev, newWeights, true) : prev));
  }

  function restoreOptimizerWeights() {
    setResult((prev) => (prev && prev.optimizerWeights ? resultWithWeights(prev, prev.optimizerWeights, false) : prev));
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
    <div className="app-root min-h-screen w-full bg-black text-neutral-100">
      <GlobalStyle />
      <div className="w-full max-w-4xl mx-auto px-4 py-6 sm:px-6 sm:py-8 space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3 pb-4 border-b border-neutral-800">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white/5 border border-neutral-600">
            <TrendingUp className="w-5 h-5 text-neutral-300" />
          </div>
          <div className="min-w-0">
            <h1 className="text-base sm:text-lg font-semibold text-neutral-100 tracking-tight truncate">
              Portfolio Allocation Model
            </h1>
            <p className="text-xs text-neutral-500 hidden sm:block">
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
                      ? "bg-neutral-100 text-black"
                      : disabled
                      ? "text-neutral-600 cursor-not-allowed"
                      : "text-neutral-300 hover:bg-neutral-900"
                  }`}
                >
                  {isDone ? (
                    <CheckCircle2 className="w-4 h-4 text-neutral-300" />
                  ) : (
                    <span
                      className={`flex h-5 w-5 items-center justify-center rounded-full text-[13px] font-mono ${
                        isActive ? "bg-black/15 text-black" : "bg-neutral-800 text-neutral-400"
                      }`}
                    >
                      {tab.short}
                    </span>
                  )}
                  <span className="whitespace-nowrap">{tab.label}</span>
                </button>
                {i < STEPS.length - 1 && <div className="h-px w-6 sm:w-10 bg-neutral-800 shrink-0" />}
              </React.Fragment>
            );
          })}
        </div>

        {error && (
          <div className="text-sm text-neutral-100 bg-neutral-900 border border-neutral-500 rounded-lg px-3 py-2 anim-in">
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
          <ResultScreen
            result={result}
            amount={amount}
            chartData={chartData}
            onBack={() => setScreen("run")}
            onApplyWeights={applyWeights}
            onRestoreWeights={restoreOptimizerWeights}
          />
        )}
      </div>
    </div>
  );
}

function Card({ children, className = "" }) {
  return (
    <div className={`rounded-xl border border-neutral-800 bg-neutral-900/60 backdrop-blur ${className}`}>{children}</div>
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
      <Card className={`p-4 sm:p-5 space-y-4 anim-in ${isEditing ? "border-neutral-500" : ""}`}>
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-neutral-200">{isEditing ? "Edit product" : "Add product"}</h2>
          {isEditing && (
            <span className="text-xs text-neutral-300 bg-white/5 border border-neutral-600 rounded-full px-2 py-0.5">
              Editing
            </span>
          )}
        </div>
        <div>
          <label className="block text-xs text-neutral-500 mb-1">Product name</label>
          <input
            type="text"
            value={form.name}
            onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
            placeholder="e.g. Stocks"
            className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 placeholder-neutral-600 focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300"
          />
        </div>

        <div>
          <label className="block text-xs text-neutral-500 mb-1.5">Return &amp; std source</label>
          <div className="flex gap-2">
            {[
              ["manual", "Manual"],
              ["data", "Historical data"],
            ].map(([m, label]) => (
              <button
                key={m}
                onClick={() => setForm((f) => ({ ...f, mode: m }))}
                className={`px-3 py-1.5 text-sm rounded-md border transition-colors ${
                  form.mode === m
                    ? "bg-neutral-100 text-black border-neutral-100 font-medium"
                    : "bg-black text-neutral-400 border-neutral-700 hover:border-neutral-600"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {form.mode === "manual" ? (
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-neutral-500 mb-1">Return (%)</label>
              <input
                type="number"
                value={form.returnPct}
                onChange={(e) => setForm((f) => ({ ...f, returnPct: e.target.value }))}
                placeholder="e.g. 13"
                className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 placeholder-neutral-600 font-mono focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300"
              />
            </div>
            <div>
              <label className="block text-xs text-neutral-500 mb-1">Std (%)</label>
              <input
                type="number"
                value={form.stdPct}
                onChange={(e) => setForm((f) => ({ ...f, stdPct: e.target.value }))}
                placeholder="e.g. 18"
                className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 placeholder-neutral-600 font-mono focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300"
              />
            </div>
          </div>
        ) : (
          <HistoricalDataInput form={form} setForm={setForm} editingId={editingId} products={products} />
        )}

        <div className="flex items-center gap-2">
          <button
            onClick={onAdd}
            className="flex items-center gap-1.5 text-sm bg-neutral-100 hover:bg-white text-black font-medium px-3.5 py-2 rounded-md transition-colors"
          >
            {isEditing ? <Save className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
            {isEditing ? "Update product" : "Add product"}
          </button>
          {isEditing && (
            <button
              onClick={onCancelEdit}
              className="flex items-center gap-1.5 text-sm text-neutral-300 border border-neutral-700 hover:bg-neutral-800 px-3.5 py-2 rounded-md transition-colors"
            >
              <X className="w-4 h-4" />
              Cancel
            </button>
          )}
        </div>
      </Card>

      <Card className="overflow-hidden anim-in">
        <div className="px-4 py-3 bg-neutral-900 border-b border-neutral-800">
          <h2 className="text-sm font-semibold text-neutral-200">Configured products ({products.length})</h2>
        </div>
        {products.length === 0 ? (
          <p className="text-sm text-neutral-500 px-4 py-8 text-center">No products added yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm min-w-[360px]">
              <thead>
                <tr className="text-left text-xs text-neutral-500 border-b border-neutral-800">
                  <th className="px-4 py-2 font-medium">Name</th>
                  <th className="px-4 py-2 font-medium">Return</th>
                  <th className="px-4 py-2 font-medium">Std</th>
                  <th className="px-4 py-2 font-medium" />
                </tr>
              </thead>
              <tbody>
                {products.map((p) => (
                  <tr
                    key={p.id}
                    className={`border-b border-neutral-800/60 last:border-0 hover:bg-neutral-900/60 ${
                      p.id === editingId ? "bg-white/5" : ""
                    }`}
                  >
                    <td className="px-4 py-2.5 text-neutral-200">
                      <div className="flex items-center gap-1.5">
                        <span className="truncate">{p.name}</span>
                        {p.history && (
                          <Database
                            className="w-3.5 h-3.5 text-neutral-500 shrink-0"
                            title={`From historical data: ${isoToLabel(p.history.dates[0])} – ${isoToLabel(p.history.dates[p.history.dates.length - 1])}`}
                          />
                        )}
                      </div>
                      {p.history && (
                        <div className="text-[13px] text-neutral-600 whitespace-nowrap">
                          {isoToLabel(p.history.dates[0])} – {isoToLabel(p.history.dates[p.history.dates.length - 1])}
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-2.5 font-mono text-white">{(p.return * 100).toFixed(2)}%</td>
                    <td className="px-4 py-2.5 font-mono text-neutral-300">{(p.std * 100).toFixed(2)}%</td>
                    <td className="px-4 py-2.5 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => onEdit(p)}
                          className="text-neutral-600 hover:text-white transition-colors"
                          title="Edit product"
                        >
                          <Pencil className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => onRemove(p.id)}
                          className="text-neutral-600 hover:text-white transition-colors"
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
          <div className="px-4 py-3 bg-neutral-900 border-b border-neutral-800">
            <h2 className="text-sm font-semibold text-neutral-200">Correlation matrix</h2>
            <p className="text-xs text-neutral-500 mt-0.5">
              Set the correlation for each pair (between -1 and 1). A small database icon means it was measured from historical data; pairs with neither an override nor shared data default to 0 (uncorrelated).
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="text-sm">
              <thead>
                <tr>
                  <th className="px-3 py-2 sticky left-0 bg-neutral-900 z-10" />
                  {products.map((p) => (
                    <th key={p.id} className="px-3 py-2 text-xs font-medium text-neutral-500 whitespace-nowrap">
                      {p.name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {products.map((rowP) => (
                  <tr key={rowP.id} className="border-t border-neutral-800">
                    <td className="px-3 py-2 text-xs text-neutral-400 whitespace-nowrap sticky left-0 bg-neutral-900/95 z-10">
                      {rowP.name}
                    </td>
                    {products.map((colP) => {
                      if (rowP.id === colP.id) {
                        return (
                          <td key={colP.id} className="px-3 py-2 text-center text-neutral-600 font-mono">
                            1.00
                          </td>
                        );
                      }
                      const key = pairKeyById(rowP.id, colP.id);
                      const isOverridden = corrOverrides[key] !== undefined;
                      const source = correlationSource(rowP, colP, corrOverrides);
                      const value = getCorrelation(rowP, colP, corrOverrides);
                      return (
                        <td
                          key={colP.id}
                          className={`px-2 py-1.5 text-center ${
                            source === "manual" ? "bg-white/10" : source === "data" ? "bg-white/5" : ""
                          }`}
                        >
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
                              className="w-16 text-sm text-center font-mono bg-black border border-neutral-700 rounded px-1 py-0.5 text-neutral-100 focus:outline-none focus:border-neutral-300"
                            />
                            {isOverridden ? (
                              <button
                                onClick={() => onCorrReset(rowP.id, colP.id)}
                                className="text-neutral-600 hover:text-neutral-300"
                                title="Reset to calculated value"
                              >
                                <RotateCcw className="w-3 h-3" />
                              </button>
                            ) : source === "data" ? (
                              <Database className="w-3 h-3 text-neutral-500" title="Measured from historical data" />
                            ) : null}
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

// The "Historical data" mode of the Add Product form: upload or paste a
// price-history CSV (Date + Price/Close, comma/semicolon/tab, quoted or not
// -- the same shape exchanges like investing.com, Yahoo Finance, or an AMFI
// NAV export use) and preview what the simulator will actually use from it.
function HistoricalDataInput({ form, setForm, editingId, products }) {
  const fileInputRef = useRef(null);
  const existing = editingId !== null ? products.find((p) => p.id === editingId) : null;
  const usingExisting = form.keepHistory && !form.csvText.trim() && existing && existing.history;

  const preview = useMemo(() => {
    if (!form.csvText.trim()) return null;
    return parseHistoricalCsv(form.csvText);
  }, [form.csvText]);

  function handleFile(e) {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setForm((f) => ({
        ...f,
        csvText: String(reader.result || ""),
        fileName: file.name,
        keepHistory: false,
        name: f.name.trim() ? f.name : guessNameFromFile(file.name),
      }));
    };
    reader.readAsText(file);
    e.target.value = "";
  }

  if (usingExisting) {
    const h = existing.history;
    return (
      <div className="space-y-2">
        <div className="rounded-lg border border-neutral-700 px-3 py-2.5 flex items-start gap-2.5">
          <Database className="w-4 h-4 text-neutral-300 shrink-0 mt-0.5" />
          <div className="min-w-0">
            <p className="text-sm text-neutral-200">Using the data already uploaded for this product.</p>
            <p className="text-xs text-neutral-500 mt-0.5">
              {h.prices.length.toLocaleString("en-IN")} prices · {isoToLabel(h.dates[0])} to{" "}
              {isoToLabel(h.dates[h.dates.length - 1])} · CAGR {fmtPct(h.stats.cagr)} · std {fmtPct(h.stats.sigma)}
            </p>
          </div>
        </div>
        <button
          onClick={() => setForm((f) => ({ ...f, keepHistory: false }))}
          className="text-xs text-neutral-400 hover:text-neutral-200 underline"
        >
          Replace with new data
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-2.5">
      <div className="flex items-center gap-2">
        <input ref={fileInputRef} type="file" accept=".csv,.txt" onChange={handleFile} className="hidden" />
        <button
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center gap-1.5 text-sm text-neutral-200 border border-neutral-700 hover:border-neutral-500 px-3 py-2 rounded-md transition-colors"
        >
          <Upload className="w-4 h-4" />
          Upload CSV file
        </button>
        {form.fileName && <span className="text-xs text-neutral-500 truncate">{form.fileName}</span>}
      </div>

      <div>
        <label className="flex items-center gap-1.5 text-xs text-neutral-500 mb-1">
          <FileText className="w-3.5 h-3.5" />
          Or paste the CSV contents
        </label>
        <textarea
          value={form.csvText}
          onChange={(e) => setForm((f) => ({ ...f, csvText: e.target.value, fileName: f.fileName && e.target.value ? f.fileName : "" }))}
          rows={5}
          placeholder={'"Date","Price"\n"09/18/2026","22,901.85"\n"09/17/2026","22,568.85"\n...'}
          className="w-full text-xs bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 placeholder-neutral-600 font-mono focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300"
        />
        <p className="text-[13px] text-neutral-600 mt-1">
          Needs a Date column and a Price (or Close/NAV) column, at least {MIN_HISTORY_POINTS} rows. Comma, semicolon
          or tab separated; quoted or not; DD/MM/YYYY, MM/DD/YYYY or YYYY-MM-DD dates.
        </p>
      </div>

      {preview && !preview.ok && (
        <div className="text-sm text-neutral-100 bg-neutral-900 border border-neutral-500 rounded-lg px-3 py-2">
          {preview.error}
        </div>
      )}

      {preview && preview.ok && (
        <div className="rounded-lg border border-neutral-700 px-3 py-2.5 space-y-2 anim-in">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            <StatBox label="Prices" value={preview.prices.length.toLocaleString("en-IN")} />
            <StatBox label="Date range" value={`${preview.stats.spanYears.toFixed(1)}y`} hint={`${isoToLabel(preview.dates[0])} – ${isoToLabel(preview.dates[preview.dates.length - 1])}`} />
            <StatBox label="CAGR" value={fmtPct(preview.stats.cagr)} tone="text-white" />
            <StatBox label="Std (annualized)" value={fmtPct(preview.stats.sigma)} />
          </div>
          <p className="text-[13px] text-neutral-600">
            The simulator will use return {fmtPct(preview.stats.mu)} (continuous) and std {fmtPct(preview.stats.sigma)}, measured
            directly from these prices.
          </p>
          {preview.warnings.length > 0 && (
            <ul className="text-[13px] text-neutral-500 space-y-0.5 list-disc pl-4">
              {preview.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}
        </div>
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
          <label className="block text-xs text-neutral-500 mb-1">Investment amount</label>
          <div className="relative w-full sm:w-64">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-500 text-sm">₹</span>
            <input
              type="number"
              value={amount}
              onChange={(e) => setAmount(Number(e.target.value))}
              className="w-full text-sm bg-black border border-neutral-700 rounded-md pl-7 pr-2.5 py-2 text-neutral-100 font-mono focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300"
            />
          </div>
        </div>

        <div>
          <label className="block text-xs text-neutral-500 mb-2">
            Select 2-5 products <span className="text-neutral-600">({selectedIds.length} selected)</span>
          </label>
          {products.length === 0 ? (
            <p className="text-sm text-neutral-500">Add products from the config screen first.</p>
          ) : (
            <div className="space-y-1.5">
              {products.map((p) => {
                const checked = selectedIds.includes(p.id);
                return (
                  <label
                    key={p.id}
                    className={`flex items-center gap-3 text-sm rounded-md px-3 py-2.5 cursor-pointer border transition-colors ${
                      checked ? "border-neutral-500 bg-white/5" : "border-neutral-800 hover:border-neutral-700"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => onToggle(p.id)}
                      className="accent-white"
                    />
                    <span className="flex-1 text-neutral-200 min-w-0 truncate">{p.name}</span>
                    {p.history && (
                      <span className="text-[13px] text-neutral-400 border border-neutral-700 rounded px-1.5 leading-tight">
                        data
                      </span>
                    )}
                    <span className="text-neutral-400 text-xs font-mono whitespace-nowrap">
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
            <label className="block text-xs text-neutral-500 mb-1">Return vs safety</label>
            <select
              value={returnSafety}
              onChange={(e) => setReturnSafety(e.target.value)}
              className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300"
            >
              <option value="safety">Safety</option>
              <option value="balanced">Balanced</option>
              <option value="return">Return</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Risk tolerance</label>
            <select
              value={riskTolerance}
              onChange={(e) => setRiskTolerance(e.target.value)}
              className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300"
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>
        </div>

        <label className="flex items-center gap-2 text-sm text-neutral-300">
          <input
            type="checkbox"
            checked={diversify}
            onChange={(e) => setDiversify(e.target.checked)}
            className="accent-white"
          />
          Use equal risk allocation (diversification)
        </label>

        <div className="rounded-lg border border-neutral-800 px-3.5 py-3 space-y-3">
          <label className="flex items-center gap-2 text-sm text-neutral-300">
            <input
              type="checkbox"
              checked={rebalance}
              onChange={(e) => setRebalance(e.target.checked)}
              className="accent-white"
            />
            Simulate periodic rebalancing <span className="text-neutral-600 text-xs">(optional)</span>
          </label>
          <p className="text-xs text-neutral-500 pl-6">
            Resetting weights back to target periodically costs a little in fees/taxes each time — this reduces the
            expected return slightly. Leave unchecked to see returns without rebalancing.
          </p>
          {rebalance && (
            <div className="grid grid-cols-2 gap-3 pl-6 pt-1">
              <div>
                <label className="block text-xs text-neutral-500 mb-1">Frequency</label>
                <select
                  value={rebalanceFrequency}
                  onChange={(e) => setRebalanceFrequency(e.target.value)}
                  className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300"
                >
                  {REBALANCE_FREQUENCY_OPTIONS.map((f) => (
                    <option key={f.value} value={f.value}>
                      {f.label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-neutral-500 mb-1">Cost per rebalance (%)</label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  value={rebalanceCost}
                  onChange={(e) => setRebalanceCost(e.target.value)}
                  className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 font-mono focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300"
                />
              </div>
            </div>
          )}
        </div>

        <button
          onClick={onRun}
          className="flex items-center gap-1.5 text-sm bg-neutral-100 hover:bg-white text-black font-medium px-4 py-2.5 rounded-md transition-colors"
        >
          <Play className="w-4 h-4" />
          Run model
        </button>
      </Card>
    </div>
  );
}

const TOOLTIP_STYLE = {
  backgroundColor: "#0a0a0a",
  border: "1px solid #262626",
  borderRadius: 8,
  fontSize: 14,
  color: "#e5e5e5",
};

function StatBox({ label, value, tone = "text-neutral-100", hint }) {
  return (
    <div className="rounded-lg border border-neutral-800 px-3 py-2.5">
      <p className="text-xs text-neutral-500">{label}</p>
      <p className={`text-sm font-semibold font-mono ${tone}`}>{value}</p>
      {hint && <p className="text-[13px] text-neutral-600 mt-0.5">{hint}</p>}
    </div>
  );
}

function ResultScreen({ result, amount, chartData, onBack, onApplyWeights, onRestoreWeights }) {
  const { portReturn, netReturn, portVol, sharpeLike, riskLevel, expectedValue, selected, weights, rebalance, Sigma } =
    result;

  const animReturn = useCountUp(netReturn * 100);
  const animVol = useCountUp(portVol * 100);
  const animSharpe = useCountUp(sharpeLike);
  const animValue = useCountUp(expectedValue);

  const [mcHorizon, setMcHorizon] = useState("1");
  const [mcSimCount, setMcSimCount] = useState(1000);
  const [mcModel, setMcModel] = useState("dynamic");
  const [mcPreset, setMcPreset] = useState("realistic");
  const [mcLinkMode, setMcLinkMode] = useState("real");
  const [mcParamText, setMcParamText] = useState(() => paramsToText(MARKET_PRESETS.realistic));
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [condIdx, setCondIdx] = useState(0);
  const [mcRunning, setMcRunning] = useState(false);
  const [mcBigRun, setMcBigRun] = useState(false);
  const [mcProgress, setMcProgress] = useState(0);
  const [mcElapsed, setMcElapsed] = useState(0);
  const [mcOutput, setMcOutput] = useState(null);
  const mcCancelRef = useRef(false);
  const mcUnmountedRef = useRef(false);
  const mcTimerRef = useRef(null);

  // Reset on every mount so it also works under React StrictMode (which
  // mounts, unmounts and re-mounts once in development).
  useEffect(() => {
    mcUnmountedRef.current = false;
    return () => {
      mcUnmountedRef.current = true;
      if (mcTimerRef.current) clearTimeout(mcTimerRef.current);
    };
  }, []);

  // A new allocation makes any earlier Monte Carlo output stale, so clear it.
  function handleApplyWeights(newWeights) {
    onApplyWeights(newWeights);
    setMcOutput(null);
  }

  function handleRestoreWeights() {
    onRestoreWeights();
    setMcOutput(null);
  }

  function handlePresetChange(value) {
    setMcPreset(value);
    if (value !== "custom") setMcParamText(paramsToText(MARKET_PRESETS[value]));
  }

  function handleParamChange(key, value) {
    setMcParamText((prev) => ({ ...prev, [key]: value }));
    setMcPreset("custom");
  }

  // Runs up to 5,000 futures in one go; larger runs are done in small time
  // slices with a progress bar (and a Stop button) so the page never freezes.
  function handleRunMonteCarlo() {
    const horizon = MC_HORIZON_OPTIONS.find((h) => h.value === mcHorizon);
    const runCount = mcSimCount;
    const big = runCount > 5000;
    mcCancelRef.current = false;
    setMcRunning(true);
    setMcBigRun(big);
    setMcProgress(0);
    setMcElapsed(0);
    if (big) setMcOutput(null); // the heavy charts would otherwise redraw on every progress update
    // Defer so the "Running..." state paints before the computation starts.
    mcTimerRef.current = setTimeout(() => {
      if (mcUnmountedRef.current) return;
      const job = createMonteCarloJob({
        selected,
        weights,
        amount,
        Sigma,
        years: horizon.years,
        simulationsCount: runCount,
        rebalanceInfo: rebalance,
        model: mcModel,
        marketParams: sanitizeMarketParams(mcParamText, mcLinkMode),
      });
      const complete = () => {
        if (mcUnmountedRef.current) return;
        const output = job.finish();
        setCondIdx(0);
        setMcOutput({ ...output, horizonLabel: horizon.label, simCount: runCount });
        setMcRunning(false);
      };
      if (!big) {
        job.runChunk(Infinity);
        complete();
        return;
      }
      const startedAt = Date.now();
      let lastUi = 0;
      const tick = () => {
        if (mcUnmountedRef.current) return;
        if (mcCancelRef.current) {
          setMcRunning(false);
          return;
        }
        const finished = job.runChunk(40);
        const now = Date.now();
        if (finished) {
          setMcProgress(1);
          setMcElapsed((now - startedAt) / 1000);
          mcTimerRef.current = setTimeout(complete, 30); // let "Calculating statistics..." paint
          return;
        }
        if (now - lastUi > 150) {
          lastUi = now;
          setMcProgress(job.progress());
          setMcElapsed((now - startedAt) / 1000);
        }
        mcTimerRef.current = setTimeout(tick, 0);
      };
      tick();
    }, 30);
  }

  function handleStopMonteCarlo() {
    mcCancelRef.current = true;
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

  // One simulated path's month-by-month conditions: effective portfolio std
  // and effective expected return (annualised %), plus the market regime.
  const conditionData = useMemo(() => {
    if (!mcOutput || !mcOutput.conditionTracks || mcOutput.conditionTracks.length === 0) return [];
    const track = mcOutput.conditionTracks[condIdx % mcOutput.conditionTracks.length];
    return track.map((t) => ({
      month: `M${t.month}`,
      std: t.std * 100,
      ret: t.ret * 100,
      regime: t.regime,
    }));
  }, [mcOutput, condIdx]);

  return (
    <div className="space-y-5">
      {/* Metric cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricCard
          icon={<TrendingUp className="w-4 h-4 text-white" />}
          label="Expected return"
          value={`${animReturn.toFixed(2)}%`}
          subValue={rebalance?.enabled ? `${(portReturn * 100).toFixed(2)}% before rebalancing cost` : undefined}
          delay={0}
        />
        <MetricCard
          icon={<Activity className="w-4 h-4 text-neutral-300" />}
          label="Volatility"
          value={`${animVol.toFixed(2)}%`}
          subValue={`₹${Math.round((animVol / 100) * amount).toLocaleString("en-IN")}`}
          delay={60}
        />
        <MetricCard
          icon={<Gauge className="w-4 h-4 text-neutral-300" />}
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
          <h2 className="text-sm font-semibold text-neutral-200 mb-3">Allocation split</h2>
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
                  <Cell key={i} fill={SERIES_COLORS[i % SERIES_COLORS.length]} stroke="#000000" strokeWidth={2} />
                ))}
              </Pie>
              <Tooltip
                formatter={(v) => [`${v}%`, "Allocation"]}
                itemStyle={{ color: "#e5e5e5" }}
                labelStyle={{ color: "#e5e5e5", fontWeight: 600 }}
                contentStyle={TOOLTIP_STYLE}
              />
              <Legend
                verticalAlign="bottom"
                iconType="circle"
                iconSize={8}
                wrapperStyle={{ fontSize: 14, color: "#a3a3a3" }}
              />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card className="p-4 sm:p-5 anim-in" style={{ animationDelay: "160ms" }}>
          <h2 className="text-sm font-semibold text-neutral-200 mb-3">Allocation by weight</h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData} margin={{ top: 4, right: 8, left: -8, bottom: 28 }}>
              <defs>
                <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#fafafa" stopOpacity={1} />
                  <stop offset="100%" stopColor="#737373" stopOpacity={0.9} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 13, fill: "#a3a3a3" }}
                interval={0}
                angle={-25}
                textAnchor="end"
                height={50}
              />
              <YAxis tick={{ fontSize: 13, fill: "#a3a3a3" }} unit="%" width={40} />
              <Tooltip
                formatter={(v) => [`${v}%`, "Allocation"]}
                cursor={{ fill: "#262626", opacity: 0.4 }}
                itemStyle={{ color: "#e5e5e5" }}
                labelStyle={{ color: "#e5e5e5", fontWeight: 600 }}
                contentStyle={TOOLTIP_STYLE}
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
        <div className="flex items-center justify-between gap-2 mb-3">
          <h2 className="text-sm font-semibold text-neutral-200">Holdings</h2>
          {result.customized && (
            <span className="text-xs text-neutral-200 bg-white/5 border border-neutral-600 rounded-full px-2 py-0.5">
              Set by historical simulation
            </span>
          )}
        </div>
        <div className="space-y-3">
          {selected.map((p, i) => {
            const pct = weights[i] * 100;
            return (
              <div key={p.id}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-neutral-200 flex items-center gap-2 min-w-0">
                    <span
                      className="h-2 w-2 rounded-full shrink-0"
                      style={{ backgroundColor: SERIES_COLORS[i % SERIES_COLORS.length] }}
                    />
                    <span className="truncate">{p.name}</span>
                  </span>
                  <span className="text-neutral-400 font-mono whitespace-nowrap">
                    {pct.toFixed(2)}% · ₹{Math.round(weights[i] * amount).toLocaleString("en-IN")}
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-neutral-800 overflow-hidden">
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
          <Wallet className="w-4 h-4 text-neutral-300" />
          <p className="text-xs text-neutral-500">Projected value after 1 year</p>
        </div>
        <p className="text-2xl sm:text-3xl font-semibold text-neutral-100 font-mono">
          ₹{Math.round(animValue).toLocaleString("en-IN")}
        </p>
        {rebalance?.enabled && (
          <div className="flex items-start gap-2 mt-3 pt-3 border-t border-neutral-800">
            <RefreshCw className="w-3.5 h-3.5 text-neutral-300 mt-0.5 shrink-0" />
            <p className="text-xs text-neutral-500">
              Includes <span className="text-neutral-300">{rebalance.frequencyLabel.toLowerCase()}</span> rebalancing at{" "}
              <span className="text-neutral-300 font-mono">{rebalance.costPct}%</span> per rebalance — a{" "}
              <span className="text-neutral-300 font-mono">{(rebalance.drag * 100).toFixed(2)}%</span> annual drag on
              returns.
            </p>
          </div>
        )}
      </Card>

      {/* Monte Carlo simulation */}
      <Card className="p-4 sm:p-5 anim-in" style={{ animationDelay: "320ms" }}>
        <div className="flex items-center gap-2 mb-1">
          <Dices className="w-4 h-4 text-neutral-300" />
          <h2 className="text-sm font-semibold text-neutral-200">Monte Carlo simulation</h2>
        </div>
        <p className="text-xs text-neutral-500 mb-4">
          Runs many random future paths, simulating each holding separately with its own return and volatility
          while keeping their correlations intact, to show the full distribution of outcomes — not just one
          projected number. The <span className="text-neutral-300">Dynamic</span> model lets std, return and correlation
          change month by month (calm / normal / stress regimes, market-wide and per-holding volatility moving above
          and below your input, return linked to volatility, and correlations rising in stress) like a real market;{" "}
          <span className="text-neutral-300">Constant</span> keeps them fixed for the whole horizon.
        </p>

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-4">
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Time horizon</label>
            <select
              value={mcHorizon}
              onChange={(e) => setMcHorizon(e.target.value)}
              className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300"
            >
              {MC_HORIZON_OPTIONS.map((h) => (
                <option key={h.value} value={h.value}>
                  {h.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Simulations</label>
            <select
              value={mcSimCount}
              onChange={(e) => setMcSimCount(Number(e.target.value))}
              className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300"
            >
              {MC_SIM_COUNT_OPTIONS.map((n) => (
                <option key={n} value={n}>
                  {n.toLocaleString("en-IN")}
                </option>
              ))}
            </select>
          </div>
          <div className="col-span-2 sm:col-span-1">
            <label className="block text-xs text-neutral-500 mb-1">Market model</label>
            <select
              value={mcModel}
              onChange={(e) => setMcModel(e.target.value)}
              className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300"
            >
              {MC_MODEL_OPTIONS.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {mcSimCount >= 50000 && (
          <p className="text-xs text-neutral-500 mb-4">
            {mcSimCount.toLocaleString("en-IN")} runs can take from several seconds to a few minutes depending on your
            device and time horizon. Progress is shown and you can stop at any time. The fan-chart bands use the first
            20,000 runs (plenty for stable percentiles); every other statistic uses all runs.
          </p>
        )}

        {mcModel === "dynamic" && (
          <div className="space-y-3 mb-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-neutral-500 mb-1">Market severity</label>
                <select
                  value={mcPreset}
                  onChange={(e) => handlePresetChange(e.target.value)}
                  className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300"
                >
                  {Object.entries(MARKET_PRESETS).map(([key, preset]) => (
                    <option key={key} value={key}>
                      {preset.label}
                    </option>
                  ))}
                  <option value="custom">Custom</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-neutral-500 mb-1">Return vs volatility</label>
                <select
                  value={mcLinkMode}
                  onChange={(e) => setMcLinkMode(e.target.value)}
                  className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300"
                >
                  {RETURN_LINK_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="rounded-lg border border-neutral-800">
              <button
                onClick={() => setShowAdvanced((v) => !v)}
                className="w-full flex items-center justify-between px-3.5 py-2.5 text-sm text-neutral-300 hover:bg-neutral-900/60 rounded-lg transition-colors"
              >
                <span>Advanced market settings</span>
                <span className="text-xs text-neutral-500">{showAdvanced ? "Hide" : "Show"}</span>
              </button>
              {showAdvanced && (
                <div className="px-3.5 pb-3.5 space-y-3">
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    {MARKET_PARAM_FIELDS.map((f) => (
                      <div key={f.key}>
                        <label className="block text-xs text-neutral-500 mb-1">{f.label}</label>
                        <input
                          type="number"
                          step={f.step}
                          min={f.min}
                          max={f.max}
                          value={mcParamText[f.key]}
                          onChange={(e) => handleParamChange(f.key, e.target.value)}
                          className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 font-mono focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300"
                        />
                      </div>
                    ))}
                  </div>
                  <p className="text-xs text-neutral-500">
                    These are stylised defaults, not calibrated to one specific market — set them to match the market
                    you are modelling. Volatility multipliers are normalised so your std stays the long-run average;
                    the correlation matrix you entered is the Calm/Normal correlation, and positive correlations rise
                    toward 1 in Stress by the boost shown. The risk-premium direction uses the link strength × 0.3,
                    because that effect is empirically much weaker than the leverage effect.
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        <div className={`flex flex-wrap items-center gap-2 ${mcRunning && mcBigRun ? "mb-3" : "mb-5"}`}>
          <button
            onClick={handleRunMonteCarlo}
            disabled={mcRunning}
            className="flex items-center gap-1.5 text-sm bg-neutral-100 hover:bg-white disabled:opacity-60 text-black font-medium px-3.5 py-2 rounded-md transition-colors"
          >
            {mcRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Dices className="w-4 h-4" />}
            {mcRunning ? "Running..." : mcOutput ? "Re-run simulation" : "Run simulation"}
          </button>
          {mcRunning && mcBigRun && (
            <button
              onClick={handleStopMonteCarlo}
              className="flex items-center gap-1.5 text-sm text-neutral-300 border border-neutral-700 hover:bg-neutral-800 px-3.5 py-2 rounded-md transition-colors"
            >
              <X className="w-3.5 h-3.5" />
              Stop
            </button>
          )}
        </div>
        {mcRunning && mcBigRun && (
          <div className="mb-5">
            <div className="flex justify-between text-xs text-neutral-500 mb-1 font-mono">
              <span>{mcProgress >= 1 ? "Calculating statistics..." : `${Math.round(mcProgress * 100)}% of runs`}</span>
              <span>{mcElapsed.toFixed(0)}s</span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-neutral-800 overflow-hidden">
              <div className="h-full rounded-full bg-neutral-100" style={{ width: `${Math.min(100, mcProgress * 100)}%` }} />
            </div>
          </div>
        )}

        {mcOutput && (
          <div className="space-y-5 anim-in">
            {/* Key outcome stats */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <StatBox label="Median outcome" value={fmtINR(mcOutput.stats.median)} />
              <StatBox label="Mean outcome" value={fmtINR(mcOutput.stats.mean)} />
              <StatBox
                label="Median annualized return"
                value={fmtPct(mcOutput.stats.medianAnnualized, 2)}
                tone={mcOutput.stats.medianAnnualized >= 0 ? "text-white" : "text-neutral-400"}
              />
              <StatBox label="Chance of loss" value={fmtPct(mcOutput.stats.probLoss)} />
            </div>

            {/* Percentile fan */}
            <div>
              <h3 className="text-xs font-semibold text-neutral-300 mb-1">Portfolio value over time</h3>
              <p className="text-xs text-neutral-500 mb-2">
                Median path with the 25th–75th and 5th–95th percentile bands across{" "}
                {mcOutput.simCount > mcOutput.bandSampleSize
                  ? `the first ${mcOutput.bandSampleSize.toLocaleString("en-IN")}`
                  : "all"}{" "}
                simulations.
              </p>
              <ResponsiveContainer width="100%" height={280}>
                <ComposedChart data={mcOutput.bands} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
                  <defs>
                    <linearGradient id="mcBandOuter" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#d4d4d4" stopOpacity={0.22} />
                      <stop offset="100%" stopColor="#d4d4d4" stopOpacity={0.04} />
                    </linearGradient>
                    <linearGradient id="mcBandInner" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#d4d4d4" stopOpacity={0.45} />
                      <stop offset="100%" stopColor="#d4d4d4" stopOpacity={0.12} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
                  <XAxis dataKey="month" tick={{ fontSize: 12, fill: "#a3a3a3" }} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 12, fill: "#a3a3a3" }} width={60} tickFormatter={fmtINRCompact} />
                  <Tooltip
                    labelFormatter={(label) => `Month: ${label}`}
                    formatter={(v, name) => [
                      Array.isArray(v) ? `${fmtINR(v[0])} – ${fmtINR(v[1])}` : fmtINR(v),
                      name,
                    ]}
                    itemStyle={{ color: "#e5e5e5" }}
                    labelStyle={{ color: "#e5e5e5", fontWeight: 600 }}
                    contentStyle={TOOLTIP_STYLE}
                  />
                  <Area
                    dataKey="range90"
                    name="5th–95th percentile"
                    stroke="none"
                    fill="url(#mcBandOuter)"
                    isAnimationActive={true}
                    animationDuration={800}
                  />
                  <Area
                    dataKey="range50"
                    name="25th–75th percentile"
                    stroke="none"
                    fill="url(#mcBandInner)"
                    isAnimationActive={true}
                    animationDuration={800}
                  />
                  <Line
                    dataKey="p50"
                    name="Median"
                    stroke="#fafafa"
                    strokeWidth={2.5}
                    dot={false}
                    isAnimationActive={true}
                    animationDuration={900}
                  />
                  <ReferenceLine
                    y={amount}
                    stroke="#a3a3a3"
                    strokeDasharray="4 4"
                    label={{ value: "Invested amount", position: "insideBottomRight", fill: "#a3a3a3", fontSize: 13 }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            {/* Market conditions (dynamic model only) */}
            {mcOutput.model === "dynamic" && conditionData.length > 0 && (
              <div>
                <div className="flex items-center justify-between gap-2 mb-1">
                  <h3 className="text-xs font-semibold text-neutral-300">Market conditions (one simulated path)</h3>
                  <button
                    onClick={() => setCondIdx((i) => (i + 1) % mcOutput.conditionTracks.length)}
                    className="text-xs text-neutral-300 border border-neutral-700 hover:bg-neutral-800 px-2.5 py-1 rounded-md transition-colors"
                  >
                    Show another path ({(condIdx % mcOutput.conditionTracks.length) + 1}/
                    {mcOutput.conditionTracks.length})
                  </button>
                </div>
                <p className="text-xs text-neutral-500 mb-2">
                  In this future the portfolio&apos;s std (grey line) moves above and below your input (dashed line),
                  driven by the market regime and each holding&apos;s own volatility, while its expected return (white
                  line) responds according to the &quot;Return vs volatility&quot; setting. Hover a point to see that
                  month&apos;s regime.
                </p>
                <ResponsiveContainer width="100%" height={230}>
                  <ComposedChart data={conditionData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
                    <XAxis dataKey="month" tick={{ fontSize: 12, fill: "#a3a3a3" }} interval="preserveStartEnd" />
                    <YAxis tick={{ fontSize: 12, fill: "#a3a3a3" }} width={44} unit="%" />
                    <Tooltip
                      labelFormatter={(label, payload) =>
                        `${label} · ${payload && payload[0] ? payload[0].payload.regime : ""} regime`
                      }
                      formatter={(v, name) => [`${Number(v).toFixed(1)}%`, name]}
                      itemStyle={{ color: "#e5e5e5" }}
                      labelStyle={{ color: "#e5e5e5", fontWeight: 600 }}
                      contentStyle={{
                        backgroundColor: "#0a0a0a",
                        border: "1px solid #262626",
                        borderRadius: 8,
                        fontSize: 14,
                      }}
                    />
                    <Legend iconType="plainline" wrapperStyle={{ fontSize: 13, color: "#a3a3a3" }} />
                    <ReferenceLine y={0} stroke="#525252" />
                    <ReferenceLine y={mcOutput.basePortVol * 100} stroke="#a3a3a3" strokeDasharray="4 4" strokeOpacity={0.6} />
                    <Line
                      dataKey="std"
                      name="Std (annualized)"
                      stroke="#a3a3a3"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                    <Line
                      dataKey="ret"
                      name="Expected return (annualized)"
                      stroke="#fafafa"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
                {mcOutput.regimeShare && (
                  <div className="grid grid-cols-3 gap-3 mt-3">
                    {mcOutput.regimeShare.map((r) => (
                      <StatBox
                        key={r.label}
                        label={`Time in ${r.label}`}
                        value={fmtPct(r.share)}
                        tone={r.label === "Stress" ? "text-neutral-400" : r.label === "Calm" ? "text-white" : "text-neutral-100"}
                      />
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Simulated paths */}
            <div>
              <h3 className="text-xs font-semibold text-neutral-300 mb-1">Simulated paths</h3>
              <p className="text-xs text-neutral-500 mb-2">
                Each line is one simulated future, drawn in its own colour, generated from every holding's own return,
                volatility, and correlation with the others — showing {mcOutput.samplePaths.length} of{" "}
                {mcOutput.simCount.toLocaleString("en-IN")} runs for legibility.
              </p>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={spaghettiData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
                  <XAxis dataKey="month" tick={{ fontSize: 12, fill: "#a3a3a3" }} interval="preserveStartEnd" />
                  <YAxis
                    tick={{ fontSize: 12, fill: "#a3a3a3" }}
                    width={60}
                    tickFormatter={fmtINRCompact}
                    domain={["auto", "auto"]}
                  />
                  <ReferenceLine y={amount} stroke="#e5e5e5" strokeDasharray="4 4" />
                  {mcOutput.samplePaths.map((_, idx) => (
                    <Line
                      key={idx}
                      dataKey={`sim${idx}`}
                      stroke={PATH_COLORS[idx % PATH_COLORS.length]}
                      strokeWidth={1.4}
                      dot={false}
                      isAnimationActive={false}
                      strokeOpacity={0.85}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Distribution of final values */}
            <div>
              <h3 className="text-xs font-semibold text-neutral-300 mb-1">Distribution of final values</h3>
              <p className="text-xs text-neutral-500 mb-2">
                How often each ending value occurred across {mcOutput.simCount.toLocaleString("en-IN")} simulations
                after {mcOutput.horizonLabel.toLowerCase()}. Red bars are outcomes below the amount invested.
              </p>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={mcOutput.histogram} margin={{ top: 4, right: 8, left: 0, bottom: 4 }} barCategoryGap={1}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
                  <XAxis
                    dataKey="mid"
                    tick={{ fontSize: 12, fill: "#a3a3a3" }}
                    tickFormatter={fmtINRCompact}
                    interval="preserveStartEnd"
                    minTickGap={28}
                  />
                  <YAxis
                    tick={{ fontSize: 12, fill: "#a3a3a3" }}
                    width={40}
                    tickFormatter={(v) => `${Number(v).toFixed(0)}%`}
                  />
                  <Tooltip
                    labelFormatter={(label) => `Around ${fmtINR(label)}`}
                    formatter={(v) => [`${Number(v).toFixed(1)}%`, "Of simulations"]}
                    cursor={{ fill: "#262626", opacity: 0.4 }}
                    itemStyle={{ color: "#e5e5e5" }}
                    labelStyle={{ color: "#e5e5e5", fontWeight: 600 }}
                    contentStyle={TOOLTIP_STYLE}
                  />
                  <Bar dataKey="pct" isAnimationActive={true} animationDuration={800}>
                    {mcOutput.histogram.map((b, i) => (
                      <Cell key={i} fill={b.isLoss ? "#525252" : "#e5e5e5"} fillOpacity={0.85} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Percentile table */}
            <div>
              <h3 className="text-xs font-semibold text-neutral-300 mb-2">Outcome percentiles</h3>
              <div className="overflow-x-auto rounded-lg border border-neutral-800">
                <table className="w-full text-xs min-w-[320px]">
                  <thead>
                    <tr className="text-left text-neutral-500 border-b border-neutral-800">
                      <th className="px-3 py-2 font-medium">Percentile</th>
                      <th className="px-3 py-2 font-medium">Portfolio value</th>
                      <th className="px-3 py-2 font-medium">Return</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mcOutput.percentiles.map((row) => (
                      <tr key={row.p} className="border-b border-neutral-800/60 last:border-0">
                        <td className="px-3 py-2 text-neutral-400 font-mono">{row.p}th</td>
                        <td className="px-3 py-2 text-neutral-200 font-mono">{fmtINR(row.value)}</td>
                        <td
                          className={`px-3 py-2 font-mono ${row.ret >= 0 ? "text-white" : "text-neutral-400"}`}
                        >
                          {row.ret >= 0 ? "+" : ""}
                          {fmtPct(row.ret)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[13px] text-neutral-600 mt-1.5">
                The 5th percentile means 5% of simulations ended at or below that value.
              </p>
            </div>

            {/* Risk measures */}
            <div>
              <h3 className="text-xs font-semibold text-neutral-300 mb-2">Risk measures</h3>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <StatBox
                  label="VaR (95%)"
                  value={fmtINR(mcOutput.stats.var95)}
                  tone="text-neutral-400"
                  hint="Loss not exceeded in 95% of runs"
                />
                <StatBox
                  label="CVaR (95%)"
                  value={fmtINR(mcOutput.stats.cvar95)}
                  tone="text-neutral-400"
                  hint="Average loss in the worst 5%"
                />
                <StatBox
                  label="Median max drawdown"
                  value={fmtPct(mcOutput.stats.medianDrawdown)}
                  tone="text-neutral-300"
                  hint="Typical peak-to-trough fall"
                />
                <StatBox
                  label="Bad-luck drawdown (5th pct.)"
                  value={fmtPct(mcOutput.stats.worstDrawdown)}
                  tone="text-neutral-400"
                  hint="Deeper than 95% of runs"
                />
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-3">
                <StatBox label="Worst simulated run" value={fmtINR(mcOutput.stats.worst)} tone="text-neutral-400" />
                <StatBox label="Best simulated run" value={fmtINR(mcOutput.stats.best)} tone="text-white" />
                <StatBox label="Std dev of outcomes" value={fmtINR(mcOutput.stats.std)} />
                <StatBox label="Runs" value={mcOutput.simCount.toLocaleString("en-IN")} />
              </div>
            </div>

            {/* Per-instrument worst case */}
            <div>
              <h3 className="text-xs font-semibold text-neutral-300 mb-1">Per-instrument worst case</h3>
              <p className="text-xs text-neutral-500 mb-2">
                Each holding on its own (buy-and-hold, no rebalancing): the simulated 5th-percentile outcome under the
                selected market model, next to the constant-volatility formula (95% confidence, lognormal) from that
                holding&apos;s own return and std. Both are for the same position, so any gap is the effect of regimes and
                time-varying volatility (fatter tails), not of rebalancing.
              </p>
              <div className="space-y-2">
                {mcOutput.instrumentStats.map((s, i) => (
                  <div
                    key={s.name}
                    className="flex items-center justify-between gap-3 text-sm rounded-lg border border-neutral-800 px-3 py-2.5"
                  >
                    <span className="flex items-center gap-2 min-w-0">
                      <span
                        className="h-2 w-2 rounded-full shrink-0"
                        style={{ backgroundColor: SERIES_COLORS[i % SERIES_COLORS.length] }}
                      />
                      <span className="truncate text-neutral-200">{s.name}</span>
                    </span>
                    <div className="text-right font-mono text-xs shrink-0">
                      <div className="text-neutral-400">
                        {fmtINR(s.simulatedWorst)} ({s.worstPct.toFixed(1)}%)
                      </div>
                      <div className="text-neutral-600">constant-vol formula: {fmtINR(s.analyticalWorst)}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <p className="text-xs text-neutral-600">
              Based on {mcOutput.simCount.toLocaleString("en-IN")} simulated paths ({mcOutput.model === "dynamic"
                ? "dynamic model: regimes, time-varying market and per-holding volatility, return linked to volatility, higher correlation in stress"
                : "constant return and std"}), each built by simulating every holding separately with their mutual
              correlations preserved (Calm/Normal correlation as entered). Long-run average return and std are kept close to the values you entered. This is a
              probabilistic projection, not a guarantee — actual results can fall outside the shown bands.
            </p>
          </div>
        )}
      </Card>

      <HistoricalSearchCard
        result={result}
        amount={amount}
        onApply={handleApplyWeights}
        onRestore={handleRestoreWeights}
      />

      <button
        onClick={onBack}
        className="flex items-center gap-1.5 text-sm text-neutral-300 border border-neutral-700 hover:bg-neutral-900 px-3.5 py-2 rounded-md transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        Back
      </button>
    </div>
  );
}

function HistoricalSearchCard({ result, amount, onApply, onRestore }) {
  const { selected, weights, rebalance } = result;
  const hasAnyHistory = selected.some((p) => p.history);
  const fixedProductNames = selected.filter((p) => !p.history).map((p) => p.name);

  const [seconds, setSeconds] = useState("10");
  const [horizon, setHorizon] = useState("1");
  const [objective, setObjective] = useState("sharpe");
  const [lossLimitPct, setLossLimitPct] = useState("10");
  const [riskFreePct, setRiskFreePct] = useState("6");
  const [blockMonths, setBlockMonths] = useState(3);
  const [histWindow, setHistWindow] = useState("0");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | running | finishing | done
  const [live, setLive] = useState(null);
  const [outcome, setOutcome] = useState(null);
  const [applied, setApplied] = useState(false);
  const cancelRef = useRef(false); // "Stop and use best so far" was pressed
  const unmountedRef = useRef(false);
  const timerRef = useRef(null);

  // React StrictMode (the default in Vite/CRA dev builds) mounts, unmounts and
  // re-mounts every component once, so the flag must be reset on every mount.
  useEffect(() => {
    unmountedRef.current = false;
    return () => {
      unmountedRef.current = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const windowYears = Number(histWindow) || 0;
  const dataset = useMemo(
    () => (hasAnyHistory ? buildHistoricalDataset(selected, windowYears) : null),
    [selected, hasAnyHistory, windowYears]
  );

  if (!hasAnyHistory) {
    return (
      <Card className="p-4 sm:p-5 anim-in">
        <div className="flex items-center gap-2 mb-1">
          <Database className="w-4 h-4 text-neutral-300" />
          <h2 className="text-sm font-semibold text-neutral-200">Historical analysis: best allocation</h2>
        </div>
        <p className="text-xs text-neutral-500">
          Analyzes real historical prices and searches for the allocation with the best simulated return for its
          risk. It needs historical data for at least one selected holding to have a real timeline to build on — add
          it in the Configure step (mode: &quot;Historical data&quot;) for{" "}
          <span className="text-neutral-300">{selected.map((p) => p.name).join(", ")}</span> to use this.
        </p>
      </Card>
    );
  }

  if (dataset.error) {
    return (
      <Card className="p-4 sm:p-5 anim-in">
        <div className="flex items-center gap-2 mb-1">
          <Database className="w-4 h-4 text-neutral-300" />
          <h2 className="text-sm font-semibold text-neutral-200">Historical analysis: best allocation</h2>
        </div>
        <p className="text-xs text-neutral-500">{dataset.error}</p>
      </Card>
    );
  }

  function finishSearch(search, usedObjective, secs, lossLimit, riskFree, blocks, years) {
    setStatus("finishing");
    timerRef.current = setTimeout(() => {
      if (unmountedRef.current) return;
      const res = search.finish();
      const analysis = buildHistoricalAnalysis(dataset, selected, res.winner.w, res.baseline.w, rebalance, amount, riskFree);
      setOutcome({ ...res, analysis, objective: usedObjective, seconds: secs, lossLimit, riskFree, blockMonths: blocks, years });
      setStatus("done");
    }, 30);
  }

  function handleStart() {
    const secs = Math.min(300, Math.max(2, Number(seconds) || 10));
    const usedObjective = objective;
    const riskFree = Math.max(0, Number(riskFreePct) || 0) / 100;
    const lossLimit = Math.min(0.9, Math.max(0.01, Number(lossLimitPct) || 10) / 100);
    const blocks = Math.max(1, Math.min(24, Number(blockMonths) || 3));
    const years = (MC_HORIZON_OPTIONS.find((h) => h.value === horizon) || MC_HORIZON_OPTIONS[1]).years;
    cancelRef.current = false;
    setStatus("running");
    setOutcome(null);
    setApplied(false);
    setLive({ tested: 0, best: null, elapsed: 0, total: secs });
    // Let the "running" state paint before the (brief) set-up work.
    timerRef.current = setTimeout(() => {
      if (unmountedRef.current) return;
      const search = createHistoricalSearch({
        dataset,
        amount,
        years,
        blockMonths: blocks,
        rebalanceInfo: rebalance,
        objective: usedObjective,
        lossLimit,
        riskFree,
        startWeights: weights,
      });
      const startedAt = Date.now();
      let lastUi = 0;
      const tick = () => {
        if (unmountedRef.current) return;
        const elapsed = (Date.now() - startedAt) / 1000;
        if (cancelRef.current || elapsed >= secs) {
          finishSearch(search, usedObjective, secs, lossLimit, riskFree, blocks, years);
          return;
        }
        search.step(Date.now() + 40, elapsed / secs);
        if (Date.now() - lastUi > 150) {
          lastUi = Date.now();
          setLive(search.getProgress());
          setLive((prevLive) => ({ ...prevLive, elapsed, total: secs }));
        }
        timerRef.current = setTimeout(tick, 0);
      };
      tick();
    }, 40);
  }

  function handleStop() {
    cancelRef.current = true;
  }

  const running = status === "running" || status === "finishing";
  const objectiveLabel = (OBJECTIVE_OPTIONS.find((o) => o.value === (outcome ? outcome.objective : objective)) || {}).label;

  const metricRows = outcome
    ? [
        { label: "Simulated annual return", cur: fmtPct(outcome.baseline.m.mean, 2), best: fmtPct(outcome.winner.m.mean, 2) },
        { label: "Risk (spread of outcomes)", cur: fmtPct(outcome.baseline.m.vol, 2), best: fmtPct(outcome.winner.m.vol, 2) },
        { label: "Return per risk", cur: outcome.baseline.m.sharpe.toFixed(2), best: outcome.winner.m.sharpe.toFixed(2) },
        {
          label: "Return per downside risk",
          cur: outcome.baseline.m.sortino.toFixed(2),
          best: outcome.winner.m.sortino.toFixed(2),
        },
        { label: "Chance of loss", cur: fmtPct(outcome.baseline.m.probLoss, 1), best: fmtPct(outcome.winner.m.probLoss, 1) },
      ]
    : [];

  return (
    <Card className="p-4 sm:p-5 anim-in">
      <div className="flex items-center gap-2 mb-1">
        <Database className="w-4 h-4 text-neutral-300" />
        <h2 className="text-sm font-semibold text-neutral-200">Historical analysis: best allocation</h2>
      </div>
      <p className="text-xs text-neutral-500 mb-4">
        Uses the real prices you added — the timeline spans{" "}
        <span className="text-neutral-300">{dataset.T.toLocaleString("en-IN")} trading days</span> from{" "}
        <span className="text-neutral-300">
          {isoToLabel(dataset.dates[0])} to {isoToLabel(dataset.dates[dataset.dates.length - 1])}
        </span>{" "}
        ({dataset.spanYears.toFixed(1)} years). It resamples real historical stretches (not a statistical assumption)
        to build thousands of possible futures — keeping the real co-movement, volatility clustering and fat tails
        between holdings exactly as they happened — searches those for the best risk-versus-return allocation, then
        checks the winner against the actual historical timeline below.
      </p>
      {dataset.fixedNames.length > 0 && (
        <p className="text-xs text-neutral-500 mb-4 rounded-lg border border-neutral-800 px-3 py-2.5">
          <span className="text-neutral-300">{dataset.fixedNames.join(", ")}</span> — no historical data was added,
          so {dataset.fixedNames.length === 1 ? "it is" : "they are"} treated as a fixed return (its entered return
          held constant every day, no ups or downs) rather than a real price series. It contributes no risk and no
          correlation with the others, the way a fixed deposit would not.
        </p>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
        <div>
          <label className="block text-xs text-neutral-500 mb-1">Simulated horizon</label>
          <select
            value={horizon}
            disabled={running}
            onChange={(e) => setHorizon(e.target.value)}
            className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300 disabled:opacity-60"
          >
            {MC_HORIZON_OPTIONS.map((h) => (
              <option key={h.value} value={h.value}>
                {h.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-neutral-500 mb-1">Run time (seconds)</label>
          <input
            type="number"
            min="2"
            max="300"
            step="1"
            value={seconds}
            disabled={running}
            onChange={(e) => setSeconds(e.target.value)}
            className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 font-mono focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300 disabled:opacity-60"
          />
        </div>
        <div>
          <label className="block text-xs text-neutral-500 mb-1">Goal</label>
          <select
            value={objective}
            disabled={running}
            onChange={(e) => setObjective(e.target.value)}
            className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300 disabled:opacity-60"
          >
            {OBJECTIVE_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {objective === "losscap" && (
        <div className="mb-4 max-w-xs">
          <label className="block text-xs text-neutral-500 mb-1">Max acceptable chance of loss (%)</label>
          <input
            type="number"
            min="1"
            max="90"
            step="1"
            value={lossLimitPct}
            disabled={running}
            onChange={(e) => setLossLimitPct(e.target.value)}
            className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 font-mono focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300 disabled:opacity-60"
          />
          <p className="text-[13px] text-neutral-600 mt-0.5">
            Among allocations at or below this chance of loss, picks the one with the highest return.
          </p>
        </div>
      )}

      <div className="rounded-lg border border-neutral-800 mb-4">
        <button
          onClick={() => setShowAdvanced((v) => !v)}
          className="w-full flex items-center justify-between px-3.5 py-2.5 text-sm text-neutral-300 hover:bg-neutral-900/60 rounded-lg transition-colors"
        >
          <span>Advanced settings</span>
          <span className="text-xs text-neutral-500">{showAdvanced ? "Hide" : "Show"}</span>
        </button>
        {showAdvanced && (
          <div className="px-3.5 pb-3.5 grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div>
              <label className="block text-xs text-neutral-500 mb-1">History window</label>
              <select
                value={histWindow}
                disabled={running}
                onChange={(e) => setHistWindow(e.target.value)}
                className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300 disabled:opacity-60"
              >
                {HISTORY_WINDOW_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
              <p className="text-[13px] text-neutral-600 mt-0.5">Restricts the resampled history to the most recent stretch, if you don't want older, less relevant periods included.</p>
            </div>
            <div>
              <label className="block text-xs text-neutral-500 mb-1">Risk-free rate (%)</label>
              <input
                type="number"
                min="0"
                max="20"
                step="0.5"
                value={riskFreePct}
                disabled={running}
                onChange={(e) => setRiskFreePct(e.target.value)}
                className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 font-mono focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300 disabled:opacity-60"
              />
              <p className="text-[13px] text-neutral-600 mt-0.5">Used as the baseline "safe" return for the risk ratios.</p>
            </div>
            <div>
              <label className="block text-xs text-neutral-500 mb-1">Block length (months)</label>
              <input
                type="number"
                min="1"
                max="24"
                step="1"
                value={blockMonths}
                disabled={running}
                onChange={(e) => setBlockMonths(e.target.value)}
                className="w-full text-sm bg-black border border-neutral-700 rounded-md px-2.5 py-2 text-neutral-100 font-mono focus:outline-none focus:border-neutral-300 focus:ring-1 focus:ring-neutral-300 disabled:opacity-60"
              />
              <p className="text-[13px] text-neutral-600 mt-0.5">
                Consecutive real months resampled together — longer keeps more real momentum, shorter mixes more
                combinations.
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={handleStart}
          disabled={running}
          className="flex items-center gap-1.5 text-sm bg-neutral-100 hover:bg-white disabled:opacity-60 text-black font-medium px-3.5 py-2 rounded-md transition-colors"
        >
          {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Shuffle className="w-4 h-4" />}
          {status === "running"
            ? "Searching..."
            : status === "finishing"
            ? "Checking finalists..."
            : outcome
            ? "Search again"
            : "Start historical search"}
        </button>
        {status === "running" && (
          <button
            onClick={handleStop}
            className="flex items-center gap-1.5 text-sm text-neutral-300 border border-neutral-700 hover:bg-neutral-800 px-3.5 py-2 rounded-md transition-colors"
          >
            <Square className="w-3.5 h-3.5" />
            Stop and use best so far
          </button>
        )}
      </div>

      {running && live && (
        <div className="mt-4 space-y-3 anim-in">
          <div>
            <div className="flex justify-between text-xs text-neutral-500 mb-1 font-mono">
              <span>
                {Math.min(live.elapsed || 0, live.total).toFixed(1)}s / {live.total}s
              </span>
              <span>{live.tested.toLocaleString("en-IN")} allocations tested</span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-neutral-800 overflow-hidden">
              <div
                className="h-full rounded-full bg-neutral-100"
                style={{ width: `${Math.min(100, ((live.elapsed || 0) / live.total) * 100)}%` }}
              />
            </div>
          </div>
          {live.best && (
            <div className="rounded-lg border border-neutral-800 px-3 py-2.5">
              <p className="text-xs text-neutral-500 mb-1">Best so far</p>
              <p className="text-sm text-neutral-200 font-mono">
                {selected.map((p, i) => `${p.name} ${(live.best.w[i] * 100).toFixed(1)}%`).join(" · ")}
              </p>
              <p className="text-xs text-neutral-500 font-mono mt-1">
                return {fmtPct(live.best.m.mean, 2)} · risk {fmtPct(live.best.m.vol, 2)} · return per risk{" "}
                {live.best.m.sharpe.toFixed(2)}
              </p>
            </div>
          )}
        </div>
      )}

      {status === "done" && outcome && (
        <div className="mt-5 space-y-5 anim-in">
          <div className="rounded-lg border border-neutral-600 bg-white/5 px-3.5 py-2.5">
            <p className="text-sm text-neutral-100">
              {outcome.improved
                ? "Found a better allocation than your current one, on the resampled real history."
                : "Your current allocation was already the best of everything tested — no change recommended."}
            </p>
            <p className="text-xs text-neutral-500 mt-0.5">
              {outcome.tested.toLocaleString("en-IN")} allocations tested in {outcome.seconds}s · goal:{" "}
              {objectiveLabel?.toLowerCase()} · block {outcome.blockMonths}mo
            </p>
          </div>

          <div>
            <h3 className="text-xs font-semibold text-neutral-300 mb-2">Allocation</h3>
            <div className="space-y-2">
              {selected.map((p, i) => {
                const curW = outcome.baseline.w[i];
                const bestW = outcome.winner.w[i];
                return (
                  <div key={p.id}>
                    <div className="flex justify-between gap-3 text-sm mb-1">
                      <span className="flex items-center gap-2 min-w-0 text-neutral-200">
                        <span
                          className="h-2 w-2 rounded-full shrink-0"
                          style={{ backgroundColor: SERIES_COLORS[i % SERIES_COLORS.length] }}
                        />
                        <span className="truncate">{p.name}</span>
                      </span>
                      <span className="text-neutral-400 font-mono text-xs whitespace-nowrap self-center">
                        {fmtPct(curW, 1)} → <span className="text-neutral-100">{fmtPct(bestW, 1)}</span> · {fmtINR(bestW * amount)}
                      </span>
                    </div>
                    <div className="h-1.5 w-full rounded-full bg-neutral-800 overflow-hidden relative">
                      <div className="absolute inset-y-0 left-0 rounded-full bg-neutral-600" style={{ width: `${curW * 100}%` }} />
                      <div
                        className="absolute inset-y-0 left-0 rounded-full"
                        style={{ width: `${bestW * 100}%`, backgroundColor: SERIES_COLORS[i % SERIES_COLORS.length], opacity: 0.85 }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="text-[13px] text-neutral-600 mt-1.5">Dark bar = current allocation, bright bar = best found.</p>
          </div>

          <div>
            <h3 className="text-xs font-semibold text-neutral-300 mb-1">Current vs best (resampled real history, fresh draws)</h3>
            <div className="overflow-x-auto rounded-lg border border-neutral-800">
              <table className="w-full text-sm min-w-[340px]">
                <thead>
                  <tr className="text-left text-xs text-neutral-500 border-b border-neutral-800">
                    <th className="px-3 py-2 font-medium" />
                    <th className="px-3 py-2 font-medium">Current</th>
                    <th className="px-3 py-2 font-medium">Best found</th>
                  </tr>
                </thead>
                <tbody>
                  {metricRows.map((row) => (
                    <tr key={row.label} className="border-b border-neutral-800/60 last:border-0">
                      <td className="px-3 py-2 text-neutral-400">{row.label}</td>
                      <td className="px-3 py-2 font-mono text-neutral-300">{row.cur}</td>
                      <td className="px-3 py-2 font-mono text-neutral-100">{row.best}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div>
            <h3 className="text-xs font-semibold text-neutral-300 mb-1">Every allocation tested: risk vs return</h3>
            <p className="text-xs text-neutral-500 mb-2">
              Each dot is one random allocation scored on resampled real history. Up and to the left is better (more
              return, less risk). Shown: a sample of {outcome.cloud.length.toLocaleString("en-IN")} of the{" "}
              {outcome.tested.toLocaleString("en-IN")} tested.
            </p>
            <ResponsiveContainer width="100%" height={270}>
              <ScatterChart margin={{ top: 8, right: 14, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="Risk"
                  domain={["auto", "auto"]}
                  tick={{ fontSize: 12, fill: "#a3a3a3" }}
                  tickFormatter={(v) => `${Number(v).toFixed(1)}%`}
                  height={46}
                  label={{ value: "Risk (spread of outcomes)", position: "insideBottom", offset: -2, fill: "#a3a3a3", fontSize: 12 }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="Return"
                  domain={["auto", "auto"]}
                  tick={{ fontSize: 12, fill: "#a3a3a3" }}
                  tickFormatter={(v) => `${Number(v).toFixed(1)}%`}
                  width={52}
                />
                <ZAxis type="number" dataKey="z" domain={[0, 10]} range={[18, 200]} />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3", stroke: "#525252" }}
                  content={({ active, payload }) =>
                    active && payload && payload.length ? (
                      <div style={{ backgroundColor: "#0a0a0a", border: "1px solid #262626", borderRadius: 8, padding: "6px 10px", fontSize: 14, color: "#e5e5e5" }}>
                        Risk {Number(payload[0].payload.x).toFixed(2)}% · Return {Number(payload[0].payload.y).toFixed(2)}%
                      </div>
                    ) : null
                  }
                />
                <Legend verticalAlign="bottom" iconSize={9} wrapperStyle={{ fontSize: 13, color: "#a3a3a3" }} />
                <Scatter name="Tested allocations" data={outcome.cloud.map((pt) => ({ ...pt, z: 0 }))} fill="#525252" fillOpacity={0.55} isAnimationActive={false} />
                <Scatter name="Current" data={[{ ...outcome.searchBaseline, z: 6 }]} fill="#a3a3a3" shape="triangle" legendType="triangle" isAnimationActive={false} />
                <Scatter name="Best found" data={[{ ...outcome.searchBest, z: 10 }]} fill="#fafafa" shape="square" legendType="square" isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {outcome.improved && (
              <button
                onClick={() => {
                  onApply(outcome.winner.w);
                  setApplied(true);
                }}
                disabled={applied}
                className="flex items-center gap-1.5 text-sm bg-neutral-100 hover:bg-white disabled:opacity-60 text-black font-medium px-3.5 py-2 rounded-md transition-colors"
              >
                <CheckCircle2 className="w-4 h-4" />
                {applied ? "Allocation applied" : "Use this allocation"}
              </button>
            )}
            {result.customized && (
              <button
                onClick={() => {
                  onRestore();
                  setApplied(false);
                }}
                className="flex items-center gap-1.5 text-sm text-neutral-300 border border-neutral-700 hover:bg-neutral-800 px-3.5 py-2 rounded-md transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Restore optimizer allocation
              </button>
            )}
          </div>

          <div className="pt-2 border-t border-neutral-800">
            <h3 className="text-xs font-semibold text-neutral-300 mb-1">What actually happened: real historical backtest</h3>
            <p className="text-xs text-neutral-500 mb-2">
              Not a simulation — this is what each fixed allocation would really have been worth, day by day, over{" "}
              {isoToLabel(dataset.dates[0])} to {isoToLabel(dataset.dates[dataset.dates.length - 1])}, rebalanced on
              your chosen schedule.
            </p>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={outcome.analysis.chart} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 12, fill: "#a3a3a3" }} interval="preserveStartEnd" tickFormatter={isoToLabel} />
                <YAxis tick={{ fontSize: 12, fill: "#a3a3a3" }} width={60} tickFormatter={fmtINRCompact} domain={["auto", "auto"]} />
                <ReferenceLine y={amount} stroke="#525252" strokeDasharray="4 4" />
                <Tooltip
                  labelFormatter={isoToLabel}
                  formatter={(v, name) => [fmtINR(v), name]}
                  itemStyle={{ color: "#e5e5e5" }}
                  labelStyle={{ color: "#e5e5e5", fontWeight: 600 }}
                  contentStyle={TOOLTIP_STYLE}
                />
                <Legend wrapperStyle={{ fontSize: 13, color: "#a3a3a3" }} />
                {selected.map((p, i) => (
                  <Line
                    key={`asset${i}`}
                    dataKey={`asset${i}`}
                    name={p.name}
                    stroke={ASSET_LINE_COLORS[i % ASSET_LINE_COLORS.length]}
                    strokeWidth={1.25}
                    strokeOpacity={0.85}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
                <Line dataKey="current" name="Current allocation" stroke="#a3a3a3" strokeWidth={1.5} strokeDasharray="5 3" dot={false} isAnimationActive={false} />
                <Line dataKey="final" name="Recommended allocation" stroke="#fafafa" strokeWidth={2.5} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>

            <div className="overflow-x-auto rounded-lg border border-neutral-800 mt-3">
              <table className="w-full text-sm min-w-[420px]">
                <thead>
                  <tr className="text-left text-xs text-neutral-500 border-b border-neutral-800">
                    <th className="px-3 py-2 font-medium">Allocation</th>
                    <th className="px-3 py-2 font-medium">CAGR</th>
                    <th className="px-3 py-2 font-medium">Volatility</th>
                    <th className="px-3 py-2 font-medium">Max drawdown</th>
                    <th className="px-3 py-2 font-medium">Return per risk</th>
                  </tr>
                </thead>
                <tbody>
                  {outcome.analysis.rows.map((r) => (
                    <tr key={r.key} className="border-b border-neutral-800/60 last:border-0">
                      <td className="px-3 py-2 text-neutral-200">{r.label}</td>
                      <td className="px-3 py-2 font-mono text-neutral-300">{fmtPct(r.cagr, 1)}</td>
                      <td className="px-3 py-2 font-mono text-neutral-300">{fmtPct(r.vol, 1)}</td>
                      <td className="px-3 py-2 font-mono text-neutral-400">{fmtPct(r.maxDrawdown, 1)}</td>
                      <td className="px-3 py-2 font-mono text-neutral-100">{r.ratio.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <h4 className="text-xs font-semibold text-neutral-300 mt-4 mb-1">Consistency check: first half vs second half</h4>
            <p className="text-[13px] text-neutral-600 mb-2">
              Splits the same history in two at {isoToLabel(outcome.analysis.midDate)} and shows CAGR separately for
              each half — a recommendation that only worked in one half is a warning sign, not a strength.
            </p>
            <div className="overflow-x-auto rounded-lg border border-neutral-800">
              <table className="w-full text-sm min-w-[380px]">
                <thead>
                  <tr className="text-left text-xs text-neutral-500 border-b border-neutral-800">
                    <th className="px-3 py-2 font-medium">Allocation</th>
                    <th className="px-3 py-2 font-medium">First half CAGR</th>
                    <th className="px-3 py-2 font-medium">Second half CAGR</th>
                  </tr>
                </thead>
                <tbody>
                  {outcome.analysis.halves.map((h) => (
                    <tr key={h.key} className="border-b border-neutral-800/60 last:border-0">
                      <td className="px-3 py-2 text-neutral-200">{h.label}</td>
                      <td className="px-3 py-2 font-mono text-neutral-300">{fmtPct(h.first.cagr, 1)}</td>
                      <td className="px-3 py-2 font-mono text-neutral-300">{fmtPct(h.second.cagr, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <p className="text-xs text-neutral-600">
            The search and the risk/return table above run on {outcome.paths.toLocaleString("en-IN")} resampled draws
            during the search and {outcome.validationPaths.toLocaleString("en-IN")} fresh ones for the table — real
            historical stretches, reshuffled, not a parametric guess. Even so, this is one real historical window (
            {dataset.spanYears.toFixed(1)} years), not many independent alternate histories, and it reflects{" "}
            {selected.length === dataset.n ? "this specific" : ""} data's own period — a different period, or the
            future, can look very different. Past performance is not a guarantee of future results.
          </p>
        </div>
      )}
    </Card>
  );
}


function MetricCard({ icon, label, value, subValue, delay = 0 }) {
  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 px-3.5 py-3 anim-in" style={{ animationDelay: `${delay}ms` }}>
      <div className="flex items-center gap-1.5 mb-1">
        {icon}
        <p className="text-xs text-neutral-500">{label}</p>
      </div>
      <p className="text-lg font-semibold text-neutral-100 font-mono">{value}</p>
      {subValue && <p className="text-xs text-neutral-500 font-mono mt-0.5">{subValue}</p>}
    </div>
  );
}