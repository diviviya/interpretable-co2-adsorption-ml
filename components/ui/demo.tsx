"use client";

import ClothingRailHero from "./product-rail-gallery";
import type { RailGarment } from "./product-rail-gallery";

export const ARCHITECTURE_RAIL_ITEMS: RailGarment[] = [
  {
    id: "frontend-arch",
    name: "Frontend Client Architecture",
    kind: "jacket",
    color: "#0284c7",
    trim: "#38bdf8",
    tag: "UI / UX",
    description: "Responsive, zero-build HTML5/Vanilla JS interface with real-time SHAP waterfall charts, CSV drag-and-drop batch evaluation, and physical domain input guardrails.",
    details: [
      "Vanilla JS (ES6+)",
      "Zero Node Build Overhead",
      "Real-Time SHAP Visualizer",
      "CSV Batch Engine (<500ms)",
      "Dynamic Min/Max Validation"
    ]
  },
  {
    id: "backend-arch",
    name: "FastAPI Async REST Core",
    kind: "coat",
    color: "#059669",
    trim: "#10b981",
    tag: "API Core",
    description: "High-throughput asynchronous Python FastAPI service enforcing strict Pydantic physical constraints, dual session/Admin-key authentication, and async worker retraining.",
    details: [
      "FastAPI + Starlette",
      "Async Non-Blocking Endpoints",
      "Pydantic Physical Schemas",
      "Stateless Guest + Admin Auth",
      "Sub-20ms Single Inference"
    ]
  },
  {
    id: "ml-etfm-arch",
    name: "ETFM Machine Learning Engine",
    kind: "hoodie",
    color: "#d97706",
    trim: "#f59e0b",
    tag: "ML Ensemble",
    description: "Entropy-TOPSIS Fusion Model (ETFM) combining 5 Bayesian-optimized tree algorithms (CatBoost, Extra Trees, Random Forest, XGBoost, LightGBM) achieving SOTA R² = 0.9903.",
    details: [
      "Optuna Bayesian SMBO (150 trials)",
      "Objective Entropy Weighting",
      "TOPSIS Closeness Ranking",
      "Test R² = 0.9903 · RMSE = 0.6008",
      "SHAP & ALE Global Explainability"
    ]
  },
  {
    id: "zero-sql-arch",
    name: "Strict Zero-SQL Datastore",
    kind: "shirt",
    color: "#7c3aed",
    trim: "#a78bfa",
    tag: "Storage",
    description: "Eliminates heavy RDBMS/SQL databases in favor of thread-safe, ACID-compliant JSON persistent stores using atomic staging (.tmp -> replace) to guarantee zero corruption.",
    details: [
      "Strict Zero-SQL Footprint",
      "Thread-Safe File Locks",
      "Atomic Staging (.tmp Swap)",
      "Zero-Downtime Rollback",
      "5 Stores: models, logs, jobs, data"
    ]
  },
  {
    id: "preprocessing-arch",
    name: "Data Preprocessing & Validation",
    kind: "knit",
    color: "#0d9488",
    trim: "#2dd4bf",
    tag: "Pipeline",
    description: "Rigorous pipeline curating 3,594 literature data points, Isolation Forest 5% outlier detection, Pearson correlation (|ρ| < 0.8), and GroupKFold (seed 42) zero-leakage splits.",
    details: [
      "3,594 Literature Samples",
      "Isolation Forest (5% Contam.)",
      "Pearson Check (|ρ| < 0.8)",
      "StandardScaler (Train-fitted)",
      "GroupKFold (Seed 42, 5 Folds)"
    ]
  },
  {
    id: "crystallography-arch",
    name: "Crystallography Modeling Basis",
    kind: "tee",
    color: "#475569",
    trim: "#94a3b8",
    tag: "Physics Basis",
    description: "Grounds porous carbon adsorption in turbostratic semi-crystalline physics: Bragg's Law (2d sinθ = nλ), d002 interlayer spacing, slit-pore geometries, and N/O/S defect Lewis-base centers.",
    details: [
      "Bragg's Law (2d sinθ = nλ)",
      "Turbostratic Graphene Sheets",
      "d002 Spacing & Slit Micropores",
      "Defect & Heteroatom Quadrupole",
      "Kinetic Window: 0.5 - 1.2 nm Pores"
    ]
  }
];

export default function ArchitectureRailHeroDemo({ autoplay = true }: { autoplay?: boolean }) {
  return (
    <div data-rail-demo className="flex min-h-svh w-full items-center justify-center bg-background text-foreground">
      <ClothingRailHero
        items={ARCHITECTURE_RAIL_ITEMS}
        hanger="wood"
        eyebrow="System Architecture · Checkout Each Layer"
        heading="Full-Stack CO2 Adsorption Platform Architecture"
        caption="Slide along the rail to inspect components, or click any architecture to checkout its full technical breakdown."
        autoplay={autoplay}
        action={
          <div className="flex gap-3">
            <a
              href="CO2_Adsorption_ML_Presentation.pptx"
              download
              className="rounded-full bg-foreground px-5 py-2.5 text-sm font-medium text-background shadow-sm outline-none transition-[transform,box-shadow] hover:-translate-y-px hover:shadow-md focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background motion-reduce:transition-none"
            >
              Download PPTX Deck
            </a>
          </div>
        }
      />
    </div>
  );
}
