"use client";

import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";

type Cert = {
  cert_id: string;
  title: string;
  issuer: string;
  hours: number;
  cost: number;
  relevance: number;
  cf_score: number;
  content_score: number;
  reason: string;
  rank: number;
};

type RecommendResponse = {
  learner_id: string | null;
  used_collaborative_tower: boolean;
  items: Cert[];
};

const DEMO_SKILLS = [
  "Python", "SQL", "Pandas", "Airflow", "Stakeholder Management", "Communication",
];

export default function Home() {
  const [skills, setSkills] = useState<string>(DEMO_SKILLS.join(", "));
  const [learnerId, setLearnerId] = useState<string>("L-00042");
  const [resp, setResp] = useState<RecommendResponse | null>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    try {
      const res = await fetch(`${API}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          learner_id: learnerId || null,
          learner_skills: skills.split(",").map((s) => s.trim()).filter(Boolean),
          k: 10,
          beta: 0.6,
        }),
      });
      setResp(await res.json());
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen p-8 max-w-5xl mx-auto">
      <h1 className="text-3xl font-bold">Micro-Certification Recommender</h1>
      <p className="opacity-70 mb-6">
        Two towers: what people-like-you completed, and what overlaps with your skills.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
        <label className="block">
          <div className="text-xs uppercase opacity-60 mb-1">Learner skills (comma-separated)</div>
          <textarea
            value={skills}
            onChange={(e) => setSkills(e.target.value)}
            className="w-full rounded-xl border p-2 text-sm h-24"
          />
        </label>
        <label className="block">
          <div className="text-xs uppercase opacity-60 mb-1">Learner ID (optional, warm-start)</div>
          <input
            value={learnerId}
            onChange={(e) => setLearnerId(e.target.value)}
            className="w-full rounded-xl border p-2 text-sm"
            placeholder="L-00042"
          />
        </label>
      </div>

      <button
        onClick={run}
        disabled={loading}
        className="rounded-xl px-4 py-2 bg-black text-white disabled:opacity-50"
      >
        {loading ? "Recommending..." : "Recommend top-10"}
      </button>

      {resp && (
        <>
          <p className="mt-4 text-xs opacity-60">
            Collaborative tower used: <strong>{String(resp.used_collaborative_tower)}</strong>
          </p>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
            {resp.items.map((c) => (
              <div key={c.cert_id} className="rounded-2xl border p-4">
                <div className="text-xs uppercase opacity-60">#{c.rank} - {c.issuer}</div>
                <div className="font-semibold mt-1">{c.title}</div>
                <div className="text-xs opacity-70 mt-1">
                  {c.hours.toFixed(0)} hrs - ${c.cost.toFixed(0)}
                </div>
                <div className="mt-2">
                  <div className="text-xs opacity-60">Relevance</div>
                  <div className="h-2 rounded-full bg-zinc-200">
                    <div
                      className="h-2 rounded-full bg-black"
                      style={{ width: `${Math.max(0, Math.min(1, c.relevance)) * 100}%` }}
                    />
                  </div>
                </div>
                <div className="mt-2 text-xs italic opacity-70">{c.reason}</div>
              </div>
            ))}
          </div>
        </>
      )}
    </main>
  );
}
