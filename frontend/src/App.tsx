import { useEffect, useState } from "react";

interface TaskRun {
  id: number;
  message: string;
  created_at: string;
}

export default function App() {
  const [runs, setRuns] = useState<TaskRun[]>([]);
  const [loading, setLoading] = useState(false);

  async function loadRuns() {
    const res = await fetch("/api/task-runs");
    setRuns(await res.json());
  }

  async function triggerPing() {
    setLoading(true);
    try {
      await fetch("/api/ping?message=hello-from-spa", { method: "POST" });
      // Give the worker a moment, then refresh.
      setTimeout(loadRuns, 750);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadRuns();
  }, []);

  return (
    <main className="mx-auto my-12 max-w-2xl px-4 font-sans">
      <h1 className="mb-4 text-3xl font-bold">🏈 NFL Pick'em</h1>
      <p className="mb-6 text-gray-700">
        Click the button to enqueue a Celery task. The worker writes a row to
        Postgres; the list below reads it back through FastAPI.
      </p>
      <button
        onClick={triggerPing}
        disabled={loading}
        className="rounded bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
      >
        {loading ? "Enqueuing…" : "Enqueue ping task"}
      </button>{" "}
      <button
        onClick={loadRuns}
        className="rounded border border-gray-300 bg-white px-4 py-2 font-medium text-gray-700 hover:bg-gray-50"
      >
        Refresh
      </button>

      <h2 className="mt-8 mb-2 text-xl font-semibold">Task runs</h2>
      <ul className="mt-2 space-y-1">
        {runs.map((r) => (
          <li key={r.id} className="text-sm text-gray-800">
            #{r.id} — <strong>{r.message}</strong> @ {r.created_at}
          </li>
        ))}
      </ul>
    </main>
  );
}
