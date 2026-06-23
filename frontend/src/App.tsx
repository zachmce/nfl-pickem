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
    <main style={{ fontFamily: "system-ui", maxWidth: 640, margin: "3rem auto" }}>
      <h1>🏈 NFL Pick'em</h1>
      <p>
        Click the button to enqueue a Celery task. The worker writes a row to
        Postgres; the list below reads it back through FastAPI.
      </p>
      <button onClick={triggerPing} disabled={loading}>
        {loading ? "Enqueuing…" : "Enqueue ping task"}
      </button>{" "}
      <button onClick={loadRuns}>Refresh</button>

      <h2>Task runs</h2>
      <ul>
        {runs.map((r) => (
          <li key={r.id}>
            #{r.id} — <strong>{r.message}</strong> @ {r.created_at}
          </li>
        ))}
      </ul>
    </main>
  );
}
