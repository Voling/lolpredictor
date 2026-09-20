import Link from "next/link";
import { getStatus } from "@/lib/api";

export default async function Home() {
  let status: Awaited<ReturnType<typeof getStatus>> | null = null;
  let error: string | null = null;
  try {
    status = await getStatus();
  } catch (exception) {
    error = String(exception);
  }

  const model = (status?.model ?? {}) as Record<string, number>;
  const gain = model.advantage_gain;
  const nullsAbove = model.advantage_nulls_above;
  const informative = status?.informative ?? false;

  return (
    <main>
      <h1>lolpredictor</h1>
      <p className="sub">Player compatibility from the first fifteen minutes.</p>
      <p><Link href="/pair">Read a pair</Link></p>

      {error && <div className="gate"><strong>API unreachable</strong>{error}</div>}

      {status && !informative && (
        <div className="gate">
          <strong>Pair fit shown for inspection only</strong>
          The pair block adds {gain?.toFixed(6)} R² on advantage at 15 and {nullsAbove ?? "?"} of 20
          permutation nulls reached it, so the model reports no usable pair signal at the team level.
        </div>
      )}

      {status && (
        <table>
          <thead>
            <tr><th>metric</th><th className="num">value</th></tr>
          </thead>
          <tbody>
            <tr><td>served run</td><td className="num">{status.run ? `${status.run.id} at ${status.run.git?.sha ?? "?"}${status.run.git?.dirty ? " with local changes" : ""}` : "working artifacts, no run promoted"}</td></tr>
            <tr><td>players profiled</td><td className="num">{status.players.toLocaleString()}</td></tr>
            <tr><td>known pairs</td><td className="num">{status.known_pairs.toLocaleString()}</td></tr>
            <tr><td>cache</td><td className="num">{status.cache ? "connected" : "off"}</td></tr>
            {[
              "matches",
              "advantage_matches",
              "advantage_base_r2",
              "advantage_full_r2",
              "advantage_gain",
              "advantage_nulls_above",
              "advantage_gold_sd",
            ].map(
              (key) =>
                model[key] !== undefined && (
                  <tr key={key}>
                    <td>{key}</td>
                    <td className="num">{model[key].toLocaleString()}</td>
                  </tr>
                ),
            )}
          </tbody>
        </table>
      )}
    </main>
  );
}
