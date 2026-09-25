"use client";

import Link from "next/link";
import { getStatus } from "@/lib/api";
import { useRemote } from "@/lib/remote";

export default function Home() {
  const { data: status, error } = useRemote(getStatus, "status");

  const model = (status?.model ?? {}) as Record<string, number>;
  const gain = model.advantage_gain;
  const nullsAbove = model.advantage_nulls_above;
  const informative = status?.informative ?? false;

  return (
    <main>
      <h1>lolpredictor</h1>
      <p className="sub">Your score with each friend when you duo.</p>
      <p><Link href="/friends/">Rank your friends</Link> · <Link href="/pair/">Read a pair</Link></p>

      {error && <div className="gate"><strong>API unreachable</strong>{error}</div>}

      {status && !informative && (
        <div className="gate">
          <strong>Pair fit shown for inspection only</strong>
          The pair block adds {gain?.toFixed(6)} R² on the early advantage and {nullsAbove ?? "?"} of 20
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
