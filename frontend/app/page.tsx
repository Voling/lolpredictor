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
  const sigma = model.synergy_gain_sigma;
  const informative = typeof sigma === "number" && sigma >= 2;

  return (
    <main>
      <h1>lolpredictor</h1>
      <p className="sub">Player compatibility from the first fifteen minutes.</p>

      {error && <div className="gate"><strong>API unreachable</strong>{error}</div>}

      {status && !informative && (
        <div className="gate">
          <strong>Scores are withheld</strong>
          synergy_gain_sigma is {sigma?.toFixed(2)}, below the threshold of 2.00, so the model
          reports no usable pair signal and pair scores return null.
        </div>
      )}

      {status && (
        <table>
          <thead>
            <tr><th>metric</th><th className="num">value</th></tr>
          </thead>
          <tbody>
            <tr><td>players profiled</td><td className="num">{status.players.toLocaleString()}</td></tr>
            <tr><td>known pairs</td><td className="num">{status.known_pairs.toLocaleString()}</td></tr>
            <tr><td>cache</td><td className="num">{status.cache ? "connected" : "off"}</td></tr>
            {["matches", "baseline_auc", "auc", "synergy_gain", "synergy_gain_sigma", "log_loss", "brier"].map(
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
