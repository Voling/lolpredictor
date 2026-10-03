import type { Band, Difference, Posterior } from "./api";

type Tone = "left" | "right";

function share(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function linear(value: number): number {
  return Math.min(100, Math.max(0, value * 100));
}

function logged(value: number): number {
  return ((Math.log2(Math.min(4, Math.max(0.25, value))) + 2) / 4) * 100;
}

function capital(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function Row({ label, marks, prior, place, note }: { label: string; marks: (Band & { tone: Tone })[]; prior?: number; place: (value: number) => number; note: string }) {
  return (
    <div className="post-row">
      <span className="post-label">{label}</span>
      <span className="post-track">
        {prior != null && <span className="post-prior" style={{ left: `${place(prior)}%` }} />}
        {marks.map((mark) => (
          <span key={mark.tone}>
            <span className={`post-band ${mark.tone}`} style={{ left: `${place(mark.low)}%`, width: `${Math.max(0.5, place(mark.high) - place(mark.low))}%` }} />
            <span className={`post-dot ${mark.tone}`} style={{ left: `${place(mark.mean)}%` }} />
          </span>
        ))}
      </span>
      <span className="post-note">{note}</span>
    </div>
  );
}

export function SituationChart({ posterior, tone }: { posterior: Posterior; tone: Tone }) {
  if (posterior.kind === "ratio" && posterior.ratio) {
    return (
      <div className="posterior">
        <p className="post-title">{posterior.takeaway ?? capital(posterior.words)}</p>
        <Row label="rate against expected" marks={[{ ...posterior.ratio, tone }]} prior={1} place={logged} note={`${posterior.observed} seen, ${posterior.expected} expected`} />
        <p className="post-scale"><span>¼×</span><span>1×</span><span>4×</span></p>
      </div>
    );
  }
  const chances = posterior.situation.startsWith("prio") ? `${Math.round(posterior.n)} games` : `${Math.round(posterior.n)} chances`;
  return (
    <div className="posterior">
      <p className="post-title">{posterior.takeaway ?? capital(posterior.words)}</p>
      <p className="post-count">{capital(posterior.words)}, {chances}{posterior.own != null ? `: ${share(posterior.own)} their own play, the rest the typical player` : ""}</p>
      {(posterior.outcomes ?? []).map((outcome) => (
        <Row key={outcome.name} label={outcome.words} marks={[{ mean: outcome.mean, low: outcome.low, high: outcome.high, tone }]} prior={outcome.prior} place={linear} note={share(outcome.mean)} />
      ))}
    </div>
  );
}

export function DifferenceChart({ difference }: { difference: Difference }) {
  return (
    <div className="posterior">
      <p className="post-title">{difference.takeaway ?? capital(difference.words)}</p>
      {difference.outcomes.map((outcome) => (
        <Row
          key={outcome.name}
          label={outcome.words}
          marks={[{ ...outcome.left, tone: "left" }, { ...outcome.right, tone: "right" }]}
          place={linear}
          note={`${share(outcome.left.mean)} vs ${share(outcome.right.mean)}`}
        />
      ))}
    </div>
  );
}
