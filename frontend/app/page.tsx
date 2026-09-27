"use client";

import Link from "next/link";
import { getStatus } from "@/lib/api";
import { useRemote } from "@/lib/remote";
import { Verdict } from "@/lib/verdict";

const BANDS = [
  { score: 65, range: "60 and up" },
  { score: 55, range: "50 to 59" },
  { score: 45, range: "40 to 49" },
  { score: 35, range: "below 40" },
];

export default function Home() {
  const { data: status, error } = useRemote(getStatus, "status");

  return (
    <main>
      <h1>lolpredictor</h1>
      <p className="sub">Find out which friends you play best with in ranked.</p>

      <h3>How to use it</h3>
      <p><Link href="/friends/">Rank your friends</Link>. Enter your Riot ID and your friends&apos; Riot IDs. Each friend gets a score.</p>
      <p><Link href="/pair/">Check one duo</Link>. Enter your Riot ID and one friend&apos;s Riot ID to get a score for the two of you.</p>

      <h3>What the score means</h3>
      <p>50 is an average duo for your two positions. Higher is better.</p>
      <ul className="legend">
        {BANDS.map((band) => (
          <li key={band.range}><Verdict score={band.score} /> {band.range}</li>
        ))}
      </ul>

      <h3>How it works</h3>
      <p>
        The score comes from your past ranked games. It looks at how each of you plays and how you do on the champions you pick.
        If you two have played together, those games count too.
      </p>
      <p>
        From that it predicts how far ahead your two positions will be in gold at 20 minutes. Teams that are ahead at 20 minutes win
        more often.
      </p>
      <p className="sub">A score is a prediction, not a promise. Any one game can go either way.</p>

      {error && <div className="gate"><strong>The server isn&apos;t reachable right now.</strong>{error}</div>}
      {status?.run && <p className="footer">Model {status.run.id}</p>}
    </main>
  );
}
