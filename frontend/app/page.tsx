"use client";

import Link from "next/link";
import { getRecent, getStatus } from "@/lib/api";
import { useAccount } from "@/lib/account";
import { authEnabled } from "@/lib/auth";
import { DemoCarousel } from "@/lib/demo";
import { RecentDuos } from "@/lib/recent";
import { useRemote } from "@/lib/remote";
import { Verdict } from "@/lib/verdict";

const BANDS = [
  { score: 65, range: "60 and up" },
  { score: 55, range: "50 to 59" },
  { score: 45, range: "40 to 49" },
  { score: 35, range: "below 40" },
];

export default function Home() {
  const account = useAccount();
  const canAsk = !authEnabled || account.signedIn;
  const { data: status, error } = useRemote(canAsk ? getStatus : null, `status|${canAsk}`);
  const { data: recent } = useRemote(authEnabled && account.signedIn ? getRecent : null, `recent|${account.signedIn}`);
  const daily = account.me?.daily;

  return (
    <main>
      <h1>lolpredictor</h1>
      <p className="sub">Find out which friends you play best with in ranked.</p>
      <p className="nav"><Link href="/friends/">Rank friends</Link><Link href="/pair/">Check a duo</Link>{authEnabled && <Link href="/account/">Account</Link>}</p>

      <h3>How to use it</h3>
      {authEnabled && <p>Sign in and link your Riot account first. Each account gets {daily ? `${daily} duo checks a day` : "a daily allowance of duo checks"}.</p>}
      <p><Link href="/friends/">Rank your friends</Link>. Enter your Riot ID and your friends&apos; Riot IDs. Each friend gets a score.</p>
      <p><Link href="/pair/">Check one duo</Link>. Enter your Riot ID and one friend&apos;s Riot ID to get a score for the two of you.</p>

      <h3>What the score means</h3>
      <p>50 is an average duo for your two positions. Higher is better.</p>
      <ul className="legend">
        {BANDS.map((band) => (
          <li key={band.range}><Verdict score={band.score} /> {band.range}</li>
        ))}
      </ul>

      {recent && recent.duos.length > 0 && (
        <>
          <h3>Recent predictions</h3>
          <RecentDuos duos={recent.duos} />
        </>
      )}

      <h3>Top Challenger pairs</h3>
      <DemoCarousel />

      <h3>How it works</h3>
      <p>
        The score uses your ranked games in our data. It looks at what each of you does in common situations and your usual gold lead at 20 minutes. 
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
