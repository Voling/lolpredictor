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
  const { data: recent } = useRemote(getRecent, "recent");
  const daily = account.me?.daily;

  return (
    <main>
      <h1>lolpredictor</h1>
      <p className="sub">Find out which friends you play best with in ranked.</p>
      <p className="nav"><Link href="/friends/">Rank friends</Link><Link href="/pair/">Check a duo</Link>{authEnabled && <Link href="/account/">Account</Link>}{authEnabled && <Link href="/history/">Past checks</Link>}</p>

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
        Your playstyle is how you tend to react to events and what events you initiate. When you play with a teammate who likes to take dragon, what are you doing?
        Where are you when you ward in the first 5 minutes? Where do you stand in lane against your opponent? We count what you did each time.
      </p>
      <p>
        We don&apos;t need all your games. Every reading starts from what a typical player in your position does, learned from over 100,000
        Master and higher games and each of your games further define your playstyle. League is a repetitive game so there's lots to learn about you.
      </p>
      <p>
        From both playstyles, your champions and any games you two played together, your score predicts how far ahead your two
        positions will be in gold at 20 minutes. Teams that are ahead at 20 minutes win more often.
      </p>
      <p className="sub">A score is simply a prediction. Any one game can go either way but higher scores often result in a higher winrate.</p>

      {error && <div className="gate"><strong>The server isn&apos;t reachable right now.</strong>{error}</div>}
      {status?.run && <p className="footer">Model {status.run.id}</p>}
    </main>
  );
}
