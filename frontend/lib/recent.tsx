import Link from "next/link";
import type { RecentDuo, RecentSeat } from "./api";
import { championIcon, rankName } from "./ddragon";
import { positionName } from "./positions";
import { Verdict } from "./verdict";

export type ShownDuo = RecentDuo & { href?: string };

function gold(value: number, minute: number): string {
  const rounded = Math.round(Math.abs(value) / 50) * 50;
  if (rounded === 0) return `even at ${minute} min`;
  return `${value > 0 ? "+" : "\u2212"}${rounded.toLocaleString("en-US")} gold at ${minute} min`;
}

function Seat({ seat }: { seat: RecentSeat }) {
  return (
    <span className="seat">
      {seat.name && <span className="seat-name">{seat.name}</span>}
      {(seat.champions ?? []).length > 0 && (
        <span className="seat-champions">
          {(seat.champions ?? []).map((champion) => (
            <img key={champion} src={championIcon(champion)} alt={champion} title={champion} width={24} height={24} />
          ))}
        </span>
      )}
      {rankName(seat.tier, seat.division)} <span className="demo-position">{seat.position ? positionName(seat.position) : ""}</span>
    </span>
  );
}

function Duo({ duo }: { duo: ShownDuo }) {
  return (
    <>
      <div className="demo-names"><Seat seat={duo.left} /> with <Seat seat={duo.right} /></div>
      <div className="demo-result">
        <strong>{Math.round(duo.score)}</strong> <Verdict score={duo.score} /> <span className="demo-gold">{gold(duo.gold, duo.minute)}</span>
        {duo.href && <Link href={duo.href} className="demo-gold">Open</Link>}
      </div>
    </>
  );
}

export function RecentDuos({ duos }: { duos: ShownDuo[] }) {
  return (
    <ul className="recent">
      {duos.map((duo, index) => (
        <li key={`${duo.at}-${duo.score}-${index}`}><Duo duo={duo} /></li>
      ))}
    </ul>
  );
}
