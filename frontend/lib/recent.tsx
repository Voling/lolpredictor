import type { RecentDuo, RecentSeat } from "./api";
import { championIcon, rankName } from "./ddragon";
import { positionName } from "./positions";
import { Verdict } from "./verdict";

function gold(value: number, minute: number): string {
  const rounded = Math.round(Math.abs(value) / 50) * 50;
  if (rounded === 0) return `even at ${minute} min`;
  return `${value > 0 ? "+" : "\u2212"}${rounded.toLocaleString("en-US")} gold at ${minute} min`;
}

function Seat({ seat }: { seat: RecentSeat }) {
  return (
    <span className="seat">
      {seat.champion && <img src={championIcon(seat.champion)} alt={seat.champion} width={28} height={28} />}
      {rankName(seat.tier, seat.division)} <span className="demo-position">{seat.position ? positionName(seat.position) : ""}</span>
    </span>
  );
}

export function RecentDuos({ duos }: { duos: RecentDuo[] }) {
  return (
    <ul className="recent">
      {duos.map((duo) => (
        <li key={`${duo.at}-${duo.score}-${duo.left.champion}-${duo.right.champion}`}>
          <div className="demo-names"><Seat seat={duo.left} /> with <Seat seat={duo.right} /></div>
          <div className="demo-result">
            <strong>{Math.round(duo.score)}</strong> <Verdict score={duo.score} /> <span className="demo-gold">{gold(duo.gold, duo.minute)}</span>
          </div>
        </li>
      ))}
    </ul>
  );
}
