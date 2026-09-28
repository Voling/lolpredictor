import { DEMO_DUOS, DEMO_MINUTE } from "./demo_data";
import { Verdict } from "./verdict";

function gold(value: number): string {
  if (value === 0) return `even at ${DEMO_MINUTE} min`;
  return `${value > 0 ? "+" : "−"}${Math.abs(value).toLocaleString("en-US")} gold at ${DEMO_MINUTE} min`;
}

export function DemoCarousel() {
  const rows = [...DEMO_DUOS, ...DEMO_DUOS];
  return (
    <div className="demo-window" role="region" tabIndex={0} aria-label="Sample duos from the top 50 Challenger players. Hover or focus to pause.">
      <ul className="demo-track">
        {rows.map((duo, index) => (
          <li key={index} aria-hidden={index >= DEMO_DUOS.length}>
            <div className="demo-names">
              {duo.left} <span className="demo-position">{duo.leftPosition}</span> with {duo.right} <span className="demo-position">{duo.rightPosition}</span>
            </div>
            <div className="demo-result">
              <strong>{duo.score}</strong> <Verdict score={duo.score} /> <span className="demo-gold">{gold(duo.gold)}</span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
