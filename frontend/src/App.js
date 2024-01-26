import React, { useCallback, useEffect, useMemo, useState } from 'react';
import './App.css';
import { analyseTeam, getPartners, getPlayer, getStatus, searchPlayers } from './api';
import { Donut, DriverBars, scoreColor, StyleBars } from './charts';

const SLOTS = 5;

function percent(value) {
  return value === null || value === undefined ? 'n/a' : `${(value * 100).toFixed(1)}%`;
}

function useTheme() {
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'dark');
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);
  return [theme, setTheme];
}

function Matrix({ report, selected, onSelect }) {
  const players = report.players;
  const lookup = new Map(report.pairs.map((pair) => [`${pair.a}|${pair.b}`, pair]));
  const find = (a, b) => lookup.get(`${a}|${b}`) || lookup.get(`${b}|${a}`);
  return (
    <div className="matrix" style={{ gridTemplateColumns: `120px repeat(${players.length}, 1fr)` }}>
      <span />
      {players.map((player) => (
        <span className="matrix-head" key={`head-${player.puuid}`}>
          {player.riot_id.split('#')[0]}
        </span>
      ))}
      {players.map((row) => (
        <React.Fragment key={`row-${row.puuid}`}>
          <span className="matrix-head row">{row.riot_id.split('#')[0]}</span>
          {players.map((column) => {
            if (row.puuid === column.puuid) {
              return <span className="matrix-cell self" key={`${row.puuid}-${column.puuid}`} />;
            }
            const pair = find(row.puuid, column.puuid);
            const active = selected && pair && selected.a === pair.a && selected.b === pair.b;
            return (
              <button
                type="button"
                key={`${row.puuid}-${column.puuid}`}
                className={`matrix-cell${active ? ' active' : ''}`}
                style={scoreColor(pair.score)}
                onClick={() => onSelect(pair)}
                title={`${row.riot_id} + ${column.riot_id}: ${pair.score}/100 over ${pair.games_together} shared games`}
              >
                {pair.score}
              </button>
            );
          })}
        </React.Fragment>
      ))}
    </div>
  );
}

function PairDetail({ pair, players }) {
  const a = players.find((player) => player.puuid === pair.a);
  const b = players.find((player) => player.puuid === pair.b);
  return (
    <section className="card">
      <header className="card-head">
        <h2>
          {pair.a_riot_id} <span className="plus">+</span> {pair.b_riot_id}
        </h2>
        <span className="pill" style={scoreColor(pair.score)}>
          {pair.score}/100
        </span>
      </header>
      <p className="lede">
        Compatibility is a percentile against every pairing in the corpus. 50 is an average pairing;
        the drivers below are the modelled contribution of each behaviour axis to the pair&apos;s
        win-rate residual.
      </p>
      <DriverBars drivers={pair.drivers} />
      {a && b && <StyleBars series={[a, b]} />}
      <p className="foot">
        {pair.games_together > 0
          ? `${pair.games_together} games together in the corpus`
          : 'no games together in the corpus, so the score rests on playstyle alone'}
      </p>
    </section>
  );
}

function PlayerPanel({ query }) {
  const [player, setPlayer] = useState(null);
  const [partners, setPartners] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    setPlayer(null);
    setPartners(null);
    setError(null);
    Promise.all([getPlayer(query), getPartners(query, 6)])
      .then(([profile, mates]) => {
        if (!active) return;
        setPlayer(profile);
        setPartners(mates);
      })
      .catch((exception) => active && setError(exception.message));
    return () => {
      active = false;
    };
  }, [query]);

  if (error) return <p className="error">{error}</p>;
  if (!player) return <p className="empty">loading player…</p>;

  const tendencies = Object.entries(player.tendencies || {});
  const measured = tendencies.filter(([, value]) => value.measurable);
  const unmeasured = tendencies.filter(([, value]) => !value.measurable);

  return (
    <div className="player-panel">
      <section className="card">
        <header className="card-head">
          <h2>{player.riot_id}</h2>
          <span className="tag">
            {player.tier} {player.division} · {player.main_position}
          </span>
        </header>
        <div className="stat-row">
          <Donut value={player.winrate} label={percent(player.winrate)} caption="win rate" />
          <div className="hero">
            <strong>{player.games}</strong>
            <span className="caption">games</span>
          </div>
          <div className="hero">
            <strong>{player.champion_pool}</strong>
            <span className="caption">champions</span>
          </div>
          <div className="hero">
            <strong>{Math.round((player.style_confidence ?? 1) * 100)}%</strong>
            <span className="caption">profile confidence</span>
          </div>
        </div>
        <p className="foot">
          Percentiles compare this player to others on the same champion in the same role, over the
          first 15 minutes only.
        </p>
        <StyleBars series={[player]} />
      </section>

      <section className="card">
        <h3>Tendencies, given the same situation</h3>
        <DriverBars
          drivers={measured.map(([name, value]) => ({
            axis: name,
            label: `${name.replace(/_/g, ' ')} (${value.chances} chances)`,
            impact: value.log_ratio,
          }))}
        />
        {unmeasured.length > 0 && (
          <p className="foot">
            Not separable from chance in this corpus:{' '}
            {unmeasured.map(([name]) => name.replace(/_/g, ' ')).join(', ')}.
          </p>
        )}
      </section>

      {partners && (
        <section className="card">
          <h3>Modelled best and worst partners</h3>
          <div className="split">
            <ol className="partner-list">
              {partners.best.map((entry) => (
                <li key={entry.puuid}>
                  <span>{entry.riot_id}</span>
                  <span className="pill" style={scoreColor(entry.score)}>
                    {entry.score}
                  </span>
                </li>
              ))}
            </ol>
            <ol className="partner-list">
              {partners.worst.map((entry) => (
                <li key={entry.puuid}>
                  <span>{entry.riot_id}</span>
                  <span className="pill" style={scoreColor(entry.score)}>
                    {entry.score}
                  </span>
                </li>
              ))}
            </ol>
          </div>
        </section>
      )}
    </div>
  );
}

export default function App() {
  const [theme, setTheme] = useTheme();
  const [status, setStatus] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [inputs, setInputs] = useState(Array(SLOTS).fill(''));
  const [report, setReport] = useState(null);
  const [selected, setSelected] = useState(null);
  const [focus, setFocus] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getStatus()
      .then(setStatus)
      .catch((exception) => setError(exception.message));
    searchPlayers('', 12)
      .then((payload) => setSuggestions(payload.players))
      .catch(() => setSuggestions([]));
  }, []);

  const filled = useMemo(() => inputs.map((value) => value.trim()).filter(Boolean), [inputs]);

  const run = useCallback(async () => {
    if (filled.length < 2) {
      setError('enter at least two players');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const payload = await analyseTeam(filled);
      setReport(payload);
      setSelected(payload.weakest);
      setFocus(payload.players[0].puuid);
    } catch (exception) {
      setError(exception.message);
      setReport(null);
    } finally {
      setBusy(false);
    }
  }, [filled]);

  const update = (index, value) => {
    setInputs((current) => current.map((item, position) => (position === index ? value : item)));
  };

  return (
    <div className="app">
      <header className="masthead">
        <div>
          <h1>lolpredictor</h1>
          <p>
            Player compatibility for ranked solo queue, learned from match timelines rather than raw
            win rate.
          </p>
        </div>
        <div className="masthead-side">
          {status && (
            <span className="status">
              {status.ready
                ? `${status.players} players · ${status.model.matches || 0} matches · baseline AUC ${
                    status.model.baseline_auc ?? 'n/a'
                  }`
                : 'model not trained'}
            </span>
          )}
          <button type="button" className="ghost" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
            {theme === 'dark' ? 'light' : 'dark'} mode
          </button>
        </div>
      </header>

      <section className="card form">
        <div className="inputs">
          {inputs.map((value, index) => (
            <input
              key={index}
              list="known-players"
              placeholder={`player ${index + 1}, as name#tag`}
              value={value}
              onChange={(event) => update(index, event.target.value)}
              onKeyDown={(event) => event.key === 'Enter' && run()}
            />
          ))}
          <datalist id="known-players">
            {suggestions.map((player) => (
              <option key={player.puuid} value={player.riot_id} />
            ))}
          </datalist>
        </div>
        <div className="actions">
          <button type="button" onClick={run} disabled={busy}>
            {busy ? 'scoring…' : 'score the group'}
          </button>
          {suggestions.length > 0 && (
            <button
              type="button"
              className="ghost"
              onClick={() => setInputs(suggestions.slice(0, SLOTS).map((player) => player.riot_id))}
            >
              fill with the most-seen players
            </button>
          )}
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      {report && (
        <>
          <section className="card summary">
            <div className="hero big">
              <strong>{report.team_score}</strong>
              <span className="caption">group compatibility, 0–100</span>
            </div>
            <div className="hero">
              <strong>{report.strongest.score}</strong>
              <span className="caption">
                best pair · {report.strongest.a_riot_id} + {report.strongest.b_riot_id}
              </span>
            </div>
            <div className="hero">
              <strong>{report.weakest.score}</strong>
              <span className="caption">
                weakest link · {report.weakest.a_riot_id} + {report.weakest.b_riot_id}
              </span>
            </div>
          </section>

          <section className="card">
            <h3>Pairwise compatibility</h3>
            <Matrix report={report} selected={selected} onSelect={setSelected} />
            <p className="legend">
              <span className="swatch" style={{ background: 'var(--neg-4)' }} /> below average
              <span className="swatch" style={{ background: 'var(--neutral)' }} /> average
              <span className="swatch" style={{ background: 'var(--pos-4)' }} /> above average
            </p>
          </section>

          {selected && <PairDetail pair={selected} players={report.players} />}

          <nav className="tabs">
            {report.players.map((player) => (
              <button
                type="button"
                key={player.puuid}
                className={focus === player.puuid ? 'active' : ''}
                onClick={() => setFocus(player.puuid)}
              >
                {player.riot_id}
              </button>
            ))}
          </nav>
          {focus && <PlayerPanel query={focus} />}
        </>
      )}
    </div>
  );
}
