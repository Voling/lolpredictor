import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import App from './App';

const status = { ready: true, players: 2, known_pairs: 1, model: { matches: 10, baseline_auc: 0.8 } };
const players = {
  players: [
    { puuid: 'a', riot_id: 'alpha#na1', tier: 'DIAMOND', games: 30, winrate: 0.55, main_position: 'MIDDLE' },
    { puuid: 'b', riot_id: 'beta#na1', tier: 'MASTER', games: 20, winrate: 0.51, main_position: 'TOP' },
  ],
};

beforeEach(() => {
  global.fetch = jest.fn((url) => {
    const body = url.includes('/api/status/') ? status : players;
    return Promise.resolve({ ok: true, json: () => Promise.resolve(body) });
  });
});

test('renders the masthead and the corpus status', async () => {
  render(<App />);
  expect(screen.getByText('lolpredictor')).toBeTruthy();
  await waitFor(() => expect(screen.getByText(/2 players/)).toBeTruthy());
  expect(screen.getAllByPlaceholderText(/player \d, as name#tag/).length).toBe(5);
});
