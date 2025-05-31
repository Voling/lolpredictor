import React, { useState } from 'react';

// Standard League of Legends roles
const POSITIONS = ["Top", "Jungle", "Mid", "ADC", "Support"];

// Mock API function to simulate backend call
// Now expects an object with roles as keys and player names as values
const mockAnalyzeTeamAPI = async (teamPlayers) => {
  console.log("Sending to mock API (player names per role):", teamPlayers);
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      const filledRoles = Object.entries(teamPlayers).filter(([_, playerName]) => playerName.trim() !== "").map(([role, playerName]) => `${playerName} (${role})`);
      if (filledRoles.length === 0) {
        reject(new Error("Mock API Error: No player names provided."));
        return;
      }

      // Simulate API success/failure
      if (Math.random() > 0.1) { // 90% success rate
        resolve({
          teamCompositionSummary: filledRoles.join(', ') || "The provided players",
          overallTeamStrength: Math.floor(Math.random() * 41) + 60, // Score between 60-100
          predictedWinCondition: ["Early Aggression", "Mid-Game Objective Control", "Late Game Scaling", "Split Pushing", "Teamfight Dominance"][Math.floor(Math.random() * 5)],
          keyPlayerSynergies: [
            `Strong synergy expected between ${teamPlayers.Jungle || 'Jungle'} and ${teamPlayers.Mid || 'Mid'} based on their playstyles.`,
            `${teamPlayers.ADC || 'ADC'} and ${teamPlayers.Support || 'Support'} show potential for high lane dominance.`
          ],
          areasForImprovement: [
            "Potential coordination challenges if Top laner plays too isolated.",
            "Team may be vulnerable to early ganks if vision control is lacking."
          ],
          analysisNotes: `Based on the recent gameplay trends of ${filledRoles.join(', ')}, this team's strength lies in their adaptability. The model predicts success if they can leverage their primary win condition effectively.`,
        });
      } else {
        reject(new Error("Mock API Error: Failed to analyze team player data. Please try again."));
      }
    }, 1500); // Simulate network delay
  });
};


// Player Input Component (Simplified for each role)
function PlayerInput({ role, playerName, onChange }) {
  return (
    <div className="mb-4">
      <label htmlFor={`player-${role}`} className="block text-sm font-medium text-blue-300 mb-1">{role} Laner Player Name</label>
      <input
        type="text"
        id={`player-${role}`}
        name={role} // Name attribute matches the key in the players state object
        value={playerName}
        onChange={onChange}
        placeholder={`Enter ${role} Player Name`}
        className="w-full p-3 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm"
      />
    </div>
  );
}

// Team Builder Component
function TeamBuilder({ onAnalyze, isLoading }) {
  const [players, setPlayers] = useState(
    POSITIONS.reduce((acc, pos) => ({ ...acc, [pos]: '' }), {})
  );

  const handlePlayerNameChange = (e) => {
    const { name, value } = e.target; // 'name' will be "Top", "Jungle", etc.
    setPlayers(prevPlayers => ({
      ...prevPlayers,
      [name]: value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Basic validation: ensure at least one player name is entered
    if (Object.values(players).every(name => name.trim() === '')) {
        alert("Please enter at least one player name."); // Replace with modal in real app
        return;
    }
    onAnalyze(players);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-2">
        {POSITIONS.map((role) => (
          <PlayerInput
            key={role}
            role={role}
            playerName={players[role]}
            onChange={handlePlayerNameChange}
          />
        ))}
      </div>
      <button
        type="submit"
        disabled={isLoading}
        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg text-lg transition duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center shadow-lg focus:outline-none focus:ring-4 focus:ring-blue-400 focus:ring-opacity-50"
      >
        {isLoading ? (
          <>
            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Analyzing Players...
          </>
        ) : (
          "Analyze Team Strength"
        )}
      </button>
    </form>
  );
}

// Analysis Display Component
function AnalysisDisplay({ analysis, error }) {
  if (error) {
    return (
      <div className="mt-8 p-6 bg-red-900 border border-red-700 rounded-lg text-red-100 shadow-xl">
        <h3 className="text-2xl font-semibold mb-3">Analysis Error</h3>
        <p>{error}</p>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="mt-8 p-6 bg-gray-800 border border-gray-700 rounded-lg text-gray-400 shadow-xl">
        <p className="text-center text-lg">Enter player names for each position and click "Analyze" to see the team strength assessment.</p>
      </div>
    );
  }

  return (
    <div className="mt-10 p-6 bg-gray-800 border border-gray-700 rounded-lg shadow-2xl text-gray-200">
      <h2 className="text-4xl font-bold mb-6 text-center text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-blue-500">Team Strength Analysis</h2>
      
      <div className="bg-gray-750 p-4 rounded-md border border-gray-600 mb-6">
        <h3 className="text-xl font-semibold text-blue-300 mb-2">Team Summary</h3>
        <p className="text-lg text-gray-300">{analysis.teamCompositionSummary}</p>
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <div className="bg-gray-750 p-4 rounded-md border border-gray-600">
          <h3 className="text-xl font-semibold text-teal-300 mb-2">Overall Team Strength</h3>
          <p className="text-3xl font-bold text-green-400">{analysis.overallTeamStrength}%</p>
        </div>
        <div className="bg-gray-750 p-4 rounded-md border border-gray-600">
          <h3 className="text-xl font-semibold text-teal-300 mb-2">Predicted Win Condition</h3>
          <p className="text-lg text-purple-300">{analysis.predictedWinCondition}</p>
        </div>
      </div>

      <div className="mb-6 bg-gray-750 p-4 rounded-md border border-gray-600">
        <h3 className="text-xl font-semibold text-blue-300 mb-2">Key Player Synergies (based on gameplay trends)</h3>
        <ul className="list-disc list-inside space-y-1 text-gray-300">
            {analysis.keyPlayerSynergies.map((synergy, i) => <li key={i}>{synergy}</li>)}
        </ul>
      </div>
      
      <div className="mb-6 bg-gray-750 p-4 rounded-md border border-gray-600">
        <h3 className="text-xl font-semibold text-yellow-400 mb-2">Potential Areas for Improvement</h3>
         <ul className="list-disc list-inside space-y-1 text-gray-300">
            {analysis.areasForImprovement.map((area, i) => <li key={i}>{area}</li>)}
        </ul>
      </div>
      
      <div className="bg-gray-750 p-4 rounded-md border border-gray-600">
        <h3 className="text-xl font-semibold text-gray-300 mb-2">Model Analysis Notes</h3>
        <p className="text-gray-400 leading-relaxed italic">{analysis.analysisNotes}</p>
      </div>
    </div>
  );
}

// Main App Component
export default function App() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyzeTeam = async (teamPlayers) => {
    setIsLoading(true);
    setError(null);
    setAnalysisResult(null); 
    try {
      // In a real app, this would be:
      const response = await fetch('/api/analyze_team_players', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ players: teamPlayers }), // e.g., { players: { Top: "PlayerA", Jungle: "PlayerB", ... } }
      });
      if (!response.ok) {
        // Try to get error message from backend if available
        let errorMsg = `HTTP error! status: ${response.status}`;
        try {
            const errorData = await response.json();
            errorMsg = errorData.detail || errorData.message || errorMsg;
        } catch (e) {
            // Ignore if response is not JSON or empty
        }
        throw new Error(errorMsg);
      }
      const data = await response.json();
      setAnalysisResult(data);

      // Comment out or remove mock API call when using real backend
      // const data = await mockAnalyzeTeamAPI(teamPlayers);
      // setAnalysisResult(data);

    } catch (err) {
      console.error("Analysis failed:", err);
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 font-sans p-4 sm:p-8">
      <header className="text-center mb-10">
        <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-teal-500 via-cyan-500 to-blue-500">
          Lolpredictor 
        </h1> {/* Changed title as per your last code snippet */}
        <p className="text-lg text-gray-400 mt-2">
          Team compatibility 
        </p> {/* Changed subtitle as per your last code snippet */}
      </header>

      <main className="max-w-4xl mx-auto"> {/* Adjusted max-width for a slightly more compact form */}
        <div className="bg-gray-800 shadow-2xl rounded-xl p-6 md:p-10 border border-gray-700">
          <h2 className="text-3xl font-semibold mb-8 text-center text-teal-300">Enter Player Names by Position</h2>
          <TeamBuilder onAnalyze={handleAnalyzeTeam} isLoading={isLoading} />
        </div>
        
        { (analysisResult || error || isLoading) && (
            <div className="mt-10">
              {isLoading && (
                <div className="flex justify-center items-center p-6 bg-gray-800 border border-gray-700 rounded-lg shadow-xl">
                  <svg className="animate-spin h-10 w-10 text-teal-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <p className="ml-4 text-xl text-gray-300">Analyzing Player Data...</p>
                </div>
              )}
              {!isLoading && <AnalysisDisplay analysis={analysisResult} error={error} />}
            </div>
        )}
      </main>

      <footer className="text-center mt-12 py-6 border-t border-gray-700">
        <p className="text-gray-500">Not affiliated with Riot Games.</p> 
      </footer>
    </div>
  );
}