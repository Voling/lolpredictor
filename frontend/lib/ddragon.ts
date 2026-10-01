const VERSION = "16.19.1";
const CDN = `https://ddragon.leagueoflegends.com/cdn/${VERSION}/img`;
const RENAMED: Record<string, string> = { FiddleSticks: "Fiddlesticks" };

export function championIcon(name: string): string {
  return `${CDN}/champion/${RENAMED[name] ?? name}.png`;
}

export function profileIcon(id: number): string {
  return `${CDN}/profileicon/${id}.png`;
}

export function rankName(tier: string | null, division: string | null): string {
  if (!tier) return "Unranked";
  const word = tier.charAt(0) + tier.slice(1).toLowerCase();
  return division && !["MASTER", "GRANDMASTER", "CHALLENGER"].includes(tier) ? `${word} ${division}` : word;
}
