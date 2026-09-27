import { expect, test } from "@playwright/test";

const ME = "bblskibs#gotg";
const DEEP = "Gryffinn#NA1";
const THIN = "ethnvo#goat";
const SAME_POSITION = "KFN Omelas#DAZE";
const VERDICTS = /Good duo|Ok pairing|I've seen better duos|Bad pairing!/;

test("the api answers its health check and refuses an unknown player by name", async ({ request }) => {
  // given
  const unknown = new URLSearchParams({ a: "nobody#none", b: THIN });

  // when
  const health = await request.get("/api/health");
  const refused = await request.get(`/api/pair?${unknown}`);

  // then
  expect(health.ok()).toBeTruthy();
  expect(refused.status()).toBe(404);
  expect(await refused.text()).toContain("nobody#none");
});

test("the home page explains the app, the score bands and names the model", async ({ page }) => {
  // given
  await page.goto("/");

  // when
  const legend = page.locator("ul.legend li");

  // then
  await expect(page.getByRole("link", { name: "Rank your friends" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Check one duo" })).toBeVisible();
  await expect(legend).toHaveCount(4);
  await expect(legend.nth(0)).toContainText("Good duo");
  await expect(legend.nth(3)).toContainText("Bad pairing!");
  await expect(page.locator("p.footer")).toContainText(/Model \d{8}-\d{6}/);
  await expect(page.getByText("The server isn't reachable right now.")).toHaveCount(0);
});

test("friends are ranked by one duo score with a verdict and an unknown friend keeps its reason", async ({ page }) => {
  // given
  await page.goto("/friends/");
  await page.getByLabel("Your Riot ID").fill(ME);
  await page.getByLabel(/Friends, one per line/).fill([THIN, DEEP, "ghost#none"].join("\n"));

  // when
  await page.getByRole("button", { name: "Rank" }).click();

  // then
  const rows = page.locator("tbody tr");
  await expect(rows).toHaveCount(3);
  await expect(page.getByRole("heading", { level: 2 })).toContainText("Your duo scores as top");
  await expect(page.locator("thead th")).toHaveText(["Friend", "Their position", "Score", "Verdict"]);
  const scores = [
    Number(await rows.nth(0).locator("td").nth(2).innerText()),
    Number(await rows.nth(1).locator("td").nth(2).innerText()),
  ];
  expect(scores.every((score) => score > 0 && score < 100)).toBeTruthy();
  expect(scores[0]).toBeGreaterThanOrEqual(scores[1]);
  await expect(rows.nth(0).locator(".verdict")).toHaveText(VERDICTS);
  await expect(rows.nth(2)).toContainText("Not in our data yet.");
});

test("a pair reads as one score with a verdict and a projected lead", async ({ page }) => {
  // given
  await page.goto("/pair/");
  await page.getByLabel("Your Riot ID").fill(ME);
  await page.getByLabel("Friend's Riot ID").fill(THIN);

  // when
  await page.getByRole("button", { name: "Check" }).click();

  // then
  await expect(page.getByRole("heading", { level: 2 })).toContainText(/Your duo score: \d+/);
  await expect(page.locator("p.verdict-line .verdict")).toHaveText(VERDICTS);
  await expect(page.getByText(/At \d+ minutes you two are projected to be/)).toBeVisible();
});

test("two players who share a main position are refused with a plain reason", async ({ page }) => {
  // given
  const query = new URLSearchParams({ a: ME, b: SAME_POSITION });

  // when
  await page.goto(`/pair/?${query}`);

  // then
  await expect(page.getByText("Can't score this duo.")).toBeVisible();
  await expect(page.getByText(/both play top\. Pick a different position for one of you\./)).toBeVisible();
});
