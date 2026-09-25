import { expect, test } from "@playwright/test";

const ME = "bblskibs#gotg";
const DEEP = "Gryffinn#NA1";
const THIN = "ethnvo#goat";
const SAME_POSITION = "KFN Omelas#DAZE";

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

test("the home page names the served run and links to both tools", async ({ page }) => {
  // given
  await page.goto("/");

  // when
  const served = page.getByRole("row", { name: /served run/ });

  // then
  await expect(served).toContainText(/\d{8}-\d{6}/);
  await expect(page.getByRole("link", { name: "Rank your friends" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Read a pair" })).toBeVisible();
  await expect(page.getByText("API unreachable")).toHaveCount(0);
});

test("friends are ranked by one duo score and an unknown friend keeps its reason", async ({ page }) => {
  // given
  await page.goto("/friends/");
  await page.getByLabel("you").fill(ME);
  await page.getByLabel(/friends, one per line/).fill([THIN, DEEP, "ghost#none"].join("\n"));

  // when
  await page.getByRole("button", { name: "rank" }).click();

  // then
  const rows = page.locator("tbody tr");
  await expect(rows).toHaveCount(3);
  await expect(page.getByRole("heading", { level: 2 })).toContainText("your score with each friend, you as top");
  await expect(page.locator("thead th")).toHaveText(["friend", "position", "score"]);
  const scores = [
    Number(await rows.nth(0).locator("td").nth(2).innerText()),
    Number(await rows.nth(1).locator("td").nth(2).innerText()),
  ];
  expect(scores.every((score) => score > 0 && score < 100)).toBeTruthy();
  expect(scores[0]).toBeGreaterThanOrEqual(scores[1]);
  await expect(rows.nth(2)).toContainText("not in the corpus");
});

test("a pair reads as one score together", async ({ page }) => {
  // given
  await page.goto("/pair/");
  await page.getByLabel("you").fill(ME);
  await page.getByLabel("friend").fill(THIN);

  // when
  await page.getByRole("button", { name: "read" }).click();

  // then
  await expect(page.getByRole("heading", { level: 2 })).toContainText(/your score together: \d+/);
  await expect(page.getByText(/gold at \d+ minutes/)).toBeVisible();
  await expect(page.locator("tbody tr")).toHaveCount(0);
});

test("two players who share a main position are refused with the reason", async ({ page }) => {
  // given
  const query = new URLSearchParams({ a: ME, b: SAME_POSITION });

  // when
  await page.goto(`/pair/?${query}`);

  // then
  await expect(page.getByText("No reading")).toBeVisible();
  await expect(page.getByText(/two different\s+positions/)).toBeVisible();
});
