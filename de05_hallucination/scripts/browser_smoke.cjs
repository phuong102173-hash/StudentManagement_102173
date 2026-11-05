/* Optional UI test: NODE_PATH must expose Playwright; no frontend build is required. */
const { chromium } = require("playwright");
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");

(async () => {
  const executablePath = process.env.BROWSER_EXECUTABLE || "/usr/bin/chromium";
  const browser = await chromium.launch({executablePath, headless: true, args: ["--no-sandbox"]});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1100}, deviceScaleFactor: 1});
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.goto(process.env.APP_URL || "http://127.0.0.1:8000", {waitUntil: "networkidle"});
    assert.equal(await page.locator("#examples button").count(), 4);
    await page.getByRole("button", {name: "Sai số liệu"}).click();
    await page.getByRole("button", {name: "Kiểm chứng nội dung"}).click();
    await page.locator("#result").waitFor({state: "visible"});
    assert.match(await page.locator("#verdict-label").innerText(), /CẦN KIỂM TRA LẠI/);
    assert.match(await page.locator("#findings").innerText(), /1995/);
    const downloadPromise = page.waitForEvent("download");
    await page.locator("#download").click();
    const download = await downloadPromise;
    const report = JSON.parse(fs.readFileSync(await download.path(), "utf8"));
    assert.equal(report.is_hallucination, true);
    fs.mkdirSync("artifacts", {recursive: true});
    await page.screenshot({path: path.join("artifacts", "ui-desktop.png"), fullPage: true});
    await page.getByRole("button", {name: "Có căn cứ"}).click();
    await page.locator("#submit").click();
    await page.locator("#result").waitFor({state: "visible"});
    assert.match(await page.locator("#verdict-label").innerText(), /DƯỚI NGƯỠNG/);
    await page.locator("#context").fill("");
    await page.locator("#submit").click();
    await page.locator("#result").waitFor({state: "visible"});
    assert.match(await page.locator("#verdict-label").innerText(), /CHƯA ĐỦ/);
    await page.locator("#claim").fill('<img src=x onerror="window.xss=true"> is a claim.');
    await page.locator("#context").fill("A source document.");
    await page.locator("#submit").click();
    await page.locator("#result").waitFor({state: "visible"});
    assert.equal(await page.evaluate(() => !!window.xss), false);
    await page.locator("#method").selectOption("llm_judge");
    await page.locator("#submit").click();
    await page.locator("#error").waitFor({state: "visible"});
    assert.match(await page.locator("#error").innerText(), /JUDGE_BASE_URL/);
    await page.locator("#clear-form").click();
    assert.equal(await page.locator("#claim").inputValue(), "");
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({path: path.join("artifacts", "ui-mobile.png"), fullPage: true});
    assert.deepEqual(errors, []);
    console.log("PASS: examples, detected/supported/unknown, download JSON, missing-model error, XSS escaping, reset, mobile layout.");
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
