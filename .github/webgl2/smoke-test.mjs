// Opens the demo from link.sh in headless Chromium, on SwiftShader so that
// it needs no GPU, and checks that it draws: the clear color in a corner, and
// the checkerboard's top-left square white and bottom-left square black. It
// checks the canvas at the end of the animation frame that drew the first
// frame, with Present and without it, so a frame shown late fails.
//
// Usage: node .github/webgl2/smoke-test.mjs <directory with index.html> [screenshot.png]

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { chromium } from 'playwright';

const root = process.argv[2];
const screenshot = process.argv[3];
const types = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm' };
const server = createServer(async (request, response) => {
  try {
    const path = join(root, new URL(request.url, 'http://localhost').pathname.replace(/\/$/, '/index.html'));
    const body = await readFile(path);
    response.writeHead(200, { 'Content-Type': types[extname(path)] ?? 'application/octet-stream' });
    response.end(body);
  } catch {
    response.writeHead(404);
    response.end();
  }
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));

const browser = await chromium.launch({
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
// What the demo prints when something goes wrong.
const failure = /^(error|device lost|requestAdapter|requestDevice|getCurrentTexture failed|surface has no formats)|Aborted/;
const timeout = (milliseconds, what) =>
  new Promise((_, reject) => setTimeout(() => reject(new Error(`timed out waiting for ${what}`)), milliseconds));
const failures = [];
const pixelsOf = {};

// Once with Present, and once without, as programs for Emdawnwebgpu draw.
for (const mode of ['present', 'no present']) {
  const page = await browser.newPage({ viewport: { width: 800, height: 600 } });
  const fail = (message) => failures.push(`${mode}: ${message}`);
  page.on('console', (message) => {
    const line = message.text();
    console.log(`[${mode}] ${line}`);
    if (failure.test(line)) {
      fail(line);
    }
  });
  page.on('pageerror', (error) => fail(`page error: ${error.message}`));
  // Reads the canvas at the end of the animation frame callback that drew the first frame,
  // before the browser shows it: the canvas must hold the frame by then, not in a later task.
  await page.addInitScript(() => {
    let drawn = false;
    const log = console.log;
    console.log = (...args) => {
      drawn ||= args[0] === 'first frame drawn';
      return log.apply(console, args);
    };
    const requestAnimationFrame = window.requestAnimationFrame;
    window.requestAnimationFrame = (callback) => requestAnimationFrame.call(window, (time) => {
      callback(time);
      if (!drawn || window.firstFramePixels) {
        return;
      }
      const canvas = document.getElementById('canvas');
      const gl = canvas.getContext('webgl2');
      const read = (x, y) => {
        const pixel = new Uint8Array(4);
        const readFramebuffer = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING);
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
        gl.readPixels(Math.round(canvas.width * x), Math.round(canvas.height * y), 1, 1, gl.RGBA,
                      gl.UNSIGNED_BYTE, pixel);
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, readFramebuffer);
        return Array.from(pixel);
      };
      // y counts from the bottom. The quad spans the middle half of the canvas.
      window.firstFramePixels = { corner: read(0.01, 0.01), quadTopLeft: read(0.26, 0.74), quadBottomLeft: read(0.26, 0.26) };
    });
  });
  try {
    await page.goto(`http://127.0.0.1:${server.address().port}/${mode === 'present' ? '' : '?present=no'}`);
    await page.waitForFunction(() => window.firstFramePixels, null, { timeout: 120_000 });
    pixelsOf[mode] = await page.evaluate(() => window.firstFramePixels);
    console.log(`[${mode}] pixels: ${JSON.stringify(pixelsOf[mode])}`);
    if (screenshot && mode === 'no present') {
      // Let a few frames go by: the canvas must keep showing the demo.
      await Promise.race([page.evaluate(() => new Promise((resolve) => {
        let frames = 0;
        const next = () => (++frames === 10 ? resolve() : requestAnimationFrame(next));
        requestAnimationFrame(next);
      })), timeout(30_000, 'animation frames')]);
      await page.screenshot({ path: screenshot });
    }
  } catch (error) {
    fail(error.message);
  }
  await page.close();
}

await browser.close();
server.close();

const close = (actual, expected) => actual && actual.every((value, i) => Math.abs(value - expected[i]) <= 2);
const expected = { corner: [100, 149, 237, 255], quadTopLeft: [255, 255, 255, 255], quadBottomLeft: [0, 0, 0, 255] };
for (const [mode, pixels] of Object.entries(pixelsOf)) {
  for (const [where, color] of Object.entries(expected)) {
    if (!close(pixels[where], color)) {
      failures.push(`${mode}: ${where} is ${JSON.stringify(pixels[where])}, expected ${JSON.stringify(color)}`);
    }
  }
}
for (const line of failures) {
  console.log(`FAIL ${line}`);
}
console.log(failures.length === 0 ? 'PASS' : `${failures.length} failure(s)`);
process.exit(failures.length === 0 ? 0 : 1);
