// Opens surface-test from link.sh in headless Chromium, on SwiftShader, in each
// scene with and without Present, and checks what the page shows in the middle
// of the canvas, as a WebGPU canvas would show it. It takes a screenshot,
// because what the browser shows can't be read back from WebGL.
//
// Usage: node .github/webgl2/surface-test.mjs <directory with index.html>

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { chromium } from 'playwright';

const root = process.argv[2];
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
const timeout = (milliseconds, what) =>
  new Promise((_, reject) => setTimeout(() => reject(new Error(`timed out waiting for ${what}`)), milliseconds));

const red = [255, 0, 0];
const green = [0, 255, 0];
// The page's background (#222) shows through a canvas that is transparent black.
const background = [34, 34, 34];
const scenes = [
  { scene: 'draw', expected: red, what: 'what was drawn' },
  { scene: 'skip', expected: background, what: 'transparent black when nothing was submitted' },
  { scene: 'write', expected: green, what: 'what writeTexture wrote' },
];

let failures = 0;
for (const { scene, expected, what } of scenes) {
  for (const present of [true, false]) {
    const name = `${scene}${present ? '' : ' without Present'} shows ${what}`;
    const page = await browser.newPage({ viewport: { width: 800, height: 600 } });
    const errors = [];
    let ready;
    const isReady = new Promise((resolve) => (ready = resolve));
    page.on('console', (message) => {
      const line = message.text();
      if (line === 'ready') {
        ready();
      } else if (/^(FAIL|device lost)|Aborted/.test(line)) {
        errors.push(line);
      }
    });
    page.on('pageerror', (error) => errors.push(`page error: ${error.message}`));
    let color;
    try {
      await page.goto(`http://127.0.0.1:${server.address().port}/?scene=${scene}${present ? '' : '&present=no'}`);
      await Promise.race([isReady, timeout(120_000, 'ready')]);
      // The middle of the 640x360 canvas at the top left of the page.
      const png = await page.screenshot({ clip: { x: 320, y: 180, width: 1, height: 1 } });
      color = await page.evaluate(async (base64) => {
        const image = new Image();
        image.src = `data:image/png;base64,${base64}`;
        await image.decode();
        const context = new OffscreenCanvas(1, 1).getContext('2d');
        context.drawImage(image, 0, 0);
        return Array.from(context.getImageData(0, 0, 1, 1).data.slice(0, 3));
      }, png.toString('base64'));
    } catch (error) {
      errors.push(error.message);
    }
    await page.close();
    if (color && !color.every((value, i) => Math.abs(value - expected[i]) <= 2)) {
      errors.push(`shows ${JSON.stringify(color)}, expected ${JSON.stringify(expected)}`);
    }
    if (errors.length === 0) {
      console.log(`PASS ${name}`);
    } else {
      console.log(`FAIL ${name}: ${errors.join('; ')}`);
      failures++;
    }
  }
}

await browser.close();
server.close();
console.log(failures === 0 ? 'All tests passed' : `${failures} test(s) failed`);
process.exit(failures === 0 ? 0 : 1);
