// Opens the demo from link-demo.sh in headless Chromium, on SwiftShader so that
// it needs no GPU, and checks that it draws: the clear color in a corner, and
// the checkerboard's top-left square white and bottom-left square black.
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
const page = await browser.newPage({ viewport: { width: 800, height: 600 } });
const failures = [];
// What the demo prints when something goes wrong.
const failure = /^(error|device lost|requestAdapter|requestDevice|getCurrentTexture failed|surface has no formats)|Aborted/;
let presented;
const firstFrame = new Promise((resolve) => (presented = resolve));
page.on('console', (message) => {
  const line = message.text();
  console.log(line);
  if (failure.test(line)) {
    failures.push(line);
  }
  if (line === 'first frame presented') {
    presented();
  }
});
page.on('pageerror', (error) => failures.push(`page error: ${error.message}`));

const timeout = (milliseconds, what) =>
  new Promise((_, reject) => setTimeout(() => reject(new Error(`timed out waiting for ${what}`)), milliseconds));

let pixels;
try {
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await Promise.race([firstFrame, timeout(120_000, 'the first frame')]);
  // Reads the default framebuffer right after a present's blit, while it holds the frame.
  pixels = await Promise.race([
    page.evaluate(() => new Promise((resolve) => {
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
      const blit = gl.blitFramebuffer;
      gl.blitFramebuffer = function (...args) {
        const result = blit.apply(gl, args);
        gl.blitFramebuffer = blit;
        // y counts from the bottom. The quad spans the middle half of the canvas.
        resolve({ corner: read(0.01, 0.01), quadTopLeft: read(0.26, 0.74), quadBottomLeft: read(0.26, 0.26) });
        return result;
      };
    })),
    timeout(30_000, 'a present'),
  ]);
  console.log(`pixels: ${JSON.stringify(pixels)}`);
  if (screenshot) {
    await page.screenshot({ path: screenshot });
  }
} catch (error) {
  failures.push(error.message);
}

await browser.close();
server.close();

const close = (actual, expected) => actual && actual.every((value, i) => Math.abs(value - expected[i]) <= 2);
if (pixels) {
  const expected = { corner: [100, 149, 237, 255], quadTopLeft: [255, 255, 255, 255], quadBottomLeft: [0, 0, 0, 255] };
  for (const [where, color] of Object.entries(expected)) {
    if (!close(pixels[where], color)) {
      failures.push(`${where} is ${JSON.stringify(pixels[where])}, expected ${JSON.stringify(color)}`);
    }
  }
}
for (const line of failures) {
  console.log(`FAIL ${line}`);
}
console.log(failures.length === 0 ? 'PASS' : `${failures.length} failure(s)`);
process.exit(failures.length === 0 ? 0 : 1);
