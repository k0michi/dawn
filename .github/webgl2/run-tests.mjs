// Opens a test page from link.sh in headless Chromium on SwiftShader and waits
// for it to print "All tests passed" or "<n> test(s) failed".
//
// Usage: node .github/webgl2/run-tests.mjs <directory with index.html>

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
const page = await browser.newPage();
let result = '';
let pageError = '';
const finished = new Promise((resolve) => {
  page.on('console', (message) => {
    const line = message.text();
    console.log(line);
    if (line === 'All tests passed' || / test\(s\) failed$/.test(line)) {
      result = line;
      resolve();
    }
  });
  page.on('pageerror', (error) => {
    pageError = error.stack ?? error.message;
    resolve();
  });
});
const timeout = new Promise((resolve) => setTimeout(resolve, 120_000));

await page.goto(`http://127.0.0.1:${server.address().port}/`);
await Promise.race([finished, timeout]);
await browser.close();
server.close();

if (pageError) {
  console.log(`FAIL page error: ${pageError}`);
} else if (!result) {
  console.log('FAIL timed out');
}
process.exit(result === 'All tests passed' && !pageError ? 0 : 1);
