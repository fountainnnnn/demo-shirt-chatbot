"""Record a scripted walkthrough of the running app as a 1080p video.

Drives the site in headless Chrome, captures frames through the DevTools
screencast (frames arrive only when the page changes), then assembles them with
ffmpeg. Time spent waiting for a reply is compressed and idle gaps are capped,
so the final video has no dead time.

    .venv/bin/python scripts/record_demo.py --photo path/to/person.jpg

Needs the app running (default http://127.0.0.1:8010), Google Chrome and ffmpeg.
Output: demo/fitting-room-demo.mp4 (the demo/ folder is gitignored).
"""
from __future__ import annotations

import argparse
import base64
import pathlib
import shutil
import subprocess
import time

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "demo"
FRAMES = OUT / "frames"

W, H = 1920, 1080                     # frames are captured at exactly this size
ZOOM = 4 / 3                          # page zoom, so the layout is the 1440x810 design
WAIT_KEEP = 1.1                       # seconds kept from each wait for a reply
IDLE_CAP = 2.4                        # longest stretch with nothing changing

CURSOR_JS = """
(() => {
  if (window.__cur) return;
  const Z = __ZOOM__;
  const zoom = () => { document.documentElement.style.zoom = String(Z); };
  document.documentElement ? zoom() : addEventListener('DOMContentLoaded', zoom);
  const c = document.createElement('div');
  c.id = '__cur';
  c.style.cssText = 'position:fixed;left:0;top:0;width:22px;height:22px;z-index:2147483647;pointer-events:none;' +
    'transform:translate(-100px,-100px);will-change:transform';
  c.innerHTML = '<svg width="22" height="22" viewBox="0 0 24 24"><path d="M4 2l15 9-6.5 1.6L16 20l-3 1.4-3.4-7.3L5 18z" ' +
    'fill="#141414" stroke="#fff" stroke-width="1.4" stroke-linejoin="round"/></svg>';
  const add = () => document.body && document.body.appendChild(c);
  document.body ? add() : addEventListener('DOMContentLoaded', add);
  addEventListener('mousemove', e => { c.style.transform = `translate(${(e.clientX - 3) / Z}px,${(e.clientY - 2) / Z}px)`; }, true);
  addEventListener('mousedown', e => {
    const r = document.createElement('div');
    r.style.cssText = `position:fixed;left:${e.clientX / Z - 14}px;top:${e.clientY / Z - 14}px;width:28px;height:28px;border-radius:50%;` +
      'border:2px solid #141414;opacity:.55;z-index:2147483646;pointer-events:none;transition:transform .35s ease-out,opacity .35s ease-out';
    document.body.appendChild(r);
    requestAnimationFrame(() => { r.style.transform = 'scale(1.7)'; r.style.opacity = '0'; });
    setTimeout(() => r.remove(), 400);
  }, true);
  window.__cur = c;
})();
"""


class Recorder:
    def __init__(self, page, cdp):
        self.page, self.cdp = page, cdp
        self.frames: list[tuple[float, pathlib.Path]] = []
        self.waits: list[tuple[float, float]] = []
        self.pos = (W / 2, H / 2)
        self.last_ts = 0.0
        cdp.on("Page.screencastFrame", self._on_frame)

    def _on_frame(self, ev):
        ts = float(ev["metadata"]["timestamp"])
        p = FRAMES / f"f{len(self.frames):06d}.jpg"
        p.write_bytes(base64.b64decode(ev["data"]))
        self.frames.append((ts, p))
        self.last_ts = ts
        self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})

    # -- time ------------------------------------------------------------
    def now(self) -> float:
        """Page clock in the same units as frame timestamps (wall seconds)."""
        return time.time()

    def hold(self, seconds: float):
        self.page.wait_for_timeout(int(seconds * 1000))

    # -- pointer ---------------------------------------------------------
    def move(self, x: float, y: float):
        x0, y0 = self.pos
        dist = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
        steps = max(8, min(34, int(dist / 22)))
        for i in range(1, steps + 1):
            t = i / steps
            e = t * t * (3 - 2 * t)                      # ease in-out
            self.page.mouse.move(x0 + (x - x0) * e, y0 + (y - y0) * e)
            self.page.wait_for_timeout(12)
        self.pos = (x, y)

    def click(self, target, *, pause: float = 0.25, nth: int = 0, dx: float = 0, dy: float = 0):
        loc = self.page.locator(target).nth(nth) if isinstance(target, str) else target
        loc.wait_for(state="visible", timeout=20000)
        loc.scroll_into_view_if_needed()
        self.page.wait_for_timeout(150)
        box = loc.bounding_box()
        x, y = box["x"] + box["width"] / 2 + dx, box["y"] + box["height"] / 2 + dy
        self.move(x, y)
        self.page.wait_for_timeout(int(pause * 1000))
        self.page.mouse.click(x, y)

    def type_message(self, text: str):
        self.click("#message", pause=0.1)
        self.page.keyboard.type(text, delay=28)
        self.page.wait_for_timeout(250)

    # -- waits -----------------------------------------------------------
    def wait_idle(self, start_delay: float = 0.35, timeout: float = 90):
        """Wait for the assistant turn to finish; the interval is compressed in the final cut."""
        a = self.now()
        self.page.wait_for_timeout(int(start_delay * 1000))
        self.page.wait_for_function(
            "() => !document.querySelector('.trace-toggle.live') && !document.querySelector('.thinking')",
            timeout=timeout * 1000,
        )
        self.page.wait_for_timeout(250)
        self.waits.append((a, self.now()))

    def wait_for(self, selector: str, timeout: float = 90, state: str = "visible"):
        a = self.now()
        self.page.locator(selector).first.wait_for(state=state, timeout=timeout * 1000)
        self.page.wait_for_timeout(200)
        self.waits.append((a, self.now()))

    def send(self, text: str):
        self.type_message(text)
        self.page.keyboard.press("Enter")
        self.wait_idle()

    def scroll_to(self, container: str, top: float, seconds: float = 0.7):
        self.page.evaluate(
            """([sel, top, ms]) => new Promise(done => {
                const el = document.querySelector(sel); const from = el.scrollTop; const t0 = performance.now();
                const step = now => { const t = Math.min(1, (now - t0) / ms); const e = t * t * (3 - 2 * t);
                  el.scrollTop = from + (top - from) * e; t < 1 ? requestAnimationFrame(step) : done(); };
                requestAnimationFrame(step); })""",
            [container, top, int(seconds * 1000)],
        )

    def scroll_into(self, container: str, selector: str, offset: float = 24, seconds: float = 0.7):
        top = self.page.evaluate(
            """([c, s, off]) => { const el = document.querySelector(c), t = document.querySelector(s);
                 if (!el || !t) return null;
                 return Math.max(0, t.getBoundingClientRect().top - el.getBoundingClientRect().top + el.scrollTop - off); }""",
            [container, selector, offset],
        )
        if top is not None:
            self.scroll_to(container, top, seconds)


def scenario(r: Recorder, photo: str):
    p = r.page
    r.hold(1.2)

    # 1. A conversational search with clarifying questions
    r.click("[data-ex]", nth=0, pause=0.4)                       # example prompt: beach wedding
    r.wait_idle()
    r.hold(1.0)
    r.click("[data-opt]:has-text('Men')")                        # who is it for
    r.wait_idle()
    r.hold(0.8)
    r.click("[data-opt]:has-text('Short')")                      # sleeves
    r.wait_idle()
    r.hold(0.8)
    if p.locator("[data-opt]:has-text('Under $50')").count():
        r.click("[data-opt]:has-text('Under $50')")              # budget
        r.wait_idle()
    r.hold(2.4)                                                  # read the results and the reply

    # 2. Shortlist two, then drop a filter
    r.click("[data-heart]", nth=0)
    r.hold(0.5)
    r.click("[data-heart]", nth=1)
    r.hold(0.9)
    crit = p.locator("[data-crit]")
    if crit.count() > 2:
        r.click(crit.nth(crit.count() - 1))                      # remove the budget filter
        r.wait_idle(start_delay=0.5)
        r.hold(1.8)

    # 3. A precise request: exact and close matches
    r.click("#new-search", pause=0.3)
    r.hold(0.6)
    r.send("A plain black long sleeve shirt for men, under $30")
    r.hold(2.6)

    # 4. Sizing from height and weight
    r.send("I'm 178 cm and 76 kg, and I like it a bit loose")
    r.hold(0.8)
    r.scroll_into("#rack", ".sizing", offset=16)
    r.hold(2.4)
    r.click("[data-fit='regular']")
    r.wait_for(".sizing", timeout=20)
    r.hold(1.5)
    r.click("[data-fit='loose']")
    r.hold(1.2)
    r.click("[data-act='measure']")
    r.hold(0.4)
    r.click("#chest-input", pause=0.1)
    p.keyboard.type("104", delay=90)
    p.keyboard.press("Enter")
    r.hold(2.2)

    # 5. Try it on
    r.click("[data-heart]", nth=0)                               # keep the black shirt too
    r.hold(0.5)
    r.click(".sizing [data-tryit], [data-tryit]", pause=0.35)
    r.hold(1.2)
    target = p.locator(".drop, [data-act='upload'], .frame").first
    box = target.bounding_box()
    if box:
        r.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    r.hold(0.4)
    p.set_input_files("#photo-file", photo)
    r.wait_for("figure.out img, .look.out img, img[alt*='wearing' i]", timeout=120)
    r.hold(3.0)

    # 6. Another shirt from the shortlist
    wear = p.locator("[data-wear]")
    if wear.count() > 1:
        green = p.locator("[data-wear]:has-text('Green oxford')")
        r.click(green.first if green.count() else wear.nth(0))
        r.hold(0.6)
        r.click("button:has-text('Generate')")
        r.wait_for("button:has-text('Generate again'):not([disabled])", timeout=120)
        r.hold(3.0)

    # 7. Shortlist comparison
    r.click("[data-view='list']", pause=0.3)
    r.hold(2.0)
    r.scroll_to(".list", 320, 0.9)
    r.hold(2.4)
    r.scroll_to(".list", 0, 0.6)

    # 8. History: reopen the first conversation and expand its steps
    r.click("[data-view='find']", pause=0.3)
    r.hold(0.6)
    conv = p.locator("[data-conv]")
    if conv.count() > 1:
        r.click(conv.nth(conv.count() - 1))
        r.hold(1.6)
        tr = p.locator("[data-trace]")
        if tr.count():
            r.click(tr.nth(tr.count() - 1))
            r.hold(2.2)
    r.hold(1.0)


def assemble(frames, waits, out: pathlib.Path):
    frames = sorted(frames)
    if not frames:
        raise SystemExit("no frames captured")
    cuts = []                                                   # (start, end) removed from each wait
    for a, b in waits:
        if b - a > WAIT_KEEP:
            cuts.append((a + WAIT_KEEP * 0.45, b - WAIT_KEEP * 0.55))

    def keep(t):
        return not any(a < t < b for a, b in cuts)

    def mapped(t):
        return t - sum(min(t, b) - a for a, b in cuts if t > a)

    kept = [(mapped(t), f) for t, f in frames if keep(t)]
    lines, total = [], 0.0
    for i, (t, f) in enumerate(kept):
        d = (kept[i + 1][0] - t) if i + 1 < len(kept) else 1.5
        d = max(1 / 60, min(d, IDLE_CAP))
        total += d
        lines += [f"file '{f}'", f"duration {d:.4f}"]
    lines.append(f"file '{kept[-1][1]}'")
    lst = OUT / "frames.txt"
    lst.write_text("\n".join(lines))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-vf", "fps=30,scale=1920:1080:flags=lanczos,format=yuv420p",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-movflags", "+faststart", str(out)],
        check=True,
    )
    raw = frames[-1][0] - frames[0][0]
    print(f"frames {len(frames)} kept {len(kept)}; raw {raw:.1f}s -> final {total:.1f}s; waits {len(waits)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010/")
    ap.add_argument("--photo", required=True)
    ap.add_argument("--out", default=str(OUT / "fitting-room-demo.mp4"))
    a = ap.parse_args()
    shutil.rmtree(FRAMES, ignore_errors=True)
    FRAMES.mkdir(parents=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True, args=["--hide-scrollbars"])
        ctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=1)
        ctx.add_init_script(CURSOR_JS.replace("__ZOOM__", repr(ZOOM)))
        page = ctx.new_page()
        page.goto(a.url, wait_until="networkidle")
        page.wait_for_timeout(800)
        cdp = ctx.new_cdp_session(page)
        rec = Recorder(page, cdp)
        cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 95, "everyNthFrame": 1})
        try:
            scenario(rec, a.photo)
        finally:
            page.wait_for_timeout(400)
            cdp.send("Page.stopScreencast")
            page.screenshot(path=str(OUT / "last.png"))
            browser.close()
    assemble(rec.frames, rec.waits, pathlib.Path(a.out))


if __name__ == "__main__":
    main()
