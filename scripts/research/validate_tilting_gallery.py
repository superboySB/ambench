# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify real tilt-gallery controls and native MP4 playback in existing Chrome.

This CPU-only standard-library tool installs nothing on the host. Run offline
first, then repeat with --url after the matching GitHub Pages release is live.
--media decodes, seeks near the end and plays every actual topic MP4.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import signal
import socket
import struct
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]

UI_CHECKS = r"""(()=>{
 const checks=[];const check=(label,passed,actual)=>checks.push({label,passed:Boolean(passed),actual});
 check('analytic numerical self-check',document.querySelector('#analytic-checks').dataset.status==='passed',document.querySelector('#analytic-checks').textContent);
 const none=omniAnalytic([4,4,4,4,4,4],[0,0,0,0,0,0]),single=omniAnalytic([4,0,0,0,0,0],[30,0,0,0,0,0]);
 check('actual trim fixed allocation rank4',allocationRank([0,0,0,0,0,0])===4);check('active cos/sin rank6',allocationRank([0,0,0,0,0,0],true)===6);
 check('zero tilt force 24N',Math.abs(none.bodyForce[2]-24)<1e-9,none);
 check('single exact geometry',Math.abs(single.bodyForce[0]+1)<1e-9&&Math.abs(single.bodyForce[1]-Math.sqrt(3))<1e-9,single);
 for(const family of ['scripted','act','dp','pi0','pi05']){
   const policy=document.querySelector('#policy');policy.value=family;policy.dispatchEvent(new Event('change'));
   for(const item of document.querySelectorAll('.node')){item.dispatchEvent(new MouseEvent("click"));check(`flow ${family}/${item.dataset.step}`,document.querySelector('#step-detail').textContent.length>60&&item.getAttribute('aria-pressed')==='true',document.querySelector('#step-counter').textContent);}
 }
 document.querySelector('.node').dispatchEvent(new MouseEvent('click'));
 document.querySelector('#analytic-level').click();check('level button',Math.abs(Number(document.querySelector('#analytic-values').dataset.bodyFx))<1e-9&&Number(document.querySelector('#analytic-values').dataset.bodyFz)===24);
 document.querySelector('#analytic-lateral').click();check('lateral button level body Fx8N',Math.abs(Number(document.querySelector('#analytic-values').dataset.bodyFx)-8)<1e-9);
 document.querySelector('#analytic-reset').click();

 const names=RECORDS.map(row=>row.name), identities=[...new Set(RECORDS.map(row=>row.trial_id))];
 check('unique record names',names.length===new Set(names).size);
 check('one trial has exactly three real cameras',identities.every(id=>RECORDS.filter(row=>row.trial_id===id).map(row=>row.camera).sort().join(',')==='base_camera,ee_camera,scene_camera'));
 check('all source/video hashes present',RECORDS.every(row=>['source_sha256','source_report_sha256','gif_sha256','mp4_sha256'].every(key=>/^[0-9a-f]{64}$/.test(row[key]))));
 check('correct aircraft scope',RECORDS.every(row=>row.task_id.includes('-Am-OmniHexa-')||row.experiment_id==='tilting_reference_fahexa'&&row.task_id.includes('-Am-FAHexa-')));
 for(const record of RECORDS.filter(row=>row.policy_family!=='scripted'))check(`model no fabricated reset ${record.name}`,record.initial_observation_available===false);
 for(const selector of FILTERS){
   const filter=document.querySelector(selector);
   for(const option of filter.options){filter.value=option.value;filter.dispatchEvent(new Event('change'));const visible=trials.filter(item=>!item.container.hidden);const valueOfTrial={'#task-filter':item=>item.task,'#seed-filter':item=>String(item.record.env_seed),'#outcome-filter':item=>resultType(item.record),'#record-policy':item=>item.family,'#training-filter':item=>item.regime};const expected=trials.filter(item=>option.value==='all'||valueOfTrial[selector](item)===option.value);check(`filter ${selector}/${option.value}`,visible.length===expected.length&&visible.every(item=>expected.includes(item)),{visible:visible.length,expected:expected.length});}
   filter.value='all';filter.dispatchEvent(new Event('change'));
 }
 for(const trial of [...summaryTrials,...policyTrials]){
   const selector=document.querySelector('#telemetry-trial');selector.value=trial.name;selector.dispatchEvent(new Event('change'));
   const reference=trial.profile?.robot_id==='fa_hexa';
   const paths=document.querySelectorAll('#tilt-command-chart polyline').length;
   check(`actual telemetry ${trial.name}`,!document.querySelector('#telemetry-detail').hidden&&(reference?paths===0:paths===12),{commandPaths:paths});
   if(!reference){
     const model=policyByName.has(trial.name),measured=document.querySelectorAll('#tilt-measured-chart polyline').length;
     check(`actual measured phases ${trial.name}`,measured===(model?6:12),{measuredPaths:measured,model});
     const polylines=[...document.querySelectorAll('#tilt-measured-chart polyline')];
     const firstX=line=>Number(line.getAttribute('points').split(' ')[0].split(',')[0]);
     const t0=selectedPoints[0].timestamp_s,t1=selectedPoints.at(-1).timestamp_s;
     if(model)check(`model post-step no extra dt ${trial.name}`,Math.abs(firstX(polylines[0])-52)<.011);
     else {
       const beforeX=firstX(polylines[0]),afterX=firstX(polylines[6]);
       check(`script actual after at t+dt ${trial.name}`,Math.abs(afterX-beforeX-trial.step_dt_s/(t1-t0)*494)<.021,{beforeX,afterX,dt:trial.step_dt_s});
     }

   }
   const point=document.querySelector('#telemetry-point');point.value=point.max;point.dispatchEvent(new Event('input'));
   check(`point cursor last ${trial.name}`,Number(point.value)===selectedPoints.length-1&&document.querySelector('#point-detail').textContent.includes('真实采样'));
   if(trial.profile?.robot_id==='omni_hexa')check(`terminal actual-after reset null ${trial.name}`,selectedPoints.at(-1).motor_arm_actual_after_rad===null&&document.querySelector('#point-detail').textContent.includes('terminal / reset')&&chartCursors.filter(cursor=>cursor.timeOffsetS>0).every(cursor=>cursor.line.getAttribute('visibility')==='hidden'));

 }
 check('honest analytical and hardware boundary',document.body.textContent.includes('非 Isaac 仿真')&&document.body.textContent.includes('PX4')&&document.body.textContent.includes('RL'));
 return checks;
})()
"""


class DevTools:
    """Talk to an existing Chrome DevTools websocket using the standard library."""

    def __init__(self, url: str) -> None:
        parsed = urlsplit(url)
        self.sock = socket.create_connection((parsed.hostname, parsed.port), timeout=30)
        self.buffer = b""
        self.next_id = 0
        self.events = []
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            (
                f"GET {parsed.path} HTTP/1.1\r\nHost: {parsed.netloc}\r\nUpgrade: websocket\r\nConnection:"
                f" Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        headers = b""
        while b"\r\n\r\n" not in headers:
            headers += self.sock.recv(4096)
        header, self.buffer = headers.split(b"\r\n\r\n", 1)
        if header.splitlines()[0].split()[1] != b"101":
            raise RuntimeError(f"Chrome websocket handshake failed: {header.splitlines()[0]!r}")

    def exact(self, count: int) -> bytes:
        while len(self.buffer) < count:
            chunk = self.sock.recv(max(4096, count - len(self.buffer)))
            if not chunk:
                raise RuntimeError("Chrome websocket closed")
            self.buffer += chunk
        result, self.buffer = self.buffer[:count], self.buffer[count:]
        return result

    def send(self, payload: bytes, opcode: int = 1) -> None:
        mask = os.urandom(4)
        length = len(payload)
        if length < 126:
            header = bytes([0x80 | opcode, 0x80 | length])
        elif length < 65536:
            header = bytes([0x80 | opcode, 0xFE]) + struct.pack("!H", length)
        else:
            header = bytes([0x80 | opcode, 0xFF]) + struct.pack("!Q", length)
        self.sock.sendall(header + mask + bytes(value ^ mask[index % 4] for index, value in enumerate(payload)))

    def receive(self) -> dict:
        result = b""
        while True:
            first, second = self.exact(2)
            length = second & 127
            if length == 126:
                length = struct.unpack("!H", self.exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self.exact(8))[0]
            mask = self.exact(4) if second & 128 else None
            payload = self.exact(length)
            if mask:
                payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
            opcode = first & 15
            if opcode == 8:
                raise RuntimeError("Chrome websocket closed")
            if opcode == 9:
                self.send(payload, 10)
                continue
            if opcode in (0, 1):
                result += payload
                if first & 128:
                    return json.loads(result)

    def call(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        identifier = self.next_id
        self.send(json.dumps({"id": identifier, "method": method, "params": params or {}}).encode())
        while True:
            reply = self.receive()
            if reply.get("id") == identifier:
                if "error" in reply:
                    raise RuntimeError(reply["error"])
                return reply["result"]
            if "method" in reply:
                self.events.append(reply)

    def evaluate(self, expression: str) -> object:
        response = self.call(
            "Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True}
        )
        if "exceptionDetails" in response:
            raise RuntimeError(response["exceptionDetails"])
        return response["result"].get("value")


def launch_browser(args: argparse.Namespace) -> tuple[subprocess.Popen, DevTools, object]:
    """Start one owned CPU browser and connect to its page without host packages."""
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile = args.output.parent / f"chrome-profile-{time.time_ns()}"
    log = (args.output.parent / "chrome.log").open("w")
    process = subprocess.Popen(
        [
            str(args.chrome),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--autoplay-policy=no-user-gesture-required",
            f"--user-data-dir={profile}",
            f"--remote-debugging-port={port}",
            "--remote-debugging-address=127.0.0.1",
            "about:blank",
        ],
        stdout=log,
        stderr=log,
        start_new_session=True,
    )
    deadline = time.monotonic() + 20
    try:
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1) as response:
                    pages = json.load(response)
                page = next(page for page in pages if page["type"] == "page")
                break
            except (OSError, StopIteration):
                if time.monotonic() > deadline or process.poll() is not None:
                    raise RuntimeError("Owned Chrome DevTools did not start") from None
                time.sleep(0.1)
        client = DevTools(page["webSocketDebuggerUrl"])
        for domain in ("Page", "Runtime", "Log"):
            client.call(f"{domain}.enable")
        return process, client, log
    except Exception:
        stop_browser(process, None, log)
        raise


def stop_browser(process: subprocess.Popen, client: DevTools | None, log: object) -> None:
    """Close only this tool's socket and owned Chrome process group."""
    if client is not None:
        client.sock.close()
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    log.close()


def open_topic(client: DevTools, url: str) -> bytes:
    """Wait for the real HTML script and return the browser-loaded document bytes."""
    client.call("Page.navigate", {"url": url})
    deadline = time.monotonic() + 30
    while not client.evaluate(
        "document.readyState==='complete' && typeof RECORDS!=='undefined'"
        " && Boolean(document.querySelector('#analytic-checks'))"
    ):
        if time.monotonic() > deadline:
            raise RuntimeError("The actual tilt topic did not initialize")
        time.sleep(0.1)
    frame = client.call("Page.getResourceTree")["frameTree"]["frame"]
    loaded = client.call("Page.getResourceContent", {"frameId": frame["id"], "url": frame["url"]})
    return base64.b64decode(loaded["content"]) if loaded.get("base64Encoded") else loaded["content"].encode("utf-8")


def check_layout(client: DevTools, screenshot_dir: Path) -> list[dict]:
    """Inspect actual desktop/mobile overflow and save screenshots of four panels."""
    checks = []
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    for label, width, height, mobile in (("desktop", 1440, 1000, False), ("mobile", 390, 844, True)):
        client.call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": width,
                "height": height,
                "deviceScaleFactor": 1,
                "mobile": mobile,
            },
        )
        time.sleep(0.15)
        layout = client.evaluate(
            "({innerWidth,clientWidth:document.documentElement.clientWidth,"
            "scrollWidth:document.documentElement.scrollWidth})"
        )
        checks.append({
            "label": f"{label} no horizontal overflow",
            "passed": layout["innerWidth"] == width and layout["scrollWidth"] <= layout["clientWidth"],
            **layout,
        })
        for anchor in ("flow-title", "analytic-title", "record-title", "telemetry-title"):
            client.evaluate(f"document.getElementById({json.dumps(anchor)}).scrollIntoView();true")
            capture = client.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
            (screenshot_dir / f"{label}-{anchor}.png").write_bytes(base64.b64decode(capture["data"]))
    return checks


def check_media(client: DevTools) -> list[dict]:
    """Load, seek, decode and actually play every declared native video element."""
    client.call(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": 1440,
            "height": 1000,
            "deviceScaleFactor": 1,
            "mobile": False,
        },
    )
    names = client.evaluate("RECORDS.map(record=>record.name)")
    results = []
    for index, name in enumerate(names):
        expression = r"""(async()=> {
          const record=RECORDS.find(row=>row.name===NAME);
          const article=[...document.querySelectorAll('.trial')].find(item=>item.dataset.trialId===record.trial_id);
          article.querySelector('.views').open=true;
          await new Promise(resolve=>setTimeout(resolve,0));
          const video=[...article.querySelectorAll('video')].find(item=>item.getAttribute('src')?.endsWith(record.mp4));
          const event=(name,action)=>new Promise((resolve,reject)=>{
            const timer=setTimeout(()=>{cleanup();reject(new Error(name+' timeout'));},15000);
            const ok=()=>{cleanup();resolve();};const bad=()=>{cleanup();reject(new Error('media error '+video.error?.code));};
            const cleanup=()=>{clearTimeout(timer);video.removeEventListener(name,ok);video.removeEventListener('error',bad);};
            video.addEventListener(name,ok,{once:true});video.addEventListener('error',bad,{once:true});action();
          });
          try {
            if(video.readyState<1)await event('loadedmetadata',()=>video.load());
            await event('seeked',()=>{video.currentTime=6.8;});
            await Promise.race([video.play(),new Promise((_,reject)=>setTimeout(()=>reject(new Error('play timeout')),15000))]);
            await new Promise(resolve=>setTimeout(resolve,60));
            const result={name:record.name,passed:!video.paused&&Math.abs(video.duration-7)<.05&&video.videoWidth===record.mp4_width&&video.videoHeight===record.mp4_height,currentTime:video.currentTime,duration:video.duration,width:video.videoWidth,height:video.videoHeight};
            article.querySelectorAll('.sync-controls button')[1].click();article.querySelector('.views').open=false;
            return result;
          }catch(error){video.pause();return{name:record.name,passed:false,error:String(error),mediaError:video.error?.code};}
        })()""".replace("NAME", json.dumps(name))
        result = client.evaluate(expression)
        results.append(result)
        if index % 9 == 0 or not result["passed"]:
            print(f"Chrome decoded/played {index+1}/{len(names)}: {name}: {result['passed']}", flush=True)
    return results


def check_general(client: DevTools, url: str, manifest_path: Path) -> list[dict]:
    """Verify the published gallery still embeds every prior and new media record."""
    manifest = json.loads(manifest_path.read_text())
    expected = [{key: record[key] for key in ("name", "gif", "mp4", "gif_sha256", "mp4_sha256")} for record in manifest]
    client.events.clear()
    client.call("Page.navigate", {"url": url})
    deadline = time.monotonic() + 30
    while not client.evaluate("document.readyState==='complete' && typeof experimentGroups!=='undefined'"):
        if time.monotonic() > deadline:
            raise RuntimeError("The general gallery did not initialize")
        time.sleep(0.1)
    actual = client.evaluate(
        "records.map(record=>Object.fromEntries(['name','gif','mp4','gif_sha256','mp4_sha256']"
        ".map(key=>[key,record[key]])))"
    )
    client.evaluate("document.querySelector('[data-category=tilting]').click();true")
    visible = client.evaluate("[...document.querySelectorAll('article')].filter(article=>!article.hidden).length")
    topic_groups = len({record["experiment_id"] for record in manifest if record.get("category") == "tilting"})
    return [
        {"label": "general gallery preserves all actual records and hashes", "passed": actual == expected},
        {
            "label": "general gallery tilt filter",
            "passed": visible == topic_groups,
            "actual": visible,
            "expected": topic_groups,
        },
    ]


def page_errors(client: DevTools) -> list[dict]:
    """Keep actual browser JavaScript and resource failures for review."""
    return [
        event
        for event in client.events
        if event["method"] == "Runtime.exceptionThrown"
        or event["method"] == "Runtime.consoleAPICalled"
        and event["params"].get("type") == "error"
        or event["method"] == "Log.entryAdded"
        and event["params"].get("entry", {}).get("level") == "error"
    ]


def main() -> int:
    """Write actual CPU browser evidence; fail if any requested check fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=(REPO_ROOT / "usage_assets/tilting/index.html").as_uri())
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/tilting-ui-check/browser.json")
    parser.add_argument("--chrome", type=Path, default=Path("/usr/bin/google-chrome"))
    parser.add_argument("--expected-records", type=int, default=126)
    parser.add_argument("--media", action="store_true")
    parser.add_argument("--general-url")
    parser.add_argument(
        "--general-manifest",
        type=Path,
        default=REPO_ROOT / "usage_assets/animations/manifest.json",
        help="Actual manifest for --general-url; set this for an independently reproduced fresh site.",
    )
    args = parser.parse_args()
    report = {"url": args.url, "checks": [], "media": [], "browser": "Existing Chrome CPU; no host dependencies"}
    process, client, log = launch_browser(args)
    try:
        report["browser_version"] = client.call("Browser.getVersion")
        document = open_topic(client, args.url)
        report["loaded_document_sha256"] = hashlib.sha256(document).hexdigest()
        count = client.evaluate("RECORDS.length")
        report["checks"].append({
            "label": "actual preview count",
            "passed": count == args.expected_records,
            "actual": count,
            "expected": args.expected_records,
        })
        report["checks"].extend(client.evaluate(UI_CHECKS))
        report["checks"].extend(check_layout(client, args.output.parent / "screenshots"))
        if args.media:
            report["media"] = check_media(client)
        report["topic_errors"] = page_errors(client)
        report["checks"].append(
            {"label": "topic has no JavaScript/resource errors", "passed": not report["topic_errors"]}
        )
        if args.general_url:
            report["general_manifest"] = str(args.general_manifest)
            report["general_manifest_sha256"] = hashlib.sha256(args.general_manifest.read_bytes()).hexdigest()
            report["checks"].extend(check_general(client, args.general_url, args.general_manifest))
            report["general_errors"] = page_errors(client)
            report["checks"].append(
                {"label": "general gallery has no JavaScript/resource errors", "passed": not report["general_errors"]}
            )
    finally:
        stop_browser(process, client, log)
    report["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "passed" if all(check["passed"] for check in report["checks"] + report["media"]) else "failed"
    report["verifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"{report['status']}: {len(report['checks'])} UI/provenance checks, {len(report['media'])} actual MP4 checks")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
