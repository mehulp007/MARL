"""Static consistency checks for the browser interface (it has no build step and no JS test runner).

The interface itself was verified by hand in a real browser; these tests only
catch the cheap mistakes: a broken import, an element id that no longer exists, a file the
server does not serve, or wording the measured results do not support.
"""
import re
from pathlib import Path

from fastapi.testclient import TestClient

from server.app import create_app

WEB = Path("web")
JS = sorted((WEB / "js").glob("*.js"))


def test_every_module_import_resolves():
    assert {p.name for p in JS} >= {"main.js", "ws.js", "interp.js", "render.js", "charts.js", "corridor.js", "theme.js"}
    for path in JS:
        for target in re.findall(r'from\s+"(\./[^"]+)"', path.read_text(encoding="utf-8")):
            assert (path.parent / target).is_file(), f"{path.name} imports missing {target}"


def test_every_element_id_used_by_the_script_exists_in_the_page():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    ids = set(re.findall(r'id="([\w-]+)"', html))
    used = set()
    for path in JS:
        used |= set(re.findall(r'\$\("([\w-]+)"\)', path.read_text(encoding="utf-8")))
    assert used, "expected main.js to look elements up by id"
    assert used <= ids, f"ids used in JS but missing from index.html: {sorted(used - ids)}"
    # ids built from a side name: ctrlA / ctrlB, miniA / miniB
    assert {"ctrlA", "ctrlB", "miniA", "miniB", "cityA", "cityB", "corrA", "corrB"} <= ids
    assert 'src="js/main.js"' in html and 'href="style.css"' in html and (WEB / "style.css").is_file()


def test_wording_follows_the_measured_results():
    """'Green Wave' may name the score, never a preset or a claim: the evaluation did not show one."""
    text = (WEB / "index.html").read_text(encoding="utf-8") + (WEB / "js" / "main.js").read_text(encoding="utf-8")
    assert not re.search(r"Green Wave(?! Score)", text)
    assert "Cooperative (β=0.4)" in text and "Selfish (β=0)" in text and "Untrained (chaos)" in text
    assert "did <b>not</b> beat the selfish one" in text and "No green wave" in text
    assert "do not recover faster" in text


def test_the_server_serves_the_interface():
    with TestClient(create_app(seed=3)) as client:
        page = client.get("/")
        assert page.status_code == 200 and "Learning timeline" in page.text and "has not been built yet" not in page.text
        for path in ["/style.css"] + [f"/js/{p.name}" for p in JS]:
            r = client.get(path)
            assert r.status_code == 200, path
        assert "javascript" in client.get("/js/main.js").headers["content-type"]
        assert "css" in client.get("/style.css").headers["content-type"]
