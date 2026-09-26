from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pytest
import respx
from sqlalchemy import Engine

from sentinel.agent.models import WorkItem
from sentinel.storage.db import make_engine
from sentinel.tools.builtin.lookups import MOCK_CUSTOMERS
from tests.integration.test_api import lookup_jev, make_client

UI = files("sentinel.api") / "ui"


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    return make_engine(f"sqlite:///{tmp_path / 'ui.db'}")


@respx.mock
def test_ui_page_and_examples_need_no_token(engine: Engine) -> None:
    with make_client(engine, lookup_jev()) as client:
        page = client.get("/ui")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert "Sentinel demo" in page.text

        root = client.get("/", follow_redirects=False)
        assert root.status_code == 307 and root.headers["location"] == "/ui"

        examples = client.get("/ui/examples").json()
        assert len(examples) >= 5
        # The UI's actual work still requires the token.
        assert client.post("/items", json={"id": "x", **examples[0]["item"]}).status_code == 401


def test_examples_are_valid_items_for_the_demo_accounts() -> None:
    import json

    examples = json.loads((UI / "examples.json").read_text())
    keys = [e["key"] for e in examples]
    assert len(keys) == len(set(keys))
    for ex in examples:
        assert ex["title"] and ex["expect"]
        item = WorkItem.model_validate({"id": f"ui-{ex['key']}", **ex["item"]})
        assert item.customer_id is None or item.customer_id in MOCK_CUSTOMERS


def test_page_never_renders_untrusted_text_as_html() -> None:
    page = (UI / "index.html").read_text()
    # Tickets, drafts and model output are inserted with textContent only.
    assert "innerHTML" not in page
    assert "outerHTML" not in page
    assert "insertAdjacentHTML" not in page
    assert "document.write" not in page
    assert "eval(" not in page
