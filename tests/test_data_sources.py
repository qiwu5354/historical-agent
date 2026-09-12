"""
数据源采集回归测试。

覆盖曾经导致「共 0 段资料、来源分布全 0」的那条链路：
  1. web_fetcher 用 httpx ≥0.28 支持的参数建客户端（旧代码传 `proxies=` 直接 TypeError）；
  2. 代理地址规范化（允许只写 host:port）与「配置了代理就必须用上」；
  3. 传输类错误重试、HTTP 错误不重试、非预期异常不被静默吞掉；
  4. wikipediaapi 的 kwargs 只包含它认识的参数（旧代码传 `proxies=` 同样会炸）；
  5. 维基 REST 兜底与 Bing RSS 兜底在解析层可用。

不依赖网络：httpx 全部用 MockTransport 打桩。
运行：python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from core import search_engine, web_fetcher  # noqa: E402
from core.web_fetcher import (  # noqa: E402
    _normalize_proxy,
    fetch_url,
    http_client_config,
    proxy_hint,
    reset_cache,
)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """每个用例前清理 URL 缓存与代理配置。"""
    reset_cache()
    monkeypatch.setattr(settings.search, "proxy", "", raising=False)
    monkeypatch.setattr(settings.search, "fetch_retries", 0, raising=False)
    yield
    reset_cache()


# ==================== 1. 客户端参数（核心回归） ====================

def test_client_config_never_passes_removed_proxies_param():
    cfg = http_client_config()
    assert "proxies" not in cfg
    # 必须能被当前 httpx 版本接受（旧代码在这里 TypeError）
    client = httpx.Client(**cfg)
    try:
        assert client is not None
    finally:
        client.close()


def test_client_config_uses_singular_proxy_when_configured(monkeypatch):
    monkeypatch.setattr(settings.search, "proxy", "http://127.0.0.1:7890", raising=False)
    cfg = http_client_config()
    assert cfg["proxy"] == "http://127.0.0.1:7890"
    assert "proxies" not in cfg
    client = httpx.Client(**cfg)
    client.close()


def test_client_config_omits_proxy_when_empty(monkeypatch):
    monkeypatch.setattr(settings.search, "proxy", "", raising=False)
    cfg = http_client_config()
    assert "proxy" not in cfg
    # trust_env=True 让环境里的 HTTP(S)_PROXY 仍能生效
    assert cfg["trust_env"] is True


def test_client_config_sets_browser_like_headers():
    cfg = http_client_config()
    assert "Mozilla" in cfg["headers"]["User-Agent"]
    assert "Accept-Language" in cfg["headers"]


# ==================== 2. 代理规范化 ====================

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("127.0.0.1:7890", "http://127.0.0.1:7890"),
        ("http://127.0.0.1:7890", "http://127.0.0.1:7890"),
        ("socks5://127.0.0.1:1080", "socks5://127.0.0.1:1080"),
        ("  http://127.0.0.1:7890  ", "http://127.0.0.1:7890"),
        ("", None),
        ("   ", None),
        (None, None),
    ],
)
def test_normalize_proxy(raw, expected):
    assert _normalize_proxy(raw) == expected


def test_proxy_hint_mentions_configured_proxy(monkeypatch):
    monkeypatch.setattr(settings.search, "proxy", "127.0.0.1:7890", raising=False)
    hint = proxy_hint()
    assert "http://127.0.0.1:7890" in hint
    assert "未启动" in hint


def test_proxy_hint_when_unset():
    assert "未配置代理" in proxy_hint()


# ==================== 3. fetch_url 行为 ====================

def _patch_transport(monkeypatch, handler):
    """把所有 httpx.Client 实例替换为带 MockTransport 的客户端。"""
    real_client = httpx.Client

    def factory(**kwargs):
        kwargs.pop("transport", None)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(web_fetcher.httpx, "Client", factory)


def test_fetch_url_extracts_article_text(monkeypatch):
    html = (
        "<html><head><title>马克思 - 维基百科</title></head><body>"
        "<nav>导航</nav><article>"
        + "".join(f"<p>这是关于马克思的第 {i} 段正文，用于测试正文抽取逻辑。</p>" for i in range(12))
        + "</article><footer>页脚</footer></body></html>"
    )
    _patch_transport(monkeypatch, lambda request: httpx.Response(200, html=html,
                                                                headers={"content-type": "text/html"}))

    page = fetch_url("https://example.com/marx")
    assert page is not None
    assert "马克思" in page.title
    assert "第 1 段正文" in page.text
    assert "导航" not in page.text  # nav 被剔除
    assert "页脚" not in page.text  # footer 被剔除


def test_fetch_url_retries_transport_errors_then_succeeds(monkeypatch):
    monkeypatch.setattr(settings.search, "fetch_retries", 1, raising=False)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, html="<html><body><article>" + "<p>正文内容。</p>" * 40 + "</article></body></html>",
                              headers={"content-type": "text/html"})

    _patch_transport(monkeypatch, handler)
    monkeypatch.setattr(web_fetcher.time, "sleep", lambda *_: None)

    page = fetch_url("https://example.com/retry")
    assert page is not None
    assert calls["n"] == 2


def test_fetch_url_does_not_retry_http_status(monkeypatch):
    monkeypatch.setattr(settings.search, "fetch_retries", 3, raising=False)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404, html="<html><body>gone</body></html>",
                              headers={"content-type": "text/html"})

    _patch_transport(monkeypatch, handler)
    assert fetch_url("https://example.com/404") is None
    assert calls["n"] == 1  # 404 是明确答复，不重试


def test_fetch_url_logs_unexpected_errors_instead_of_swallowing(monkeypatch, caplog):
    """旧代码只捕获 httpx.HTTPError，TypeError 这类错误会逃逸并搞垮整条链路。"""
    def factory(**kwargs):
        raise TypeError("Client.__init__() got an unexpected keyword argument 'proxies'")

    monkeypatch.setattr(web_fetcher.httpx, "Client", factory)
    monkeypatch.setattr(settings.search, "fetch_retries", 0, raising=False)

    with caplog.at_level("WARNING"):
        assert fetch_url("https://example.com/boom") is None
    assert any("非预期错误" in r.message for r in caplog.records)


def test_fetch_url_returns_none_for_non_http_scheme():
    assert fetch_url("ftp://example.com") is None
    assert fetch_url("") is None


def test_fetch_url_discards_too_short_pages(monkeypatch):
    _patch_transport(monkeypatch, lambda request: httpx.Response(
        200, html="<html><body><article><p>太短</p></article></body></html>",
        headers={"content-type": "text/html"}))
    assert fetch_url("https://example.com/short") is None


def test_fetch_url_uses_cache_on_second_call(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, html="<html><body><article>" + "<p>缓存正文。</p>" * 40 + "</article></body></html>",
                              headers={"content-type": "text/html"})

    _patch_transport(monkeypatch, handler)
    first = fetch_url("https://example.com/cache")
    second = fetch_url("https://example.com/cache")
    assert first is not None and second is not None
    assert calls["n"] == 1


# ==================== 4. wikipediaapi kwargs ====================

def test_wiki_kwargs_are_accepted_by_httpx(monkeypatch):
    """旧代码往这里塞 `proxies={...}`，会直接抛 TypeError。"""
    import wikipediaapi

    monkeypatch.setattr(settings.search, "proxy", "http://127.0.0.1:7890", raising=False)
    kwargs = search_engine._wiki_kwargs()
    assert "proxies" not in kwargs
    assert kwargs["proxy"] == "http://127.0.0.1:7890"
    assert kwargs["max_retries"] == settings.search.wiki_max_retries

    # 真的构造一次，确保参数被接受（不发起请求）
    wiki = wikipediaapi.Wikipedia(
        language="zh",
        user_agent=search_engine._WIKI_USER_AGENT,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        **kwargs,
    )
    assert wiki.language == "zh"


def test_wiki_kwargs_without_proxy(monkeypatch):
    monkeypatch.setattr(settings.search, "proxy", "", raising=False)
    kwargs = search_engine._wiki_kwargs()
    assert "proxy" not in kwargs
    assert "proxies" not in kwargs


# ==================== 5. 兜底检索源 ====================

def test_bing_rss_fallback_parses_items(monkeypatch):
    rss = """<?xml version="1.0"?><rss version="2.0"><channel>
      <item><title>马克思 - 维基百科</title>
        <link>https://zh.wikipedia.org/wiki/马克思</link>
        <description>卡尔·马克思是德国哲学家。</description></item>
      <item><title>马克思 - 百度百科</title>
        <link>https://baike.baidu.com/item/马克思</link>
        <description>马克思主义创始人。</description></item>
    </channel></rss>"""
    real_client = httpx.Client

    def factory(**kwargs):
        kwargs.pop("transport", None)
        return real_client(
            transport=httpx.MockTransport(lambda request: httpx.Response(
                200, content=rss.encode("utf-8"), headers={"content-type": "application/rss+xml"})),
            **kwargs,
        )

    monkeypatch.setattr(search_engine.httpx, "Client", factory)
    rows = search_engine._bing_rss_search_urls("马克思", max_results=5)
    assert len(rows) == 2
    assert rows[0][0].startswith("https://zh.wikipedia.org")
    assert "德国哲学家" in rows[0][2]


def test_web_search_uses_fallback_when_ddg_empty(monkeypatch):
    monkeypatch.setattr(search_engine, "_ddg_search_urls", lambda q, max_results: [])
    monkeypatch.setattr(
        search_engine, "_bing_rss_search_urls",
        lambda q, max_results: [("https://example.com/x", "标题", "摘要")],
    )
    rows = search_engine._web_search_urls("马克思", max_results=3)
    assert rows == [("https://example.com/x", "标题", "摘要")]


def test_web_search_prefers_ddg_when_available(monkeypatch):
    monkeypatch.setattr(
        search_engine, "_ddg_search_urls",
        lambda q, max_results: [("https://ddg.example/a", "A", "a")],
    )
    monkeypatch.setattr(
        search_engine, "_bing_rss_search_urls",
        lambda q, max_results: pytest.fail("DuckDuckGo 有结果时不应触发兜底"),
    )
    assert search_engine._web_search_urls("x", max_results=3)[0][0].endswith("/a")


def test_wiki_rest_summary_parses_payload(monkeypatch):
    payload = {
        "title": "卡尔·马克思",
        "description": "德国哲学家",
        "extract": "卡尔·马克思是德国哲学家、经济学家。" * 5,
        "content_urls": {"desktop": {"page": "https://zh.wikipedia.org/wiki/卡尔·马克思"}},
    }
    real_client = httpx.Client

    def factory(**kwargs):
        kwargs.pop("transport", None)
        return real_client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
            **kwargs,
        )

    monkeypatch.setattr(search_engine.httpx, "Client", factory)
    snippets = search_engine._wiki_rest_summary(
        "zh", "马克思", category="wiki", source_detail="Wikipedia", character_name="马克思",
    )
    assert len(snippets) == 1
    assert snippets[0].title == "卡尔·马克思"
    assert "REST" in snippets[0].source_detail
    assert snippets[0].url.endswith("卡尔·马克思")


def test_wiki_rest_summary_swallows_failures(monkeypatch):
    def factory(**kwargs):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(search_engine.httpx, "Client", factory)
    assert search_engine._wiki_rest_summary(
        "zh", "马克思", category="wiki", source_detail="Wikipedia", character_name="马克思",
    ) == []


def test_health_report_flags_failures():
    from core.data_collector import format_health_report

    report = format_health_report([
        ("预置著作库", True, "ok"),
        ("维基百科", False, "不可达"),
    ])
    assert "✅ 预置著作库" in report
    assert "❌ 维基百科" in report
    assert "HTTPS_PROXY" in report  # 报告里必须给出可操作的代理提示


def test_health_report_all_ok():
    from core.data_collector import format_health_report

    report = format_health_report([("网页抓取", True, "成功")])
    assert "所有数据源均可用" in report


# ==================== 6. DuckDuckGo 客户端适配 ====================

def test_ddg_proxy_passes_http_proxy(monkeypatch):
    monkeypatch.setattr(settings.search, "proxy", "127.0.0.1:7890", raising=False)
    assert search_engine._ddg_proxy() == "http://127.0.0.1:7890"


def test_ddg_proxy_skips_socks(monkeypatch, caplog):
    """DDGS 不支持 SOCKS；必须跳过而不是抛异常让整条链路失败。"""
    monkeypatch.setattr(settings.search, "proxy", "socks5://127.0.0.1:1080", raising=False)
    monkeypatch.setattr(search_engine, "_socks_warned", False, raising=False)
    with caplog.at_level("WARNING"):
        assert search_engine._ddg_proxy() is None
    assert any("SOCKS" in r.message for r in caplog.records)


def test_ddg_proxy_none_when_unset(monkeypatch):
    monkeypatch.setattr(settings.search, "proxy", "", raising=False)
    assert search_engine._ddg_proxy() is None


def test_import_ddgs_prefers_new_package():
    DDGS = search_engine._import_ddgs()
    if DDGS is None:
        pytest.skip("既未安装 ddgs 也未安装 duckduckgo_search")
    assert callable(DDGS)
