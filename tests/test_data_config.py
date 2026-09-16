import json
from datetime import date

import httpx
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from alex_quant.cli import main
from alex_quant.config import Config, load_config
from alex_quant.data.alpaca import AlpacaProvider
from alex_quant.data.base import DataError, trading_sessions, validate_bars


@pytest.mark.parametrize(
    "fault",
    ["duplicate", "unsorted", "missing", "missing_session", "nan", "price", "volume", "ohlc"],
)
def test_market_data_validation_rejects_bad_input(bars, sessions, config, fault):
    bad = bars.copy()
    if fault == "duplicate":
        bad = pd.concat([bad, bad.iloc[:1]]).sort_index()
    elif fault == "unsorted":
        bad = bad.iloc[::-1]
    elif fault == "missing":
        bad = bad.iloc[1:]
    elif fault == "missing_session":
        bad = bad.drop(sessions[10], level="date")
    elif fault == "nan":
        bad.iloc[0, bad.columns.get_loc("close")] = np.nan
    elif fault == "price":
        bad.iloc[0, bad.columns.get_loc("close")] = -1
    elif fault == "volume":
        bad.iloc[0, bad.columns.get_loc("volume")] = 0
    elif fault == "ohlc":
        bad.iloc[0, bad.columns.get_loc("high")] = 0.01
    with pytest.raises(DataError):
        validate_bars(bad, config.symbols, sessions)


def test_holiday_range_and_valid_grid(bars, config, sessions):
    assert len(validate_bars(bars, config.symbols, sessions)) == len(bars)
    holidays = trading_sessions(date(2024, 1, 1), date(2024, 1, 7))
    assert holidays[0] == pd.Timestamp("2024-01-02")
    assert holidays[-1] == pd.Timestamp("2024-01-05")


def test_missing_credentials_fail_clearly_without_artifacts(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr("alex_quant.cli.load_dotenv", lambda *args, **kwargs: None)
    output = tmp_path / "absent"
    assert main(["run", "--output", str(output)]) == 2
    message = capsys.readouterr().err
    assert "ALPACA_API_KEY" in message and "ALPACA_SECRET_KEY" in message
    assert ".env.example" in message
    assert not output.exists()


def mock_credentials(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "unit-test-not-a-real-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "unit-test-not-a-real-secret")


def test_alpaca_paginated_multisymbol_dates_and_adjustment(monkeypatch):
    mock_credentials(monkeypatch)
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.params["adjustment"] == "all"
        assert request.url.params["asof"] == "2024-01-02"
        assert request.url.params["feed"] == "sip"
        assert request.headers["APCA-API-KEY-ID"] == "unit-test-not-a-real-key"
        token = request.url.params.get("page_token")
        symbol = "MSFT" if token else "AAPL"
        bar = {"t": "2024-01-02T05:00:00Z", "o": 100, "h": 102, "l": 99, "c": 101, "v": 1000}
        return httpx.Response(
            200, json={"bars": {symbol: [bar]}, "next_page_token": None if token else "p2"}
        )

    provider = AlpacaProvider(transport=httpx.MockTransport(handler))
    frame = provider.fetch(["MSFT", "AAPL"], date(2024, 1, 2), date(2024, 1, 2))
    assert len(requests) == 2
    assert requests[1].url.params["page_token"] == "p2"
    assert frame.index.to_list() == [
        (pd.Timestamp("2024-01-02"), "AAPL"),
        (pd.Timestamp("2024-01-02"), "MSFT"),
    ]
    assert "unit-test" not in json.dumps(provider.provenance)


@pytest.mark.parametrize("status", [401, 403, 422])
def test_alpaca_http_errors_are_sanitized(monkeypatch, status):
    mock_credentials(monkeypatch)
    provider = AlpacaProvider(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, text="unit-test-not-a-real-secret")
        )
    )
    with pytest.raises(DataError, match=str(status)) as error:
        provider.fetch(["SPY"], date(2024, 1, 2), date(2024, 1, 3))
    assert "unit-test" not in str(error.value)


def test_alpaca_retries_rate_limit_without_changing_feed(monkeypatch):
    mock_credentials(monkeypatch)
    monkeypatch.setattr("alex_quant.data.alpaca.time.sleep", lambda delay: None)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429)

    provider = AlpacaProvider(transport=httpx.MockTransport(handler))
    with pytest.raises(DataError, match="429"):
        provider.fetch(["SPY"], date(2024, 1, 2), date(2024, 1, 3))
    assert len(calls) == 3
    assert all(request.url.params["feed"] == "sip" for request in calls)


def test_alpaca_respects_new_york_daylight_saving_time(monkeypatch):
    mock_credentials(monkeypatch)

    def handler(request):
        assert request.url.params["start"] == "2024-07-01T00:00:00-04:00"
        assert request.url.params["end"] == "2024-07-02T23:59:59-04:00"
        return httpx.Response(
            200,
            json={
                "bars": {
                    "SPY": [
                        {
                            "t": "2024-07-01T04:00:00Z",
                            "o": 100,
                            "h": 102,
                            "l": 99,
                            "c": 101,
                            "v": 1000,
                        }
                    ]
                }
            },
        )

    provider = AlpacaProvider(transport=httpx.MockTransport(handler))
    bars = provider.fetch(["SPY"], date(2024, 7, 1), date(2024, 7, 2))
    assert bars.index[0][0] == pd.Timestamp("2024-07-01")


def test_configuration_is_validated_and_universe_overrides_replace(config, tmp_path):
    invalid = config.model_dump(mode="json")
    invalid["model"]["typo"] = 1
    with pytest.raises(ValidationError):
        Config.model_validate(invalid)
    parent = tmp_path / "parent.toml"
    parent.write_text(
        '[data]\nstart="2023-01-01"\nend="2024-01-01"\n[universe]\nA="XLK"\nB="XLK"\n[research]\ntop_k=1\n'
    )
    child = tmp_path / "child.toml"
    child.write_text('extends="parent.toml"\n[universe]\nC="XLE"\n')
    assert load_config(child).universe == {"C": "XLE"}
    parent.write_text('extends="child.toml"\n')
    with pytest.raises(ValueError, match="circular"):
        load_config(child)
