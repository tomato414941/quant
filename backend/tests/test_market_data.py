import json

import pandas as pd
import pytest

from app import market_data as market_data_module
from app.market_data import (
    MarketDataRequest,
    YFinanceMarketDataProvider,
    build_market_data_content_fingerprint,
    build_dataset_snapshot_metadata,
    finalize_dataset_snapshot_metadata,
    read_market_data_snapshot,
    write_market_data_snapshot,
)


def build_download_frame(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "Close": values,
            "Volume": [1000.0] * len(values),
        },
        index=index,
    )


def test_yfinance_provider_allows_partial_failures(monkeypatch) -> None:
    provider = YFinanceMarketDataProvider()

    def fake_download(*, tickers, **kwargs):
        if tickers == "EWJ":
            raise TypeError("'Response' object is not subscriptable")
        if tickers == "SPY":
            return build_download_frame([100.0, 101.0, 102.0, 103.0])
        if tickers == "QQQ":
            return build_download_frame([200.0, 202.0, 204.0, 206.0])
        raise AssertionError(f"unexpected ticker: {tickers}")

    monkeypatch.setattr("app.market_data.yf.download", fake_download)

    bundle, metadata = provider.fetch_bundle(
        MarketDataRequest(
            tickers=("SPY", "EWJ", "QQQ"),
            period="10y",
            timeframe="1d",
        )
    )

    assert list(bundle["closes"].columns) == ["SPY", "QQQ"]
    assert metadata["tickers"] == ["SPY", "QQQ"]
    assert metadata["requested_tickers"] == ["SPY", "EWJ", "QQQ"]
    assert metadata["failed_tickers"] == {"EWJ": "download_exception:TypeError"}
    assert metadata["datasetSnapshot"]["source"] == "Yahoo Finance via yfinance"
    assert metadata["datasetSnapshot"]["requestedTickers"] == ["SPY", "EWJ", "QQQ"]
    assert metadata["datasetSnapshot"]["availableTickers"] == ["SPY", "QQQ"]


def test_yfinance_provider_raises_when_too_few_tickers_survive(monkeypatch) -> None:
    provider = YFinanceMarketDataProvider()

    def fake_download(*, tickers, **kwargs):
        if tickers == "SPY":
            return build_download_frame([100.0, 101.0, 102.0, 103.0])
        raise TypeError("network failure")

    monkeypatch.setattr("app.market_data.yf.download", fake_download)

    with pytest.raises(ValueError, match="At least two tickers"):
        provider.fetch_bundle(
            MarketDataRequest(
                tickers=("SPY", "EWJ"),
                period="10y",
                timeframe="1d",
            )
        )


def test_yfinance_provider_uses_explicit_date_range(monkeypatch) -> None:
    provider = YFinanceMarketDataProvider()
    calls = []

    def fake_download(*, tickers, **kwargs):
        calls.append({"tickers": tickers, **kwargs})
        return build_download_frame([100.0, 101.0, 102.0, 103.0])

    monkeypatch.setattr("app.market_data.yf.download", fake_download)

    _bundle, metadata = provider.fetch_bundle(
        MarketDataRequest(
            tickers=("SPY", "QQQ"),
            period="2015_2025",
            timeframe="1d",
            start_date="2015-01-01",
            end_date="2025-12-31",
        )
    )

    assert all(call["start"] == "2015-01-01" for call in calls)
    assert all(call["end"] == "2026-01-01" for call in calls)
    assert all("period" not in call for call in calls)
    assert metadata["period"] == "2015_2025"
    assert metadata["start_date"] == "2015-01-01"
    assert metadata["end_date"] == "2025-12-31"


def test_yfinance_provider_aligns_to_union_calendar(monkeypatch) -> None:
    provider = YFinanceMarketDataProvider()

    def fake_download(*, tickers, **kwargs):
        if tickers == "SPY":
            return pd.DataFrame(
                {"Close": [100.0, 101.0, 102.0], "Volume": [1000.0, 1100.0, 1200.0]},
                index=pd.to_datetime(["2025-01-03", "2025-01-10", "2025-01-17"]),
            )
        if tickers == "BTC-USD":
            return pd.DataFrame(
                {"Close": [200.0, 202.0, 204.0], "Volume": [2000.0, 2100.0, 2200.0]},
                index=pd.to_datetime(["2025-01-02", "2025-01-09", "2025-01-16"]),
            )
        raise AssertionError(f"unexpected ticker: {tickers}")

    monkeypatch.setattr("app.market_data.yf.download", fake_download)

    bundle, metadata = provider.fetch_bundle(
        MarketDataRequest(
            tickers=("SPY", "BTC-USD"),
            period="2015_2025",
            timeframe="1wk",
            start_date="2015-01-01",
            end_date="2025-12-31",
        )
    )

    assert list(bundle["closes"].index) == [
        "2025-01-02",
        "2025-01-03",
        "2025-01-09",
        "2025-01-10",
        "2025-01-16",
        "2025-01-17",
    ]
    assert list(bundle["closes"].columns) == ["SPY", "BTC-USD"]
    assert pd.isna(bundle["closes"].loc["2025-01-02", "SPY"])
    assert float(bundle["closes"].loc["2025-01-10", "BTC-USD"]) == 202.0
    assert metadata["assetAvailability"]["SPY"]["firstValidDate"] == "2025-01-03"
    assert metadata["assetAvailability"]["BTC-USD"]["firstValidDate"] == "2025-01-02"
    assert metadata["aligned_start_date"] == "2025-01-02"
    assert metadata["aligned_end_date"] == "2025-01-17"


def test_yfinance_provider_keeps_rows_before_late_asset_starts(monkeypatch) -> None:
    provider = YFinanceMarketDataProvider()

    def fake_download(*, tickers, **kwargs):
        if tickers == "SPY":
            return pd.DataFrame(
                {
                    "Close": [100.0, 101.0, 102.0, 103.0],
                    "Volume": [1000.0, 1100.0, 1200.0, 1300.0],
                },
                index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
            )
        if tickers == "QQQ":
            return pd.DataFrame(
                {
                    "Close": [200.0, 202.0, 204.0, 206.0],
                    "Volume": [2000.0, 2100.0, 2200.0, 2300.0],
                },
                index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
            )
        if tickers == "ETH-USD":
            return pd.DataFrame(
                {"Close": [300.0, 303.0], "Volume": [3000.0, 3300.0]},
                index=pd.to_datetime(["2025-01-03", "2025-01-04"]),
            )
        raise AssertionError(f"unexpected ticker: {tickers}")

    monkeypatch.setattr("app.market_data.yf.download", fake_download)

    bundle, metadata = provider.fetch_bundle(
        MarketDataRequest(
            tickers=("SPY", "QQQ", "ETH-USD"),
            period="2015_2025",
            timeframe="1d",
            start_date="2025-01-01",
            end_date="2025-01-04",
        )
    )

    assert list(bundle["closes"].index) == [
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
        "2025-01-04",
    ]
    assert pd.isna(bundle["closes"].loc["2025-01-01", "ETH-USD"])
    assert pd.isna(bundle["closes"].loc["2025-01-02", "ETH-USD"])
    assert float(bundle["closes"].loc["2025-01-03", "ETH-USD"]) == 300.0
    assert metadata["assetAvailability"]["ETH-USD"]["firstValidDate"] == "2025-01-03"
    assert metadata["aligned_start_date"] == "2025-01-01"


def test_dataset_snapshot_fingerprint_excludes_created_at_utc() -> None:
    first_snapshot = build_dataset_snapshot_metadata(
        source="toy",
        timeframe="1d",
        period="toy_period",
        start_date="2025-01-01",
        end_date="2025-01-07",
        requested_tickers=["SPY", "QQQ", "EWJ"],
        available_tickers=["SPY", "QQQ"],
        row_count=7,
        adjustment_policy="toy_adjusted",
        created_at_utc="2026-01-01T00:00:00Z",
    )
    second_snapshot = build_dataset_snapshot_metadata(
        source="toy",
        timeframe="1d",
        period="toy_period",
        start_date="2025-01-01",
        end_date="2025-01-07",
        requested_tickers=["SPY", "QQQ", "EWJ"],
        available_tickers=["SPY", "QQQ"],
        row_count=7,
        adjustment_policy="toy_adjusted",
        created_at_utc="2026-01-02T00:00:00Z",
    )

    assert first_snapshot["createdAtUtc"] != second_snapshot["createdAtUtc"]
    assert first_snapshot["fingerprint"] == second_snapshot["fingerprint"]
    assert first_snapshot["snapshotId"] == first_snapshot["fingerprint"]


def test_write_market_data_snapshot_persists_manifest_and_csv_content(tmp_path) -> None:
    snapshot = build_dataset_snapshot_metadata(
        source="toy",
        timeframe="1d",
        period="toy_period",
        start_date="2025-01-01",
        end_date="2025-01-02",
        requested_tickers=["SPY", "QQQ"],
        available_tickers=["SPY", "QQQ"],
        row_count=2,
        adjustment_policy="toy_adjusted",
        created_at_utc="2026-01-01T00:00:00Z",
    )
    closes = pd.DataFrame(
        {"SPY": [100.0, 101.0], "QQQ": [200.0, 202.0]},
        index=["2025-01-01", "2025-01-02"],
    )
    volumes = pd.DataFrame(
        {"SPY": [1000.0, 1100.0], "QQQ": [2000.0, 2200.0]},
        index=["2025-01-01", "2025-01-02"],
    )
    closes.index.name = "date"
    volumes.index.name = "date"
    finalize_dataset_snapshot_metadata(
        {"datasetSnapshot": snapshot},
        {"closes": closes, "volumes": volumes},
    )

    snapshot_path = write_market_data_snapshot(
        bundle={"closes": closes, "volumes": volumes},
        metadata={"datasetSnapshot": snapshot},
        storage_dir=tmp_path,
    )

    assert snapshot_path == tmp_path / str(snapshot["snapshotId"])

    manifest = json.loads((snapshot_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "schemaVersion": 1,
        "datasetSnapshot": snapshot,
        "files": {
            "closes": "closes.csv",
            "volumes": "volumes.csv",
        },
    }

    stored_closes = pd.read_csv(snapshot_path / "closes.csv", index_col="date")
    stored_volumes = pd.read_csv(snapshot_path / "volumes.csv", index_col="date")
    pd.testing.assert_frame_equal(stored_closes, closes)
    pd.testing.assert_frame_equal(stored_volumes, volumes)


def test_dataset_snapshot_fingerprint_includes_market_data_content() -> None:
    base_snapshot = build_dataset_snapshot_metadata(
        source="toy",
        timeframe="1d",
        period="toy_period",
        start_date="2025-01-01",
        end_date="2025-01-02",
        requested_tickers=["SPY", "QQQ"],
        available_tickers=["SPY", "QQQ"],
        row_count=2,
        adjustment_policy="toy_adjusted",
        created_at_utc="2026-01-01T00:00:00Z",
    )
    closes = pd.DataFrame(
        {"SPY": [100.0, 101.0], "QQQ": [200.0, 202.0]},
        index=["2025-01-01", "2025-01-02"],
    )
    changed_closes = pd.DataFrame(
        {"SPY": [100.0, 101.5], "QQQ": [200.0, 202.0]},
        index=["2025-01-01", "2025-01-02"],
    )
    volumes = pd.DataFrame(
        {"SPY": [1000.0, 1100.0], "QQQ": [2000.0, 2200.0]},
        index=["2025-01-01", "2025-01-02"],
    )
    changed_volumes = pd.DataFrame(
        {"SPY": [1000.0, 1100.0], "QQQ": [2000.0, 2300.0]},
        index=["2025-01-01", "2025-01-02"],
    )
    first_snapshot = dict(base_snapshot)
    second_snapshot = dict(base_snapshot)
    third_snapshot = dict(base_snapshot)

    finalize_dataset_snapshot_metadata(
        {"datasetSnapshot": first_snapshot},
        {"closes": closes, "volumes": volumes},
    )
    finalize_dataset_snapshot_metadata(
        {"datasetSnapshot": second_snapshot},
        {"closes": changed_closes, "volumes": volumes},
    )
    finalize_dataset_snapshot_metadata(
        {"datasetSnapshot": third_snapshot},
        {"closes": closes, "volumes": changed_volumes},
    )

    assert first_snapshot["contentFingerprint"] != second_snapshot["contentFingerprint"]
    assert first_snapshot["snapshotId"] != second_snapshot["snapshotId"]
    assert first_snapshot["contentFingerprint"] != third_snapshot["contentFingerprint"]
    assert first_snapshot["snapshotId"] != third_snapshot["snapshotId"]


def test_read_market_data_snapshot_loads_manifest_and_csv_content(tmp_path) -> None:
    snapshot = build_dataset_snapshot_metadata(
        source="toy",
        timeframe="1d",
        period="toy_period",
        start_date="2025-01-01",
        end_date="2025-01-02",
        requested_tickers=["SPY", "QQQ"],
        available_tickers=["SPY", "QQQ"],
        row_count=2,
        adjustment_policy="toy_adjusted",
        created_at_utc="2026-01-01T00:00:00Z",
    )
    closes = pd.DataFrame(
        {"SPY": [100.0, 101.0], "QQQ": [200.0, 202.0]},
        index=["2025-01-01", "2025-01-02"],
    )
    volumes = pd.DataFrame(
        {"SPY": [1000.0, 1100.0], "QQQ": [2000.0, 2200.0]},
        index=["2025-01-01", "2025-01-02"],
    )

    write_market_data_snapshot(
        bundle={"closes": closes, "volumes": volumes},
        metadata={"datasetSnapshot": snapshot},
        storage_dir=tmp_path,
    )

    bundle, metadata = read_market_data_snapshot(
        str(snapshot["snapshotId"]),
        storage_dir=tmp_path,
        expected_snapshot=snapshot,
    )

    pd.testing.assert_frame_equal(bundle["closes"], closes)
    pd.testing.assert_frame_equal(bundle["volumes"], volumes)
    assert metadata["datasetSnapshot"] == snapshot
    assert metadata["tickers"] == ["SPY", "QQQ"]
    assert metadata["requested_tickers"] == ["SPY", "QQQ"]
    assert metadata["period"] == "toy_period"
    assert metadata["timeframe"] == "1d"
    assert metadata["aligned_start_date"] == "2025-01-01"
    assert metadata["aligned_end_date"] == "2025-01-02"


def test_read_market_data_snapshot_rejects_fingerprint_mismatch(tmp_path) -> None:
    snapshot = build_dataset_snapshot_metadata(
        source="toy",
        timeframe="1d",
        period="toy_period",
        start_date=None,
        end_date=None,
        requested_tickers=["SPY", "QQQ"],
        available_tickers=["SPY", "QQQ"],
        row_count=2,
        adjustment_policy="toy_adjusted",
        created_at_utc="2026-01-01T00:00:00Z",
    )
    closes = pd.DataFrame(
        {"SPY": [100.0, 101.0], "QQQ": [200.0, 202.0]},
        index=["2025-01-01", "2025-01-02"],
    )
    volumes = pd.DataFrame(
        {"SPY": [1000.0, 1100.0], "QQQ": [2000.0, 2200.0]},
        index=["2025-01-01", "2025-01-02"],
    )
    snapshot_path = write_market_data_snapshot(
        bundle={"closes": closes, "volumes": volumes},
        metadata={"datasetSnapshot": snapshot},
        storage_dir=tmp_path,
    )
    manifest_path = snapshot_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["datasetSnapshot"]["rowCount"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        read_market_data_snapshot(str(snapshot["snapshotId"]), storage_dir=tmp_path)


def test_read_market_data_snapshot_rejects_csv_content_mismatch(tmp_path) -> None:
    snapshot = build_dataset_snapshot_metadata(
        source="toy",
        timeframe="1d",
        period="toy_period",
        start_date=None,
        end_date=None,
        requested_tickers=["SPY", "QQQ"],
        available_tickers=["SPY", "QQQ"],
        row_count=2,
        adjustment_policy="toy_adjusted",
        created_at_utc="2026-01-01T00:00:00Z",
    )
    closes = pd.DataFrame(
        {"SPY": [100.0, 101.0], "QQQ": [200.0, 202.0]},
        index=["2025-01-01", "2025-01-02"],
    )
    volumes = pd.DataFrame(
        {"SPY": [1000.0, 1100.0], "QQQ": [2000.0, 2200.0]},
        index=["2025-01-01", "2025-01-02"],
    )
    snapshot_path = write_market_data_snapshot(
        bundle={"closes": closes, "volumes": volumes},
        metadata={"datasetSnapshot": snapshot},
        storage_dir=tmp_path,
    )
    changed_closes = closes.copy()
    changed_closes.loc["2025-01-02", "SPY"] = 999.0
    changed_closes.to_csv(snapshot_path / "closes.csv", index_label="date")

    with pytest.raises(ValueError, match="contentFingerprint"):
        read_market_data_snapshot(str(snapshot["snapshotId"]), storage_dir=tmp_path)


def test_write_market_data_snapshot_is_idempotent_for_same_content(tmp_path) -> None:
    snapshot = build_dataset_snapshot_metadata(
        source="toy",
        timeframe="1d",
        period="toy_period",
        start_date=None,
        end_date=None,
        requested_tickers=["SPY", "QQQ"],
        available_tickers=["SPY", "QQQ"],
        row_count=2,
        adjustment_policy="toy_adjusted",
        created_at_utc="2026-01-01T00:00:00Z",
    )
    closes = pd.DataFrame({"SPY": [100.0, 101.0], "QQQ": [200.0, 202.0]})
    volumes = pd.DataFrame({"SPY": [1000.0, 1100.0], "QQQ": [2000.0, 2200.0]})
    closes.index.name = "date"
    volumes.index.name = "date"
    bundle = {"closes": closes, "volumes": volumes}
    metadata = {"datasetSnapshot": snapshot}

    first_path = write_market_data_snapshot(bundle, metadata, storage_dir=tmp_path)
    second_path = write_market_data_snapshot(bundle, metadata, storage_dir=tmp_path)

    assert first_path == second_path
    assert build_market_data_content_fingerprint(bundle) == snapshot["contentFingerprint"]


def test_write_market_data_snapshot_does_not_publish_partial_directory_on_failure(
    tmp_path,
    monkeypatch,
) -> None:
    snapshot = build_dataset_snapshot_metadata(
        source="toy",
        timeframe="1d",
        period="toy_period",
        start_date=None,
        end_date=None,
        requested_tickers=["SPY", "QQQ"],
        available_tickers=["SPY", "QQQ"],
        row_count=2,
        adjustment_policy="toy_adjusted",
        created_at_utc="2026-01-01T00:00:00Z",
    )
    closes = pd.DataFrame({"SPY": [100.0, 101.0], "QQQ": [200.0, 202.0]})
    volumes = pd.DataFrame({"SPY": [1000.0, 1100.0], "QQQ": [2000.0, 2200.0]})
    closes.index.name = "date"
    volumes.index.name = "date"
    bundle = {"closes": closes, "volumes": volumes}
    metadata = {"datasetSnapshot": snapshot}
    finalize_dataset_snapshot_metadata(metadata, bundle)
    snapshot_id = str(snapshot["snapshotId"])
    original_to_csv = pd.DataFrame.to_csv

    def fail_on_volumes(frame, path_or_buf=None, *args, **kwargs):
        if frame is volumes:
            raise OSError("simulated write failure")
        return original_to_csv(frame, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_on_volumes)

    with pytest.raises(OSError, match="simulated write failure"):
        write_market_data_snapshot(bundle, metadata, storage_dir=tmp_path)

    assert not (tmp_path / snapshot_id).exists()


def test_write_market_data_snapshot_rejects_existing_snapshot_with_mismatched_content(
    tmp_path,
) -> None:
    snapshot = build_dataset_snapshot_metadata(
        source="toy",
        timeframe="1d",
        period="toy_period",
        start_date=None,
        end_date=None,
        requested_tickers=["SPY", "QQQ"],
        available_tickers=["SPY", "QQQ"],
        row_count=2,
        adjustment_policy="toy_adjusted",
        created_at_utc="2026-01-01T00:00:00Z",
    )
    closes = pd.DataFrame({"SPY": [100.0, 101.0], "QQQ": [200.0, 202.0]})
    volumes = pd.DataFrame({"SPY": [1000.0, 1100.0], "QQQ": [2000.0, 2200.0]})
    bundle = {"closes": closes, "volumes": volumes}
    metadata = {"datasetSnapshot": snapshot}
    snapshot_path = write_market_data_snapshot(bundle, metadata, storage_dir=tmp_path)
    changed_closes = closes.copy()
    changed_closes.loc[1, "SPY"] = 999.0
    changed_closes.to_csv(snapshot_path / "closes.csv", index_label="date")

    with pytest.raises(ValueError, match="contentFingerprint"):
        write_market_data_snapshot(bundle, metadata, storage_dir=tmp_path)


def test_write_market_data_snapshot_replaces_manifestless_partial_directory(
    tmp_path,
) -> None:
    snapshot = build_dataset_snapshot_metadata(
        source="toy",
        timeframe="1d",
        period="toy_period",
        start_date=None,
        end_date=None,
        requested_tickers=["SPY", "QQQ"],
        available_tickers=["SPY", "QQQ"],
        row_count=2,
        adjustment_policy="toy_adjusted",
        created_at_utc="2026-01-01T00:00:00Z",
    )
    closes = pd.DataFrame({"SPY": [100.0, 101.0], "QQQ": [200.0, 202.0]})
    volumes = pd.DataFrame({"SPY": [1000.0, 1100.0], "QQQ": [2000.0, 2200.0]})
    bundle = {"closes": closes, "volumes": volumes}
    metadata = {"datasetSnapshot": snapshot}
    finalize_dataset_snapshot_metadata(metadata, bundle)
    partial_path = tmp_path / str(snapshot["snapshotId"])
    partial_path.mkdir(parents=True)
    (partial_path / "closes.csv").write_text("date,SPY,QQQ\n2025-01-01,0,0\n", encoding="utf-8")

    snapshot_path = write_market_data_snapshot(bundle, metadata, storage_dir=tmp_path)

    assert snapshot_path == partial_path
    assert (snapshot_path / "manifest.json").exists()
    stored_closes = pd.read_csv(snapshot_path / "closes.csv", index_col="date")
    pd.testing.assert_frame_equal(stored_closes, closes, check_names=False)


def test_write_market_data_snapshot_reuses_concurrently_published_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    snapshot = build_dataset_snapshot_metadata(
        source="toy",
        timeframe="1d",
        period="toy_period",
        start_date=None,
        end_date=None,
        requested_tickers=["SPY", "QQQ"],
        available_tickers=["SPY", "QQQ"],
        row_count=2,
        adjustment_policy="toy_adjusted",
        created_at_utc="2026-01-01T00:00:00Z",
    )
    closes = pd.DataFrame({"SPY": [100.0, 101.0], "QQQ": [200.0, 202.0]})
    volumes = pd.DataFrame({"SPY": [1000.0, 1100.0], "QQQ": [2000.0, 2200.0]})
    bundle = {"closes": closes, "volumes": volumes}
    metadata = {"datasetSnapshot": snapshot}
    finalize_dataset_snapshot_metadata(metadata, bundle)
    snapshot_id = str(snapshot["snapshotId"])
    snapshot_path = tmp_path / snapshot_id
    original_replace = type(snapshot_path).replace

    def publish_existing_before_replace(path, target):
        if path.name.endswith(".tmp") and target == snapshot_path:
            snapshot_path.mkdir()
            market_data_module.write_market_data_csv(snapshot_path / "closes.csv", closes)
            market_data_module.write_market_data_csv(snapshot_path / "volumes.csv", volumes)
            market_data_module.write_market_data_manifest(
                snapshot_path / "manifest.json",
                {
                    "schemaVersion": 1,
                    "datasetSnapshot": snapshot,
                    "files": {"closes": "closes.csv", "volumes": "volumes.csv"},
                },
            )
            raise OSError("simulated concurrent publish")
        return original_replace(path, target)

    monkeypatch.setattr(type(snapshot_path), "replace", publish_existing_before_replace)

    assert write_market_data_snapshot(bundle, metadata, storage_dir=tmp_path) == snapshot_path
    assert (snapshot_path / "manifest.json").exists()
    assert not list(tmp_path.glob(f".{snapshot_id}.*.tmp"))
