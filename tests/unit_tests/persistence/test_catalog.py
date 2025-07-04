# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

import datetime
import sys

import pandas as pd
import pyarrow.dataset as ds
import pytest

from nautilus_trader import TEST_DATA_DIR
from nautilus_trader.adapters.betfair.constants import BETFAIR_PRICE_PRECISION
from nautilus_trader.adapters.databento.loaders import DatabentoDataLoader
from nautilus_trader.core import nautilus_pyo3
from nautilus_trader.core.data import Data
from nautilus_trader.core.rust.model import AggressorSide
from nautilus_trader.core.rust.model import BookAction
from nautilus_trader.model.custom import customdataclass
from nautilus_trader.model.data import CustomData
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.identifiers import TradeId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import BettingInstrument
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.persistence.wranglers_v2 import QuoteTickDataWranglerV2
from nautilus_trader.persistence.wranglers_v2 import TradeTickDataWranglerV2
from nautilus_trader.test_kit.mocks.data import NewsEventData
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.test_kit.rust.data_pyo3 import TestDataProviderPyo3
from nautilus_trader.test_kit.stubs.data import TestDataStubs
from nautilus_trader.test_kit.stubs.persistence import TestPersistenceStubs


def test_list_data_types(catalog_betfair: ParquetDataCatalog) -> None:
    data_types = catalog_betfair.list_data_types()
    expected = [
        "betting_instrument",
        "custom_betfair_sequence_completed",
        "custom_betfair_ticker",
        "instrument_status",
        "order_book_delta",
        "trade_tick",
    ]
    assert data_types == expected


def test_catalog_query_filtered(
    catalog_betfair: ParquetDataCatalog,
) -> None:
    trades = catalog_betfair.trade_ticks()
    assert len(trades) == 283

    trades = catalog_betfair.trade_ticks(start="2019-12-20 20:56:18")
    assert len(trades) == 121

    trades = catalog_betfair.trade_ticks(start=1576875378384999936)
    assert len(trades) == 121

    trades = catalog_betfair.trade_ticks(start=datetime.datetime(2019, 12, 20, 20, 56, 18))
    assert len(trades) == 121

    deltas = catalog_betfair.order_book_deltas()
    assert len(deltas) == 2384

    deltas = catalog_betfair.order_book_deltas(batched=True)
    assert len(deltas) == 2007


def test_catalog_query_custom_filtered(
    catalog_betfair: ParquetDataCatalog,
) -> None:
    filtered_deltas = catalog_betfair.order_book_deltas(
        where=f"action = '{BookAction.DELETE.value}'",
    )
    assert len(filtered_deltas) == 351


def test_catalog_instruments_df(
    catalog_betfair: ParquetDataCatalog,
) -> None:
    instruments = catalog_betfair.instruments()
    assert len(instruments) == 2


def test_catalog_instruments_filtered_df(
    catalog_betfair: ParquetDataCatalog,
) -> None:
    instrument_id = catalog_betfair.instruments()[0].id.value
    instruments = catalog_betfair.instruments(instrument_ids=[instrument_id])
    assert len(instruments) == 1
    assert all(isinstance(ins, BettingInstrument) for ins in instruments)
    assert instruments[0].id.value == instrument_id


@pytest.mark.skipif(sys.platform == "win32", reason="Failing on windows")
def test_catalog_currency_with_null_max_price_loads(
    catalog_betfair: ParquetDataCatalog,
) -> None:
    # Arrange
    instrument = TestInstrumentProvider.default_fx_ccy("AUD/USD", venue=Venue("SIM"))
    catalog_betfair.write_data([instrument])

    # Act
    instrument = catalog_betfair.instruments(instrument_ids=["AUD/USD.SIM"])[0]

    # Assert
    assert instrument.max_price is None


def test_catalog_instrument_ids_correctly_unmapped(catalog: ParquetDataCatalog) -> None:
    # Arrange
    instrument = TestInstrumentProvider.default_fx_ccy("AUD/USD", venue=Venue("SIM"))
    trade_tick = TradeTick(
        instrument_id=instrument.id,
        price=Price.from_str("2.0"),
        size=Quantity.from_int(10),
        aggressor_side=AggressorSide.NO_AGGRESSOR,
        trade_id=TradeId("1"),
        ts_event=0,
        ts_init=0,
    )
    catalog.write_data([instrument, trade_tick])

    # Act
    catalog.instruments()
    instrument = catalog.instruments(instrument_ids=["AUD/USD.SIM"])[0]
    trade_tick = catalog.trade_ticks(instrument_ids=["AUD/USD.SIM"])[0]

    # Assert
    assert instrument.id.value == "AUD/USD.SIM"
    assert trade_tick.instrument_id.value == "AUD/USD.SIM"


@pytest.mark.skip("development_only")
def test_catalog_with_databento_instruments(catalog: ParquetDataCatalog) -> None:
    # Arrange
    loader = DatabentoDataLoader()
    path = TEST_DATA_DIR / "databento" / "temp" / "glbx-mdp3-20241020.definition.dbn.zst"
    instruments = loader.from_dbn_file(path, as_legacy_cython=True)
    catalog.write_data(instruments)

    # Act
    catalog.instruments()

    # Assert
    assert len(instruments) == 601_633


def test_catalog_filter(
    catalog_betfair: ParquetDataCatalog,
) -> None:
    # Arrange
    deltas = catalog_betfair.order_book_deltas()

    # Act
    filtered_deltas = catalog_betfair.order_book_deltas(
        where=f"Action = {BookAction.DELETE.value}",
    )

    # Assert
    assert len(deltas) == 2384
    assert len(filtered_deltas) == 351


def test_catalog_orderbook_deltas_precision(
    catalog_betfair: ParquetDataCatalog,
) -> None:
    # Arrange, Act
    deltas = catalog_betfair.order_book_deltas()

    # Assert
    for delta in deltas:
        assert delta.order.price.precision == BETFAIR_PRICE_PRECISION

    assert len(deltas) == 2384


def test_catalog_custom_data(catalog: ParquetDataCatalog) -> None:
    # Arrange
    TestPersistenceStubs.setup_news_event_persistence()
    data = TestPersistenceStubs.news_events()
    catalog.write_data(data)

    # Act
    data_usd = catalog.custom_data(cls=NewsEventData, filter_expr=ds.field("currency") == "USD")
    data_chf = catalog.custom_data(cls=NewsEventData, filter_expr=ds.field("currency") == "CHF")

    # Assert
    assert data_usd is not None
    assert data_chf is not None
    assert len(data_usd) == 22941
    assert len(data_chf) == 2745
    assert isinstance(data_chf[0], CustomData)


def test_catalog_bars_querying_by_bar_type(catalog: ParquetDataCatalog) -> None:
    # Arrange
    bar_type = TestDataStubs.bartype_adabtc_binance_1min_last()
    instrument = TestInstrumentProvider.adabtc_binance()
    stub_bars = TestDataStubs.binance_bars_from_csv(
        "ADABTC-1m-2021-11-27.csv",
        bar_type,
        instrument,
    )

    # Act
    catalog.write_data(stub_bars)

    # Assert
    bars = catalog.bars(bar_types=[str(bar_type)])
    all_bars = catalog.bars()
    assert len(all_bars) == 10
    assert len(bars) == len(stub_bars) == 10


def test_catalog_bars_querying_by_instrument_id(catalog: ParquetDataCatalog) -> None:
    # Arrange
    bar_type = TestDataStubs.bartype_adabtc_binance_1min_last()
    instrument = TestInstrumentProvider.adabtc_binance()
    stub_bars = TestDataStubs.binance_bars_from_csv(
        "ADABTC-1m-2021-11-27.csv",
        bar_type,
        instrument,
    )

    # Act
    catalog.write_data(stub_bars)

    # Assert
    bars = catalog.bars(instrument_ids=[instrument.id.value])
    assert len(bars) == len(stub_bars) == 10


def test_catalog_write_pyo3_order_book_depth10(catalog: ParquetDataCatalog) -> None:
    # Arrange
    instrument = TestInstrumentProvider.ethusdt_binance()
    instrument_id = nautilus_pyo3.InstrumentId.from_str(instrument.id.value)
    depth10 = TestDataProviderPyo3.order_book_depth10(instrument_id=instrument_id)

    # Act
    catalog.write_data([depth10] * 100)

    # Assert
    depths = catalog.order_book_depth10(instrument_ids=[instrument.id])
    all_depths = catalog.order_book_depth10()
    assert len(depths) == 100
    assert len(all_depths) == 100


def test_catalog_write_pyo3_quote_ticks(catalog: ParquetDataCatalog) -> None:
    # Arrange
    path = TEST_DATA_DIR / "truefx" / "audusd-ticks.csv"
    df = pd.read_csv(path)
    instrument = TestInstrumentProvider.default_fx_ccy("AUD/USD")
    wrangler = QuoteTickDataWranglerV2.from_instrument(instrument)
    # Data must be sorted as the raw data was not originally sorted
    pyo3_quotes = sorted(wrangler.from_pandas(df), key=lambda x: x.ts_init)

    # Act
    catalog.write_data(pyo3_quotes)

    # Assert
    quotes = catalog.quote_ticks(instrument_ids=[instrument.id])
    all_quotes = catalog.quote_ticks()
    assert len(quotes) == 100_000
    assert len(all_quotes) == 100_000


def test_catalog_write_pyo3_trade_ticks(catalog: ParquetDataCatalog) -> None:
    # Arrange
    path = TEST_DATA_DIR / "binance" / "ethusdt-trades.csv"
    df = pd.read_csv(path)
    instrument = TestInstrumentProvider.ethusdt_binance()
    wrangler = TradeTickDataWranglerV2.from_instrument(instrument)
    pyo3_trades = wrangler.from_pandas(df)

    # Act
    catalog.write_data(pyo3_trades)

    # Assert
    trades = catalog.trade_ticks(instrument_ids=[instrument.id])
    all_trades = catalog.trade_ticks()
    assert len(trades) == 69_806
    assert len(all_trades) == 69_806


def test_catalog_multiple_bar_types(catalog: ParquetDataCatalog) -> None:
    # Arrange
    bar_type1 = TestDataStubs.bartype_adabtc_binance_1min_last()
    instrument1 = TestInstrumentProvider.adabtc_binance()
    stub_bars1 = TestDataStubs.binance_bars_from_csv(
        "ADABTC-1m-2021-11-27.csv",
        bar_type1,
        instrument1,
    )

    bar_type2 = TestDataStubs.bartype_btcusdt_binance_100tick_last()
    instrument2 = TestInstrumentProvider.btcusdt_binance()
    stub_bars2 = TestDataStubs.binance_bars_from_csv(
        "ADABTC-1m-2021-11-27.csv",
        bar_type2,
        instrument2,
    )

    # Act
    catalog.write_data(stub_bars1)
    catalog.write_data(stub_bars2)

    # Assert
    bars1 = catalog.bars(bar_types=[str(bar_type1)])
    bars2 = catalog.bars(bar_types=[str(bar_type2)])
    bars3 = catalog.bars(instrument_ids=[instrument1.id.value])
    all_bars = catalog.bars()
    assert len(bars1) == 10
    assert len(bars2) == 10
    assert len(bars3) == 10
    assert len(all_bars) == 20


def test_catalog_bar_query_instrument_id(
    catalog_betfair: ParquetDataCatalog,
) -> None:
    # Arrange
    bar = TestDataStubs.bar_5decimal()
    catalog_betfair.write_data([bar])

    # Act
    data = catalog_betfair.bars(bar_types=[str(bar.bar_type)])

    # Assert
    assert len(data) == 1


def test_catalog_persists_equity(
    catalog: ParquetDataCatalog,
) -> None:
    # Arrange
    instrument = TestInstrumentProvider.equity()
    quote_tick = TestDataStubs.quote_tick(instrument=instrument)

    # Act
    catalog.write_data([instrument, quote_tick])

    # Assert
    instrument_from_catalog = catalog.instruments(instrument_ids=[instrument.id.value])[0]
    quotes_from_catalog = catalog.quote_ticks(instrument_ids=[instrument.id.value])
    assert instrument_from_catalog == instrument
    assert len(quotes_from_catalog) == 1
    assert quotes_from_catalog[0].instrument_id == instrument.id


def test_list_backtest_runs(
    catalog_betfair: ParquetDataCatalog,
) -> None:
    # Arrange
    mock_folder = f"{catalog_betfair.path}/backtest/abc"
    catalog_betfair.fs.mkdir(mock_folder)

    # Act
    result = catalog_betfair.list_backtest_runs()

    # Assert
    assert result == ["abc"]


def test_list_live_runs(
    catalog_betfair: ParquetDataCatalog,
) -> None:
    # Arrange
    mock_folder = f"{catalog_betfair.path}/live/abc"
    catalog_betfair.fs.mkdir(mock_folder)

    # Act
    result = catalog_betfair.list_live_runs()

    # Assert
    assert result == ["abc"]


# Custom data class for testing metadata functionality
@customdataclass
class TestCustomData(Data):
    value: str = "test"
    number: int = 42


def test_catalog_query_with_static_metadata(catalog: ParquetDataCatalog) -> None:
    """
    Test query method with static (non-callable) metadata.
    """
    # Arrange
    test_data = [
        TestCustomData(value="data1", number=1, ts_event=1, ts_init=1),
        TestCustomData(value="data2", number=2, ts_event=2, ts_init=2),
    ]
    catalog.write_data(test_data)

    static_metadata = {"source": "test", "version": "1.0"}

    # Act
    result = catalog.query(TestCustomData, metadata=static_metadata)

    # Assert
    assert len(result) == 2
    assert all(isinstance(item, CustomData) for item in result)

    # Check that all items have the same static metadata
    for item in result:
        assert item.data_type.metadata == static_metadata
        assert item.data_type.type == TestCustomData


def test_catalog_query_with_callable_metadata(catalog: ParquetDataCatalog) -> None:
    """
    Test query method with callable metadata that generates different metadata per data
    item.
    """
    # Arrange
    test_data = [
        TestCustomData(value="data1", number=1, ts_event=1, ts_init=1),
        TestCustomData(value="data2", number=2, ts_event=2, ts_init=2),
        TestCustomData(value="data3", number=3, ts_event=3, ts_init=3),
    ]
    catalog.write_data(test_data)

    # Define a callable metadata function that generates metadata based on the data
    def metadata_func(data_item):
        return {
            "value": data_item.value,
            "number_category": "even" if data_item.number % 2 == 0 else "odd",
            "timestamp": str(data_item.ts_event),
        }

    # Act
    result = catalog.query(TestCustomData, metadata=metadata_func)

    # Assert
    assert len(result) == 3
    assert all(isinstance(item, CustomData) for item in result)

    # Check that each item has different metadata based on its data
    expected_metadata = [
        {"value": "data1", "number_category": "odd", "timestamp": "1"},
        {"value": "data2", "number_category": "even", "timestamp": "2"},
        {"value": "data3", "number_category": "odd", "timestamp": "3"},
    ]

    for i, item in enumerate(result):
        assert item.data_type.metadata == expected_metadata[i]
        assert item.data_type.type == TestCustomData


def test_catalog_query_without_metadata_parameter(catalog: ParquetDataCatalog) -> None:
    """
    Test query method without metadata parameter (should default to None).
    """
    # Arrange
    test_data = [
        TestCustomData(value="data1", number=1, ts_event=1, ts_init=1),
    ]
    catalog.write_data(test_data)

    # Act
    result = catalog.query(TestCustomData)

    # Assert
    assert len(result) == 1
    assert isinstance(result[0], CustomData)
    assert result[0].data_type.metadata == {}
    assert result[0].data_type.type == TestCustomData
