import src.universe as universe


def test_load_universe_falls_back_to_sample_when_csv_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "UNIVERSE_FILE", tmp_path / "does_not_exist.csv")

    result = universe.load_universe()

    assert result[0] == "RELIANCE.NS"
    assert len(result) == len(universe._SAMPLE_UNIVERSE)
    assert all(t.endswith(".NS") for t in result)


def test_load_universe_reads_symbol_column_from_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "nifty500.csv"
    csv_path.write_text("Symbol,Industry\n RELIANCE ,Energy\nTCS,IT\n")
    monkeypatch.setattr(universe, "UNIVERSE_FILE", csv_path)

    result = universe.load_universe()

    assert result == ["RELIANCE.NS", "TCS.NS"]


def test_load_sector_map_returns_empty_dict_when_csv_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "UNIVERSE_FILE", tmp_path / "does_not_exist.csv")

    assert universe.load_sector_map() == {}


def test_load_sector_map_maps_ticker_to_industry(tmp_path, monkeypatch):
    csv_path = tmp_path / "nifty500.csv"
    csv_path.write_text("Symbol,Industry\nRELIANCE,Energy\nTCS,IT\nINFY, IT \n")
    monkeypatch.setattr(universe, "UNIVERSE_FILE", csv_path)

    sector_map = universe.load_sector_map()

    assert sector_map == {"RELIANCE.NS": "Energy", "TCS.NS": "IT", "INFY.NS": "IT"}
