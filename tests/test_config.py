from pathlib import Path

import pytest
import yaml

from mvs.config import ConfigError, SafetyError, load_speaker, require_disclosure

REPO = Path(__file__).resolve().parents[1]


def test_example_speaker_loads_and_passes_gate():
    spk = load_speaker("example-speaker", REPO)
    assert spk["chain"]["order"] == 2
    assert spk.sources_dir == REPO / "data/example-speaker/audio"
    require_disclosure(spk)


def _write(project: Path, **output):
    cfg = yaml.safe_load((project / "speakers/test-speaker.yaml").read_text())
    cfg["output"].update(output)
    (project / "speakers/test-speaker.yaml").write_text(yaml.safe_dump(cfg))
    return load_speaker("test-speaker", project)


@pytest.mark.parametrize("output, message", [
    ({"disclaimer_enabled": False}, "disclaimer_enabled"),
    ({"metadata_labels": False}, "metadata_labels"),
    ({"disclaimer_every_minutes": 0}, "disclaimer_every_minutes"),
    ({"disclaimer_every_minutes": 30}, "disclaimer_every_minutes"),
    ({"stream_title": "Test Speaker Radio"}, "stream_title"),
    ({"stream_title": "Official Markov remix (not real statements)"}, "official"),
    ({"disclaimer_text": "Enjoy the show."}, "disclaimer_text"),
    ({"blocklist": "config/missing.txt"}, "blocklist"),
])
def test_gate_refuses(project, output, message):
    spk = _write(project, **output)
    with pytest.raises(SafetyError, match=message):
        require_disclosure(spk)


def test_unknown_key_rejected(project):
    cfg = yaml.safe_load((project / "speakers/test-speaker.yaml").read_text())
    cfg["chain"]["ordr"] = 3
    (project / "speakers/test-speaker.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigError, match="chain.ordr"):
        load_speaker("test-speaker", project)


def test_pause_must_stay_under_one_second(project):
    cfg = yaml.safe_load((project / "speakers/test-speaker.yaml").read_text())
    cfg["chain"]["end_token_pause_ms"] = [400, 1500]
    (project / "speakers/test-speaker.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigError, match="end_token_pause_ms"):
        load_speaker("test-speaker", project)
