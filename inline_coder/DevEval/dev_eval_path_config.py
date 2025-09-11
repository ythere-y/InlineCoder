from pathlib import Path


class PathConfig:
    BenchmarkDir = Path("data/Benchmark/dev_eval")
    OriginDataPath = BenchmarkDir / "processed_clean_data" / "data.jsonl"
