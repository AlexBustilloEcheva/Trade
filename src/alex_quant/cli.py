import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from alex_quant.config import load_config
from alex_quant.data.alpaca import AlpacaProvider
from alex_quant.data.fixture import FixtureProvider
from alex_quant.data.snapshot import SnapshotProvider
from alex_quant.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alex Quant Lab — investigación reproducible")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Ejecutar StockRanker y generar el informe")
    run.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    run.add_argument("--provider", choices=["alpaca", "fixture", "snapshot"], default="alpaca")
    run.add_argument("--snapshot", type=Path, help="Directorio con input_bars.csv y manifest.json")
    run.add_argument("--output", type=Path, help="Por defecto artifacts/latest")
    args = parser.parse_args(argv)
    load_dotenv(Path.cwd() / ".env", override=False)
    try:
        config = load_config(args.config)
        output = args.output or Path("artifacts/latest")
        if args.provider == "alpaca":
            provider = AlpacaProvider(config.data.feed)
        elif args.provider == "fixture":
            provider = FixtureProvider(config.research.seed)
        else:
            if args.snapshot is None:
                parser.error("--provider snapshot requiere --snapshot RUTA")
            if output.resolve() == args.snapshot.resolve():
                parser.error("El directorio de salida debe ser distinto del snapshot de entrada")
            provider = SnapshotProvider(args.snapshot)
        if args.snapshot is not None and args.provider != "snapshot":
            parser.error("--snapshot solo se admite con --provider snapshot")
        if provider.provenance["synthetic"]:
            print("DATOS SINTÉTICOS DE PRUEBA: no representan resultados de mercado.")
        metrics = run_pipeline(config, provider, output)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Informe: {(output / 'report.md').resolve()}")
    print(
        f"Modelos y resultados guardados; {metrics['stockranker']['rebalance_count']} rebalanceos."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
